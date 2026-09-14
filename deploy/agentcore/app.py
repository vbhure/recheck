"""Recheck on Amazon Bedrock AgentCore Runtime.

One invocation audits one decision letter through Recheck's real pipeline -
the Strands Graph the CLI runs (extract -> classify -> assess -> compute) -
and returns the result as JSON. A later invocation on the SAME runtime
session, naming the same case_id and carrying answers, resumes the case from
its Strands session, through the same code `recheck resume` runs.

Payload (JSON object, no other keys):

    {"letter_text": "...",            # required to audit; the letter as UTF-8 text
     "case_id": "rc-1",               # optional; generated when absent
     "answers": "1=left,2=right"}     # optional; present = resume case_id

Nothing here re-implements Recheck. The audit goes through
recheck.cli._run_audit (the function `audit` and `sweep` share), the classifier
through recheck.cli._resolve_factory or recheck.models.factory.build_agent_factory
(the CLI's own provider seam), the resume through recheck.cli.cmd_resume, and
the report through recheck.report.render.

Environment:

    RECHECK_BEDROCK_MODEL_ID     Bedrock model or inference profile id (else RECHECK_MODEL_ID,
                                 else the factory's default)
    AWS_REGION / RECHECK_REGION  region for the Bedrock call (AgentCore sets AWS_REGION)
    RECHECK_AGENTCORE_SCRIPTED   path to a --classifications fixture: use Recheck's zero-model
                                 path instead of Bedrock (local verification; no network)
    RECHECK_AGENTCORE_STORE      case store root (default /tmp/recheck-agentcore)

State lives under the store root inside the session's microVM. It survives
between invocations only while AgentCore keeps that session's microVM alive
(idle timeout, maximum lifetime); after that a resume is refused and the
letter must be audited again. Nothing is written to a database.

The letter text is never logged. Only the case id, the mode, the outcome and
the letter's size in bytes are.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import logging
import os
import pathlib
import threading
import uuid
from typing import Any

from recheck import cli
from recheck.case import CaseBusy, CaseCorrupt, CaseStore
from recheck.graph import accepted_answers, asked
from recheck.provenance import Actor
from recheck.report import render, verdict

try:  # the runtime SDK is needed to serve, not to exercise the handler
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
except ImportError:  # pragma: no cover - exercised only where the SDK is absent
    BedrockAgentCoreApp = None  # type: ignore[assignment,misc]

log = logging.getLogger("recheck.agentcore")

#: A decision letter is a few pages; Recheck's own text cap is 500,000 characters.
#: The HTTP server has already parsed the body when this check runs, so this
#: bounds the work Recheck does, not the bytes AgentCore accepts.
MAX_LETTER_BYTES = 256 * 1024
MAX_ANSWERS_CHARS = 2_000
ALLOWED_KEYS = frozenset({"letter_text", "case_id", "answers"})
DEFAULT_STORE = "/tmp/recheck-agentcore"
REPORTED_STATUSES = ("complete", "awaiting_human", "undetermined", "unparsed")

# cmd_resume prints its report and refusals; they are captured, and
# redirecting stdout is process-wide, so one resume captures at a time.
_STDIO_LOCK = threading.Lock()


class Rejected(ValueError):
    """The payload is refused before any case is touched."""


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def _validate(payload: Any) -> tuple[str | None, str | None, str | None]:
    """(letter_text, case_id, answers), or Rejected."""
    if not isinstance(payload, dict):
        raise Rejected("the payload must be a JSON object")
    unknown = sorted(str(k) for k in payload if k not in ALLOWED_KEYS)
    if unknown:
        raise Rejected(f"unknown payload keys: {', '.join(unknown)}; allowed: {', '.join(sorted(ALLOWED_KEYS))}")
    letter, case_id, answers = payload.get("letter_text"), payload.get("case_id"), payload.get("answers")
    if letter is not None:
        if not isinstance(letter, str):
            raise Rejected("letter_text must be a string")
        size = len(letter.encode("utf-8", "surrogatepass"))
        if size > MAX_LETTER_BYTES:
            raise Rejected(f"letter_text is {size:,} bytes, over the {MAX_LETTER_BYTES:,}-byte limit; "
                           f"a decision letter is a few pages")
        if not letter.strip():
            raise Rejected("letter_text is empty")
    if case_id is not None:
        if not isinstance(case_id, str):
            raise Rejected("case_id must be a string")
        try:
            CaseStore("unused").dir_for(case_id)  # Recheck's own case id rules
        except ValueError as exc:
            raise Rejected(str(exc)) from None
    if answers is not None:
        if not isinstance(answers, str) or not answers.strip():
            raise Rejected('answers must be a non-empty string such as "1=left,2=right"')
        if len(answers) > MAX_ANSWERS_CHARS:
            raise Rejected(f"answers is over {MAX_ANSWERS_CHARS:,} characters")
        if case_id is None:
            raise Rejected("answers need the case_id of the case they answer")
    elif letter is None:
        raise Rejected("send letter_text to audit a letter, or case_id and answers to resume a case")
    return letter, case_id, answers


# ---------------------------------------------------------------------------
# Store and classifier
# ---------------------------------------------------------------------------

def _session_root(session_id: str | None) -> pathlib.Path:
    """One directory per runtime session. The session id is hashed, never used as a path."""
    root = pathlib.Path(os.environ.get("RECHECK_AGENTCORE_STORE") or DEFAULT_STORE)
    tag = hashlib.sha256((session_id or "no-session").encode("utf-8")).hexdigest()[:12]
    return root / tag


def _resolve_classifier() -> tuple[Any, str, dict[str, Any]]:
    """(agent factory, classifier label recorded on the case, model description)."""
    fixture = os.environ.get("RECHECK_AGENTCORE_SCRIPTED")
    if fixture:
        factory, _note, label = cli._resolve_factory(
            argparse.Namespace(scripted=True, classifications=fixture, model=None))
        return factory, label, {"provider": "scripted", "model_id": None}

    from recheck.models.factory import build_agent_factory

    env = dict(os.environ)
    if env.get("RECHECK_BEDROCK_MODEL_ID"):
        env["RECHECK_MODEL_ID"] = env["RECHECK_BEDROCK_MODEL_ID"]
    # Runs Recheck's preflight: raises ProviderNotConfigured before any call.
    factory = build_agent_factory("bedrock", env)
    config = factory.config  # type: ignore[attr-defined]
    return factory, f"{config.provider}:{config.model_id}", {
        "provider": config.provider, "model_id": config.model_id, "region": config.region}


def _counted(factory: Any) -> tuple[Any, list[int]]:
    """The factory, recording each time classify builds an agent to call.

    recheck.classify builds the agent only when it sends names to the model,
    once per letter, so a recorded build is a model call attempted.
    """
    calls: list[int] = []

    def make() -> Any:
        calls.append(1)
        return factory()

    for attr in ("config", "timeout_s"):
        if hasattr(factory, attr):
            setattr(make, attr, getattr(factory, attr))
    return make, calls


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

def _who(actor: Actor | None, *, side: bool, scripted: bool) -> str | None:
    if actor is None:
        return None
    if actor is Actor.DETERMINISTIC:
        return "letter" if side else "lexicon"
    if actor is Actor.AI:
        return "AI (replayed fixture)" if scripted else "AI"
    return "reviewer"


def _model_from_label(label: str) -> dict[str, Any]:
    if label.startswith("scripted"):
        return {"provider": "scripted", "model_id": None}
    provider, _, model_id = label.partition(":")
    return {"provider": provider or None, "model_id": model_id or None}


def _result(store: CaseStore, case_id: str, *, model: dict[str, Any], model_called: bool,
            detail: str | None = None) -> dict[str, Any]:
    case = store.load(case_id)
    decisions = case.load_decisions()
    scripted = (case.classifier or "").startswith("scripted")
    headline, explanation = verdict(case)
    status = case.status if case.status in REPORTED_STATUSES else "error"

    question = None
    if case.status == "awaiting_human":
        indexes = asked(decisions)
        # The CLI's question, without its closing `recheck resume` command line:
        # here the answer goes back through the runtime (how_to_answer).
        text = cli.question_text(store, case_id, "", exiting=False)
        text = text.split("\nAnswer from any later process:")[0].rstrip()
        question = {
            "text": text,
            "answer_format": ",".join(f"{i}=<{'|'.join(accepted_answers(decisions[i]))}>" for i in indexes),
            "items": [{"index": i, "condition": decisions[i].condition, "percent": decisions[i].percent,
                       "not_established": list(decisions[i].missing),
                       "choices": accepted_answers(decisions[i])} for i in indexes],
            "how_to_answer": ("invoke again on the SAME runtime session with "
                              f'{{"case_id": "{case_id}", "answers": "<index>=<choice>,..."}}'),
            "rejected_answer": case.rejected_answer,
        }

    ratings = [{
        "index": i,
        "condition": d.condition,
        "percent": d.percent,
        "extremity_group": d.extremity_group,
        "extremity_group_decided_by": _who(d.group_by, side=False, scripted=scripted),
        "side": d.laterality if d.extremity_group != "none" else None,
        "side_decided_by": _who(d.side_by, side=True, scripted=scripted) if d.extremity_group != "none" else None,
        "confidence": d.confidence,
    } for i, d in enumerate(decisions)]

    return {
        "status": status,
        "case_id": case.case_id,
        "stated_combined": case.stated_combined,
        "recomputed_degree": case.recomputed_degree if case.status == "complete" else None,
        "recomputed_combined": case.recomputed_combined if case.status == "complete" else None,
        "bilateral_applied": case.bilateral_applied if case.status == "complete" else None,
        "verdict": headline,
        "verdict_detail": explanation,
        "possible_degrees": list(case.possible_degrees),
        "question": question,
        "ratings": ratings,
        "reviewer_answers": dict(case.human_answers),
        "model": {**model, "classifier": case.classifier, "model_call_made": model_called},
        "detail": detail,
        "report": render(case, show_trace=False),
    }


def _refusal(status: str, message: str, case_id: str | None = None) -> dict[str, Any]:
    return {"status": status, "case_id": case_id, "error": message}


# ---------------------------------------------------------------------------
# Audit and resume
# ---------------------------------------------------------------------------

def _audit(root: pathlib.Path, case_id: str, letter: str) -> dict[str, Any]:
    store = CaseStore(root / "cases")
    with store.lock(case_id):
        if store.exists(case_id):
            return _refusal("rejected", f"case {case_id} already exists in this runtime session; send "
                            f"case_id with answers to resume it, or audit under a new case_id", case_id)
        # Resolved before the letter is written: a provider that is not
        # ready refuses here, as `recheck audit --model` does.
        factory, label, model = _resolve_classifier()
        letters = root / "letters"
        letters.mkdir(parents=True, exist_ok=True)
        source = letters / f"{case_id}.txt"
        source.write_bytes(letter.encode("utf-8", "surrogatepass"))
        make, calls = _counted(factory)
        outcome = cli._run_audit(store, case_id, source, make, label)
    if not store.exists(case_id):
        return {**_refusal("error", outcome.detail or "the audit stopped before a case was written", case_id),
                "model": {**model, "model_call_made": bool(calls)}}
    detail = outcome.detail if outcome.state == outcome.FAILED else None
    return _result(store, case_id, model=model, model_called=bool(calls), detail=detail)


def _resume(root: pathlib.Path, case_id: str, answers: str, letter: str | None) -> dict[str, Any]:
    store = CaseStore(root / "cases")
    if not store.exists(case_id):
        return _refusal("rejected", f"no case {case_id} in this runtime session. Cases live in the session's "
                        f"microVM and end with it (idle timeout or maximum lifetime); audit the letter again",
                        case_id)
    case = store.load(case_id)
    if letter is not None and hashlib.sha256(letter.encode("utf-8", "surrogatepass")).hexdigest() != case.document_sha256:
        return _refusal("rejected", f"letter_text differs from the letter case {case_id} audited; send the "
                        f"answers without letter_text, or audit the new letter under a new case_id", case_id)
    args = argparse.Namespace(store=str(store.root), case=case_id, answer=answers, brief=True)
    out, err = io.StringIO(), io.StringIO()
    with _STDIO_LOCK, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.cmd_resume(args)  # the CLI's resume: lock, checks, Strands interrupt response, recovery
    # Recheck's own refusals only; framework warnings on stderr are not product output.
    refusal = " ".join(line.removeprefix("[recheck] ").strip() for line in err.getvalue().splitlines()
                       if line.startswith("[recheck] ")) or None
    # A resume never calls the model: classification happened at audit.
    return _result(store, case_id, model=_model_from_label(case.classifier or ""), model_called=False,
                   detail=refusal if code != cli.EXIT_OK else None)


def handle(payload: Any, session_id: str | None = None) -> dict[str, Any]:
    """The entrypoint's work, callable without a server."""
    try:
        letter, case_id, answers = _validate(payload)
    except Rejected as exc:
        log.info("invocation rejected before any case was touched")
        return _refusal("rejected", str(exc), payload.get("case_id") if isinstance(payload, dict)
                        and isinstance(payload.get("case_id"), str) else None)

    root = _session_root(session_id)
    mode = "resume" if answers is not None else "audit"
    case_id = case_id or f"rc-{uuid.uuid4().hex[:12]}"
    try:
        if answers is not None:
            result = _resume(root, case_id, answers, letter)
        else:
            result = _audit(root, case_id, letter)  # type: ignore[arg-type]
    except CaseBusy as exc:
        result = _refusal("busy", str(exc), case_id)
    except CaseCorrupt as exc:
        result = _refusal("error", f"case {case_id} cannot be trusted: {exc}", case_id)
    except ValueError as exc:  # a fixture that does not load, a store path too long
        result = _refusal("error", str(exc), case_id)
    except Exception as exc:  # noqa: BLE001 - a provider not ready, a store that cannot be written
        from recheck.models.factory import ProviderError

        if not isinstance(exc, (ProviderError, OSError)):
            log.exception("invocation failed: case=%s mode=%s", case_id, mode)
            result = _refusal("error", f"{type(exc).__name__}: internal error", case_id)
        else:
            result = _refusal("error", f"{type(exc).__name__}: {exc}", case_id)
    log.info("invocation: case=%s mode=%s status=%s letter_bytes=%s model_call=%s", case_id, mode,
             result.get("status"), len(letter.encode("utf-8", "surrogatepass")) if letter else 0,
             (result.get("model") or {}).get("model_call_made"))
    return result


app = BedrockAgentCoreApp() if BedrockAgentCoreApp is not None else None


def invoke(payload: Any, context: Any = None) -> dict[str, Any]:
    """AgentCore entrypoint. The runtime session id scopes the case store."""
    return handle(payload, session_id=getattr(context, "session_id", None))


if app is not None:
    invoke = app.entrypoint(invoke)


if __name__ == "__main__":
    # As the CLI does without --debug: framework and HTTP logging stay out of
    # the output, and credential-shaped text is masked in what remains.
    cli._configure_logging(False)
    log.setLevel(logging.INFO)
    # In the runtime, allow botocore to take credentials from the metadata
    # endpoint if no other source supplies them. Recheck's preflight skips
    # instance metadata unless this is "false" (see recheck.models.factory).
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "false")
    if app is None:
        raise SystemExit("bedrock-agentcore is not installed: pip install -r deploy/agentcore/requirements.txt")
    app.run()
