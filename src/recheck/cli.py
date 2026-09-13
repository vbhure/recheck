"""Recheck command line.

    recheck sweep     <directory> [--scripted | --model PROVIDER] [--fresh]
    recheck audit     <letter> --case ID [--scripted | --model PROVIDER] [--fresh]
    recheck resume    --case ID --answer "2=left,3=right"
    recheck show      --case ID
    recheck preflight [--model PROVIDER]

`audit` may stop at a question and exit; `resume` is a DIFFERENT PROCESS that
continues from the Strands session on disk. That is the whole point, so it is
the default way to drive the tool rather than a special mode.

Exit codes:
    0  finished with a result
    2  waiting on a reviewer's answer (the question is persisted)
    3  no result: unreadable document, undetermined result, rejected answer,
       or a case that cannot be used
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import sys

from recheck.case import CaseCorrupt, CaseStore
from recheck.graph import (
    DocumentTooLarge,
    ScannedDocument,
    accepted_answers,
    build_graph,
    interrupt_id,
    open_case,
    outstanding_interrupt,
)
from recheck.report import render
from recheck.sweep import Outcome, outcome_from_case, render_triage, sweep

EXIT_OK = 0
EXIT_AWAITING_HUMAN = 2
EXIT_CANNOT_PROCEED = 3
DEFAULT_STORE = "runs"


def _scripted_factory(fixture: pathlib.Path | None):
    """Build an agent whose model replays a committed classification fixture.

    Two fixture shapes are supported:

      BATCH  {"classifications": [...]} - a fixed answer, for a single letter
             whose conditions are known in advance.
      MAP    {"anatomy": {substring: group}} - answers whatever it is asked,
             which a sweep across many letters needs. Still a fixture, not a
             model: it cannot generalise beyond the substrings it lists.
             Anything it does not list comes back at confidence 0.0, which the
             confidence floor refuses, so an unlisted term is left UNKNOWN
             rather than silently classified.

    Either way the payload goes through Strands' real structured-output path
    and Recheck's real validation; no side is ever taken from a fixture,
    because the schema has no field for one.
    """
    from strands import Agent

    from recheck.classify import SYSTEM_PROMPT
    from recheck.models.scripted import ScriptedModel

    if fixture is None or not fixture.exists():
        payload = {"classifications": []}
    else:
        payload = json.loads(fixture.read_text(encoding="utf-8"))

    anatomy = payload.get("anatomy")
    if anatomy:
        confidence = float(payload.get("confidence", 0.9))

        def responder(_tool, messages):
            text = "".join(block.get("text", "") for block in messages[-1].get("content", []))
            out = []
            for line in text.splitlines():
                if not line.startswith("- "):
                    continue
                name = line[2:].strip()
                match = next(((key, g) for key, g in anatomy.items() if key in name.lower()), None)
                out.append({
                    "condition": name,
                    "extremity_group": match[1] if match else "none",
                    "confidence": confidence if match else 0.0,
                })
            return {"classifications": out}

        def make():
            return Agent(model=ScriptedModel(responder=responder),
                         system_prompt=SYSTEM_PROMPT, callback_handler=None)

        return make

    def make():
        # callback_handler=None suppresses Strands' default stdout handler,
        # which otherwise prints tool chatter into the middle of the report.
        return Agent(model=ScriptedModel(payload=payload), system_prompt=SYSTEM_PROMPT, callback_handler=None)

    return make


def _resolve_factory(args) -> tuple[object | None, str, str]:
    """(agent factory, note to print, classifier label recorded on each case)."""
    if getattr(args, "scripted", False):
        fixture = pathlib.Path(args.classifications) if args.classifications else None
        label = f"scripted: {fixture.as_posix() if fixture else 'empty fixture'}"
        return (_scripted_factory(fixture),
                "zero-model path: AI classifications are replayed from a committed fixture through "
                "Strands' real structured-output path; no network, no inference cost", label)
    if getattr(args, "model", None):
        from recheck.models.factory import build_agent_factory

        factory = build_agent_factory(args.model)
        config = getattr(factory, "config", None)
        detail = config.describe() if config is not None else args.model
        return factory, f"LIVE model provider: {detail}", f"{config.provider}:{config.model_id}" if config else args.model
    return None, ("no classifier configured: terms outside the deterministic lexicon are left "
                  "unknown, never guessed"), "none"


def _quoted(value: object) -> str:
    """A path as it must be typed. The printed commands are meant to be pasted,
    and an unquoted store under a profile such as "C:/Users/Jane Doe" split
    into two arguments. Double quotes work in bash, cmd and PowerShell alike."""
    text = str(value)
    return f'"{text}"' if any(c.isspace() for c in text) else text


def _store_flag(args) -> str:
    return "" if args.store == DEFAULT_STORE else f" --store {_quoted(args.store)}"


# ---------------------------------------------------------------------------
# Running one document
# ---------------------------------------------------------------------------

def _run_audit(store: CaseStore, case_id: str, source: pathlib.Path, factory, classifier: str = "none") -> Outcome:
    """Audit one document. Shared by `audit` and `sweep` so they cannot drift."""
    try:
        open_case(store, case_id, str(source), classifier)
        graph = build_graph(store, case_id, str(source), factory)
        result = asyncio.run(graph.invoke_async(f"audit {source.name}"))
    except (ScannedDocument, DocumentTooLarge, FileNotFoundError, UnicodeDecodeError) as exc:
        return Outcome(case_id, Outcome.FAILED, str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad document must not stop a sweep
        return Outcome(case_id, Outcome.FAILED, f"{type(exc).__name__}: {exc}")

    outcome = outcome_from_case(store, case_id)
    if outcome.state == Outcome.AWAITING_HUMAN:
        outcome.interrupt = result.interrupts[0] if result.interrupts else None
        if outcome.interrupt is None:
            return Outcome(case_id, Outcome.FAILED, "case awaits an answer but the graph raised no question")
    return outcome


def cmd_audit(args) -> int:
    store = CaseStore(args.store)
    source = pathlib.Path(args.letter)
    if not source.is_file():
        print(f"[recheck] no such document: {source}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    if store.exists(args.case):
        if not args.fresh:
            print(f"[recheck] case {args.case} already exists. Use `recheck{_store_flag(args)} show --case "
                  f"{args.case}` to see it, `resume` to answer its question, or add --fresh to "
                  f"discard it and audit again.", file=sys.stderr)
            return EXIT_CANNOT_PROCEED
        store.discard(args.case)

    factory, note, label = _resolve_factory(args)
    print(f"[recheck] {note}")
    outcome = _run_audit(store, args.case, source, factory, label)
    return _finish(args, store, outcome)


def _finish(args, store: CaseStore, outcome: Outcome) -> int:
    if outcome.state == Outcome.FAILED:
        print(f"[recheck] {outcome.detail}", file=sys.stderr)
        if store.exists(outcome.case_id):
            _show(store, outcome.case_id)
        return EXIT_CANNOT_PROCEED
    if outcome.state == Outcome.AWAITING_HUMAN:
        print(question_text(store, outcome.case_id, _store_flag(args), exiting=True))
        return EXIT_AWAITING_HUMAN
    _show(store, outcome.case_id)
    return EXIT_OK if outcome.state == Outcome.COMPLETE else EXIT_CANNOT_PROCEED


def question_text(store: CaseStore, case_id: str, store_flag: str, *, exiting: bool) -> str:
    """The question for the reviewer, rebuilt from committed case state."""
    from recheck.materiality import assess

    case = store.load(case_id)
    decisions = case.load_decisions()
    m = assess(decisions)
    lines = ["", "=" * 74, f"QUESTION FOR THE REVIEWER - case {case_id}", "=" * 74]
    if case.rejected_answer:
        lines += [f"Your previous answer was not accepted: {case.rejected_answer}", ""]
    lines.append("The letter does not establish facts that change this rating.")
    lines.append(f"Depending on the answers it could be {_or(m.possible)}. "
                 f"The letter states {case.stated_combined}%.")
    lines.append("")
    placeholder = []
    for i in m.unknown:
        d = decisions[i]
        where = f", letter {d.evidence}" if d.evidence else ""
        lines.append(f"  [{i}] {d.condition}  ({d.percent}%{where})")
        known = []
        if not d.group_missing:
            known.append(f"extremity group: {d.extremity_group}")
        if not d.side_missing and d.extremity_group != "none":
            known.append(f"side: {d.laterality}")
        if known:
            lines.append(f"      established: {'; '.join(known)}")
        lines.append(f"      not established: {' and '.join(d.missing)} - {d.note or 'not stated in the letter'}")
        if len(m.unknown) == 1:
            for answer, degrees in m.outcomes_for(i).items():
                shown = "/".join(part for part in answer if part not in ("unknown",))
                lines.append(f"        if {shown or 'unknown'}: {_or(degrees)}")
        choices = accepted_answers(d)
        lines.append(f"      answer with one of: {', '.join(choices)}")
        placeholder.append(f"{i}=<{'|'.join(choices)}>")
    lines += ["", "Recheck asks for facts only. It never asks for a percentage or a combined evaluation.", ""]
    if exiting:
        lines.append(f"Process {os.getpid()} is exiting. The Strands session in "
                     f"{store.session_dir(case_id)} holds the open question.")
    lines.append("Answer from any later process:")
    lines.append(f"  recheck{store_flag} resume --case {case_id} --answer \"{','.join(placeholder)}\"")
    lines.append("=" * 74)
    return "\n".join(lines)


def _or(values) -> str:
    items = [f"{v}%" for v in values]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _answer_payload(raw: str):
    """"2=left,3=right" -> {"2": "left", "3": "right"}, so the persisted interrupt
    response is structured. Anything that does not parse is passed through as
    text, and the assess node rejects it with a specific reason."""
    clauses = [c.strip() for c in raw.replace(";", ",").split(",") if c.strip()]
    if not clauses or any(c.count("=") != 1 for c in clauses):
        return raw
    keys = [c.split("=", 1)[0].strip() for c in clauses]
    if len(set(keys)) != len(keys):
        return raw  # duplicates are rejected by the node, with the index named
    return {k: c.split("=", 1)[1].strip() for k, c in zip(keys, clauses)}


def cmd_resume(args) -> int:
    store = CaseStore(args.store)
    try:
        case = store.load(args.case)
    except (FileNotFoundError, CaseCorrupt) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    if case.status != "awaiting_human":
        answered = f" (answers on file: {case.human_answers})" if case.human_answers else ""
        print(f"[recheck] case {args.case} is not waiting on an answer; its status is "
              f"{case.status!r}{answered}. Nothing was changed. To audit it again, run "
              f"`recheck{_store_flag(args)} audit {_quoted(case.source_path)} --case {args.case} --fresh`.",
              file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    try:
        graph = build_graph(store, args.case, case.source_path, None)
    except Exception as exc:  # noqa: BLE001 - the persisted session is untrusted input
        print(f"[recheck] the Strands session for case {args.case} could not be restored "
              f"({type(exc).__name__}). The case is corrupt; re-audit it with --fresh.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    pending = outstanding_interrupt(graph)
    if pending is None or pending.id != interrupt_id(args.case):
        print(f"[recheck] case {args.case} says it is waiting on an answer, but its Strands session "
              f"holds no open question (was {store.session_dir(args.case)} deleted?). "
              f"Nothing to resume; re-audit it with --fresh.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    print(f"[recheck] process {os.getpid()}: restored case {args.case} from its Strands session; "
          f"continuing at node 'assess'")
    try:
        asyncio.run(graph.invoke_async(
            [{"interruptResponse": {"interruptId": pending.id, "response": _answer_payload(args.answer)}}]
        ))
    except Exception as exc:  # noqa: BLE001
        print(f"[recheck] resume failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    outcome = outcome_from_case(store, args.case)
    if outcome.state == Outcome.AWAITING_HUMAN:
        # The answer was rejected; the same question is open again.
        print(question_text(store, args.case, _store_flag(args), exiting=True))
        return EXIT_CANNOT_PROCEED
    return _finish(args, store, outcome)


def cmd_sweep(args) -> int:
    store = CaseStore(args.store)
    directory = pathlib.Path(args.directory)
    if not directory.is_dir():
        print(f"[recheck] not a directory: {directory}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    from recheck.sweep import documents_in

    documents = documents_in(directory)
    if not documents:
        print(f"[recheck] no .txt or .pdf documents in {directory}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    factory, note, label = _resolve_factory(args)
    print(f"[recheck] sweeping {len(documents)} document(s) from {directory.as_posix()}")
    print(f"[recheck] {note}")
    print()

    def run(store_, case_id, path, factory_):
        return _run_audit(store_, case_id, path, factory_, label)

    outcomes, code = sweep(store, directory, factory, run, fresh=args.fresh)
    print(render_triage(store, outcomes, store_flag=_store_flag(args)))
    return code


def cmd_preflight(args) -> int:
    """Validate provider configuration without making an inference call."""
    from recheck.models.factory import ProviderNotConfigured, load_config, preflight

    try:
        config = load_config(getattr(args, "model", None))
    except ProviderNotConfigured as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    print(f"[recheck] preflight for {config.describe()}")
    print("[recheck] no inference call is made by this command")
    print()
    failed = 0
    for check in preflight(config):
        mark = "PASS" if check.ok else "FAIL"
        failed += not check.ok
        print(f"  [{mark}]  {check.name:26}  {check.detail}")
    print()
    if failed:
        print(f"[recheck] {failed} check(s) failed; live inference would not succeed.")
        print("[recheck] the zero-cost path is unaffected: use --scripted.")
        return EXIT_CANNOT_PROCEED
    print("[recheck] configuration is ready. A live run WILL consume provider credits.")
    return EXIT_OK


def cmd_show(args) -> int:
    store = CaseStore(args.store)
    try:
        case = store.load(args.case)
        _show(store, args.case)
        if case.status == "awaiting_human":
            print(question_text(store, args.case, _store_flag(args), exiting=False))
    except (FileNotFoundError, CaseCorrupt) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    return EXIT_OK


def _show(store: CaseStore, case_id: str) -> None:
    print()
    print(render(store.load(case_id)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recheck", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=DEFAULT_STORE, help="case store directory (default: runs)")
    parser.add_argument("--debug", action="store_true",
                        help="show framework logging (Strands node/graph internals), normally suppressed")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_flags(p):
        p.add_argument("--scripted", action="store_true",
                       help="replay AI classifications from a fixture: no network, no cost, real Strands code path")
        p.add_argument("--classifications", default=None, help="JSON fixture for --scripted")
        p.add_argument("--model", default=None, help="live provider id (bedrock, anthropic, ollama)")
        p.add_argument("--fresh", action="store_true", help="discard any existing case and audit again")

    audit = sub.add_parser("audit", help="audit one decision letter")
    audit.add_argument("letter")
    audit.add_argument("--case", required=True)
    add_model_flags(audit)
    audit.set_defaults(func=cmd_audit)

    resume = sub.add_parser("resume", help="answer a case's question, in a new process")
    resume.add_argument("--case", required=True)
    resume.add_argument("--answer", required=True,
                        help='facts only, e.g. "2=left,3=right" or "4=upper-left"')
    resume.set_defaults(func=cmd_resume)

    sweep_p = sub.add_parser("sweep", help="audit a directory of letters unattended; report only what needs you")
    sweep_p.add_argument("directory")
    add_model_flags(sweep_p)
    sweep_p.set_defaults(func=cmd_sweep)

    show = sub.add_parser("show", help="print a stored case report")
    show.add_argument("--case", required=True)
    show.set_defaults(func=cmd_show)

    pre = sub.add_parser("preflight", help="check live-provider configuration WITHOUT an inference call")
    pre.add_argument("--model", default=None, help="provider id; defaults to RECHECK_PROVIDER")
    pre.set_defaults(func=cmd_preflight)
    return parser


def _configure_logging(debug: bool) -> None:
    """Keep framework logging out of product output.

    Strands logs node and graph failures at ERROR, so a deliberate refusal -
    an oversized document, a scanned PDF - printed a stack of "node failed /
    graph execution failed" lines underneath Recheck's own one-line
    explanation. The refusal is the product behaving correctly; the framework
    trace is noise that makes it look like a crash. --debug restores it.
    """
    logging.basicConfig(level=logging.DEBUG if debug else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not debug:
        for name in ("strands", "botocore", "urllib3", "httpx", "anthropic"):
            logging.getLogger(name).setLevel(logging.CRITICAL)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "debug", False))
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    except Exception as exc:  # provider misconfiguration, unreadable case, etc.
        from recheck.models.factory import ProviderError

        if isinstance(exc, ProviderError):
            print(f"[recheck] {exc}", file=sys.stderr)
            return EXIT_CANNOT_PROCEED
        raise


if __name__ == "__main__":
    raise SystemExit(main())
