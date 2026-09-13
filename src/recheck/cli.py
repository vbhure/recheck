"""Recheck command line.

    recheck sweep     <directory> [--scripted | --model PROVIDER] [--fresh]
    recheck audit     <letter> --case ID [--scripted | --model PROVIDER] [--fresh]
    recheck resume    --case ID --answer "2=left,3=right"
    recheck resume    --case ID          (finish a case whose answers are on file)
    recheck show      --case ID
    recheck preflight [--model PROVIDER]

`audit` may stop at a question and exit; `resume` is a DIFFERENT PROCESS that
continues from the Strands session on disk. That is the whole point, so it is
the default way to drive the tool rather than a special mode.

Exit codes:
    0  finished with a result
    2  waiting on a reviewer's answer (the question is persisted)
    3  no result: unreadable document, undetermined result, rejected answer,
       a case that cannot be used or is busy in another process, or a
       malformed command line
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import re
import sys

from recheck.case import CaseBusy, CaseCorrupt, CaseStore
from recheck.graph import (
    ComputeNode,
    DocumentTooLarge,
    ScannedDocument,
    accepted_answers,
    asked,
    build_graph,
    open_case,
)
from recheck.report import render
from recheck.sweep import Outcome, open_question, outcome_from_case, question_is_open, render_triage, sweep

EXIT_OK = 0
EXIT_AWAITING_HUMAN = 2
EXIT_CANNOT_PROCEED = 3
DEFAULT_STORE = "runs"


#: Wording that joins two conditions in one name, for the MAP fixture below:
#: a spaced dash, a slash, "&", "+", ";", "with" and "and".
_JOINED_CONDITIONS = re.compile(r"\s[-–—]+\s|[/\\&+;]|\b(?:with|and)\b", re.I)


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
    from recheck.extract.deterministic import primary_clause
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
                # Only the rated condition is matched. Matched over the whole
                # name, "Left Lisfranc injury, secondary to de Quervain
                # tenosynovitis" was classified "upper" from its LINKED
                # condition - a foot injury joined the arms' bilateral factor
                # and a 10-point-higher figure was reported. The lexicon
                # strips linked clauses for the same reason. A name whose
                # rated condition lists terms of both groups is ambiguous and
                # is not classified either.
                #
                # Nor is a name that joins two conditions (" - ", "/", "with",
                # "and", ...): primary_clause strips only linked-condition
                # wording, and "Left Lisfranc injury - cubital tunnel
                # syndrome" was classified "upper" from the term after the
                # dash. Which side of the join is the rated condition is not
                # something a substring map can know, so a listed term on
                # either side leaves the condition unclassified.
                primary = primary_clause(name).lower()
                groups = {group for key, group in anatomy.items() if key in primary}
                joined = _JOINED_CONDITIONS.search(primary) is not None
                match = groups.pop() if len(groups) == 1 and not joined else None
                out.append({
                    "condition": name,
                    "extremity_group": match or "none",
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


# ---------------------------------------------------------------------------
# Printed commands
# ---------------------------------------------------------------------------

# Characters no shell in use here (bash, zsh, PowerShell, cmd) treats
# specially in an argument: word characters, "." "/" ":" "-". A path of only
# these, plus spaces inside double quotes, is printed as it is.
_SHELL_SAFE = re.compile(r"[\w./:-]+")


def _shell_path(value: object, placeholder: str) -> tuple[str, str]:
    """A path as it must be typed in a printed command: (token, note).

    The commands are meant to be pasted, and the paths in them come from the
    command line and from FILE NAMES, which are untrusted. Double quotes, the
    old quoting, stop word splitting and nothing else: a letter named
    'a$(touch PWNED).txt' printed a command that ran `touch PWNED` in bash
    and PowerShell when pasted, and a store 'rt/R&D' backgrounded half the
    command. No single quoting is literal in bash, PowerShell and cmd alike,
    so a path with any character a shell interprets is NOT printed in the
    command: a placeholder stands in, and the note names the path.

    Paths are printed with forward slashes, which Windows accepts: a stored
    "fixtures\\letters\\06_agrees.txt" pasted into Git Bash became
    "fixturesletters06_agrees.txt".
    """
    text = pathlib.PurePath(str(value)).as_posix()
    if text.startswith("-"):
        text = "./" + text  # otherwise parsed as an option
    if _SHELL_SAFE.fullmatch(text):
        return text, ""
    if _SHELL_SAFE.fullmatch(text.replace(" ", "")):
        return f'"{text}"', ""
    return placeholder, (f"{placeholder} stands for {text!r}, left out of the command because a shell "
                         f"would interpret some of its characters.")


def _shown_path(value: object) -> str:
    """A path mentioned in a sentence rather than in a command."""
    token, note = _shell_path(value, "")
    return token if not note else repr(pathlib.PurePath(str(value)).as_posix())


def _store_flag(args) -> str:
    return "" if args.store == DEFAULT_STORE else f" --store {_shell_path(args.store, 'STORE')[0]}"


def _store_note(args) -> str:
    return "" if args.store == DEFAULT_STORE else _shell_path(args.store, "STORE")[1]


def _audit_again(args, source_path: str, case_id: str) -> str:
    """The re-audit command, with a note when a path could not be printed in it."""
    letter, letter_note = _shell_path(source_path, "LETTER")
    notes = " ".join(n for n in (_store_note(args), letter_note) if n)
    command = f"`recheck{_store_flag(args)} audit {letter} --case {case_id} --fresh`"
    return command + (f" ({notes})" if notes else "")


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
        print(f"[recheck] no such document: {_shown_path(source)}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    try:
        with store.lock(args.case):
            if store.exists(args.case):
                if not args.fresh:
                    print(f"[recheck] case {args.case} already exists. Use `recheck{_store_flag(args)} show --case "
                          f"{args.case}` to see it, `resume` to answer its question, or add --fresh to "
                          f"discard it and audit again.", file=sys.stderr)
                    return EXIT_CANNOT_PROCEED
                try:
                    store.discard(args.case)
                except OSError as exc:
                    print(f"[recheck] could not discard case {args.case}: {exc}. Nothing was deleted.",
                          file=sys.stderr)
                    return EXIT_CANNOT_PROCEED

            factory, note, label = _resolve_factory(args)
            print(f"[recheck] {note}")
            outcome = _run_audit(store, args.case, source, factory, label)
            return _finish(args, store, outcome)
    except CaseBusy as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED


def _finish(args, store: CaseStore, outcome: Outcome) -> int:
    # A case that cannot be read back is a refusal, not a traceback: a
    # corrupt case.json here escaped as CaseCorrupt with exit 1.
    try:
        if outcome.state == Outcome.FAILED:
            print(f"[recheck] {outcome.detail}", file=sys.stderr)
            if store.exists(outcome.case_id):
                _show(store, outcome.case_id, brief=getattr(args, "brief", False))
            return EXIT_CANNOT_PROCEED
        if outcome.state == Outcome.AWAITING_HUMAN:
            print(question_text(store, outcome.case_id, _store_flag(args), exiting=True,
                                store_note=_store_note(args)))
            return EXIT_AWAITING_HUMAN
        _show(store, outcome.case_id, brief=getattr(args, "brief", False))
    except (FileNotFoundError, CaseCorrupt) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    return EXIT_OK if outcome.state == Outcome.COMPLETE else EXIT_CANNOT_PROCEED


def question_text(store: CaseStore, case_id: str, store_flag: str, *, exiting: bool, store_note: str = "") -> str:
    """The question for the reviewer, rebuilt from committed case state."""
    from recheck.materiality import assess

    case = store.load(case_id)
    decisions = case.load_decisions()
    m = assess(decisions)
    lines = ["", "=" * 74, f"QUESTION FOR THE REVIEWER - case {case_id}", "=" * 74]
    if case.rejected_answer:
        lines += [f"Your previous answer was not accepted: {case.rejected_answer}", ""]
    # "does not establish facts that change this rating" read as "nothing
    # changes the rating", directly above a line saying it could be 70% or 80%.
    lines.append("The letter leaves out facts that could change this rating.")
    if m.possible:
        lines.append(f"Depending on the answers it could be {_or(m.possible)}. "
                     f"The letter states {case.stated_combined}%.")
    else:
        lines.append(f"The ratings the answers could lead to could not be enumerated: too many facts are "
                     f"unknown. The letter states {case.stated_combined}%.")
    lines.append("")
    placeholder = []
    questions = asked(decisions)
    for i in questions:
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
        if len(questions) == 1:
            for answer, degrees in m.outcomes_for(i).items():
                shown = "/".join(part for part in answer if part not in ("unknown",))
                lines.append(f"        if {shown or 'unknown'}: {_or(degrees)}")
        choices = accepted_answers(d)
        lines.append(f"      answer with one of: {', '.join(choices)}")
        placeholder.append(f"{i}=<{'|'.join(choices)}>")
    lines += ["", "Recheck asks for facts only. It never asks for a percentage or a combined evaluation.", ""]
    if exiting:
        lines.append(f"Process {os.getpid()} is exiting. The Strands session in "
                     f"{_shown_path(store.session_dir(case_id))} holds the open question.")
    lines.append("Answer from any later process:")
    lines.append(f"  recheck{store_flag} resume --case {case_id} --answer \"{','.join(placeholder)}\"")
    if store_note:
        lines.append(f"  ({store_note})")
    lines.append("=" * 74)
    return "\n".join(lines)


def _or(values) -> str:
    items = [f"{v}%" for v in values]
    if not items:
        # An empty set (an enumeration over its limit) raised IndexError here.
        return "not established"
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
        # Read once before locking so that a missing case is reported without
        # creating anything in a store that may not exist. From here on the
        # id is spelled as the case records it: on a case-insensitive
        # filesystem `--case pair` reaches a case audited as PAIR, and its
        # question is keyed by that spelling.
        args.case = store.load(args.case).case_id
        with store.lock(args.case):
            return _resume_locked(args, store)
    except (FileNotFoundError, CaseCorrupt, CaseBusy) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED


def _resume_locked(args, store: CaseStore) -> int:
    case = store.load(args.case)
    if case.letter_changed():
        # resume never re-reads the letter. After the letter was edited to
        # state a different figure and name a side, an answer was accepted for
        # a condition the file no longer described and the report compared
        # against a figure the file no longer stated.
        print(f"[recheck] the letter for case {args.case}, {_shown_path(case.source_path)}, has changed since "
              f"it was audited; this case describes the earlier version. Nothing was changed. To audit the "
              f"letter as it is now, run {_audit_again(args, case.source_path, args.case)}.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    if case.status == "ready":
        return _finish_ready(args, store, case)
    if case.status != "awaiting_human":
        answered = f" (answers on file: {case.human_answers})" if case.human_answers else ""
        print(f"[recheck] case {args.case} is not waiting on an answer; its status is "
              f"{case.status!r}{answered}. Nothing was changed. To audit it again, run "
              f"{_audit_again(args, case.source_path, args.case)}.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    if args.answer is None:
        print(f"[recheck] case {args.case} is waiting on an answer; pass it with --answer. "
              f"Use `recheck{_store_flag(args)} show --case {args.case}` to see the question.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    try:
        graph = build_graph(store, args.case, case.source_path, None)
    except Exception as exc:  # noqa: BLE001 - the persisted session is untrusted input
        print(f"[recheck] the Strands session for case {args.case} could not be restored "
              f"({type(exc).__name__}). The case is corrupt; re-audit it with --fresh.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    pending = open_question(graph, args.case)
    if pending is None:
        print(f"[recheck] case {args.case} says it is waiting on an answer, but its Strands session in "
              f"{_shown_path(store.session_dir(args.case))} holds no open question (it may have been deleted, "
              f"or a run stopped part-way). Nothing to resume; re-audit it with --fresh.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    try:
        snapshot = store.snapshot(args.case)
    except OSError as exc:
        print(f"[recheck] could not read case {args.case} before resuming it ({type(exc).__name__}: {exc}). "
              f"Nothing was changed.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    print(f"[recheck] process {os.getpid()}: restored case {args.case} from its Strands session; "
          f"continuing at node 'assess'")
    try:
        asyncio.run(graph.invoke_async(
            [{"interruptResponse": {"interruptId": pending.id, "response": _answer_payload(args.answer)}}]
        ))
    except BaseException as exc:
        kept = _recover_failed_resume(args, store, snapshot, case.source_path)
        if not isinstance(exc, Exception):
            print(f"[recheck] resume interrupted ({type(exc).__name__}). {kept}", file=sys.stderr)
            raise
        print(f"[recheck] resume failed: {type(exc).__name__}: {exc}. {kept}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    outcome = outcome_from_case(store, args.case)
    if outcome.state == Outcome.AWAITING_HUMAN:
        # Normally the answer was rejected and the same question is open again.
        # If the session no longer holds that question (a tampered session the
        # compute node refused), printing it would hand out a command that
        # cannot work, so say what can.
        try:
            reopened = open_question(build_graph(store, args.case, case.source_path, None), args.case)
        except Exception:  # noqa: BLE001
            reopened = None
        if reopened is None:
            print(f"[recheck] case {args.case} could not be continued and its question is no longer open "
                  f"in the Strands session. Re-audit it with {_audit_again(args, case.source_path, args.case)}.",
                  file=sys.stderr)
            return EXIT_CANNOT_PROCEED
        print(question_text(store, args.case, _store_flag(args), exiting=True, store_note=_store_note(args)))
        # A rejected answer is a refusal (3). A question raised after an
        # accepted answer - a follow-up for a fact still missing - is the
        # case waiting on a reviewer, the documented 2; it exited 3, so a
        # wrapper recorded an open question as a failure.
        return EXIT_CANNOT_PROCEED if store.load(args.case).rejected_answer else EXIT_AWAITING_HUMAN
    return _finish(args, store, outcome)


def _recover_failed_resume(args, store: CaseStore, snapshot, source_path: str) -> str:
    """After a resume raised part-way, leave the case in a state a later command can use; say which.

    A run that raised (a case.json another program holds, a full disk,
    Ctrl+C) had consumed the question in the Strands session and left no node
    waiting on it, while case.json still said "awaiting_human": the reviewer
    could never answer again, and only --fresh - which discards everything -
    got the case moving. So the case and its session are put back as they
    were before this attempt.

    Except when the answer was already committed. A case that now loads as
    "ready" has its accepted answers on file and is finished by `resume`
    without an answer; one that loads as "complete" has a result the load
    re-derived. Rolling either back would throw away an accepted answer.
    """
    try:
        status = store.load(args.case).status
    except Exception:  # noqa: BLE001 - whatever is on disk now is not trusted
        status = None
    if status == "ready":
        return (f"The answer was accepted and is on file; finish the case with "
                f"`recheck{_store_flag(args)} resume --case {args.case}` (no --answer).")
    if status == "complete":
        return f"The case is complete; `recheck{_store_flag(args)} show --case {args.case}` prints it."
    try:
        store.restore(snapshot)
    except BaseException as restore_exc:  # noqa: BLE001 - report both failures
        return (f"The case could not be put back as it was ({type(restore_exc).__name__}: {restore_exc}); "
                f"re-audit it with {_audit_again(args, source_path, args.case)}.")
    return "The case and its question were put back as they were; answer again when the cause is fixed."


def _finish_ready(args, store: CaseStore, case) -> int:
    """Finish a case whose answers were accepted but whose arithmetic never ran.

    A resume killed after assess committed "ready" (Ctrl+C, a crash, a kill
    between the two writes) left the reviewer's answer on file and no figure,
    and the only recovery offered was --fresh, which threw the answer away.
    "ready" is written only by the assess node after the answers passed
    validation and nothing unknown could change the result, so the compute
    node - which checks the committed state for itself - finishes it from
    case.json. The Strands session is not needed: the question is closed.

    An --answer is refused rather than ignored: the answers on file are what
    the result will rest on, and a different one typed now must not look
    accepted.
    """
    from recheck.materiality import assess

    on_file = case.human_answers or "none"
    if args.answer is not None:
        print(f"[recheck] case {args.case} is not waiting on an answer: its answers are on file ({on_file}) "
              f"and it stopped before its arithmetic ran. Nothing was changed. To finish it from those "
              f"answers, run `recheck{_store_flag(args)} resume --case {args.case}` without --answer; to "
              f"audit it again, run {_audit_again(args, case.source_path, args.case)}.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    m = assess(case.load_decisions())
    if m.unknown and not m.settled:
        print(f"[recheck] case {args.case} is marked ready, but unknown facts on file could change its result. "
              f"Nothing was computed. Re-audit it with {_audit_again(args, case.source_path, args.case)}.",
              file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    print(f"[recheck] process {os.getpid()}: finishing case {args.case} from the facts and answers on file "
          f"({on_file}); running node 'compute'")
    # The same record the graph's node hook makes, so the report shows which
    # process ran the arithmetic.
    case.timeline.append({"node": "compute", "pid": os.getpid()})
    store.save(case)
    try:
        asyncio.run(ComputeNode(store, args.case).invoke_async("compute"))
    except Exception as exc:  # noqa: BLE001
        print(f"[recheck] resume failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    return _finish(args, store, outcome_from_case(store, args.case))


def cmd_sweep(args) -> int:
    store = CaseStore(args.store)
    directory = pathlib.Path(args.directory)
    if not directory.is_dir():
        print(f"[recheck] not a directory: {_shown_path(directory)}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    from recheck.sweep import documents_in

    documents = documents_in(directory)
    if not documents:
        print(f"[recheck] no .txt or .pdf documents in {_shown_path(directory)}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    factory, note, label = _resolve_factory(args)
    print(f"[recheck] sweeping {len(documents)} document(s) from {_shown_path(directory)}")
    print(f"[recheck] {note}")
    print()

    def run(store_, case_id, path, factory_):
        return _run_audit(store_, case_id, path, factory_, label)

    outcomes, code = sweep(store, directory, factory, run, fresh=args.fresh)
    print(render_triage(store, outcomes, store_flag=_store_flag(args), store_note=_store_note(args)))
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
    print(f"[recheck] audit and sweep call it only when run with --model {config.provider}.")
    return EXIT_OK


def cmd_show(args) -> int:
    """Print a stored case. Exit 0 only when what it shows can be acted on as shown."""
    store = CaseStore(args.store)
    try:
        case = store.load(args.case)
        args.case = case.case_id  # as recorded; see cmd_resume
        changed = bool(case.letter_changed())
        # A case "awaiting" a question its session does not hold printed the
        # question and a resume command that was always refused.
        lost = case.status == "awaiting_human" and not changed and not question_is_open(store, case)
        notice = None
        if changed:
            notice = (f"THE LETTER HAS CHANGED since this audit: {_shown_path(case.source_path)} no longer matches "
                      f"the document that was read. This report describes the earlier version.")
        elif lost:
            notice = ("THIS QUESTION CANNOT BE ANSWERED: the case's Strands session does not hold it, so any "
                      "answer would be refused.")
        print()
        print(render(case, show_trace=not args.brief, notice=notice, question_open=not lost))
        if changed:
            print(f"[recheck] the letter for case {args.case} has changed since it was audited. To audit it as it "
                  f"is now, run {_audit_again(args, case.source_path, args.case)}.", file=sys.stderr)
            return EXIT_CANNOT_PROCEED
        if lost:
            print(f"[recheck] case {args.case} says it is waiting on an answer, but its Strands session holds no "
                  f"open question. Re-audit it with {_audit_again(args, case.source_path, args.case)}.",
                  file=sys.stderr)
            return EXIT_CANNOT_PROCEED
        if case.status == "awaiting_human":
            print(question_text(store, args.case, _store_flag(args), exiting=False, store_note=_store_note(args)))
    except (FileNotFoundError, CaseCorrupt) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    return EXIT_OK


def _show(store: CaseStore, case_id: str, *, brief: bool = False) -> None:
    print()
    print(render(store.load(case_id), show_trace=not brief))


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 3, not argparse's 2.

    2 is the documented code for "waiting on a reviewer's answer". A missing
    --answer, or `--case -x`, exited 2, so an unattended wrapper that maps 2
    to "queued for a reviewer" recorded a malformed command as a persisted
    question. Subparsers inherit this class.
    """

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_CANNOT_PROCEED, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="recheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=DEFAULT_STORE, help="case store directory (default: runs)")
    parser.add_argument("--debug", action="store_true",
                        help="show framework logging (Strands node/graph internals), normally suppressed; "
                             "credential-bearing HTTP and signing logs stay suppressed")
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
    audit.add_argument("--brief", action="store_true", help="report without the decision trace")
    add_model_flags(audit)
    audit.set_defaults(func=cmd_audit)

    resume = sub.add_parser("resume", help="answer a case's question, in a new process")
    resume.add_argument("--case", required=True)
    resume.add_argument("--answer", default=None,
                        help='facts only, e.g. "2=left,3=right" or "4=upper-left". Omit it to finish a case '
                             'whose answers are already on file')
    resume.add_argument("--brief", action="store_true", help="report without the decision trace")
    resume.set_defaults(func=cmd_resume)

    sweep_p = sub.add_parser("sweep", help="audit a directory of letters unattended; report only what needs you")
    sweep_p.add_argument("directory")
    add_model_flags(sweep_p)
    sweep_p.set_defaults(func=cmd_sweep)

    show = sub.add_parser("show", help="print a stored case report")
    show.add_argument("--case", required=True)
    show.add_argument("--brief", action="store_true", help="report without the decision trace")
    show.set_defaults(func=cmd_show)

    pre = sub.add_parser("preflight", help="check live-provider configuration WITHOUT an inference call")
    pre.add_argument("--model", default=None, help="provider id; defaults to RECHECK_PROVIDER")
    pre.set_defaults(func=cmd_preflight)
    return parser


# Loggers whose DEBUG output carries request headers or signing material.
# With --debug, botocore.auth printed the canonical request including
# x-amz-security-token, and botocore.endpoint printed the Authorization header
# with the access key id - against the promise that Recheck never logs or
# prints a credential, which has no exception for --debug.
CREDENTIAL_LOGGERS = (
    "botocore.auth", "botocore.endpoint", "botocore.httpsession", "botocore.awsrequest",
    "urllib3", "httpcore", "httpx", "anthropic._base_client",
)


class RedactCredentials(logging.Filter):
    """Second line of defence: masks credential-shaped values in any record
    that still reaches the handler, from a logger not listed above."""

    PATTERNS = (
        (re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*b?['\"]?)[^'\"\r\n]+"), r"\1[REDACTED]"),
        (re.compile(r"(?i)(x-amz-security-token['\"]?\s*[:=]\s*b?['\"]?)[^'\"\s,}]+"), r"\1[REDACTED]"),
        (re.compile(r"(?i)(x-api-key['\"]?\s*[:=]\s*b?['\"]?)[^'\"\s,}]+"), r"\1[REDACTED]"),
        (re.compile(r"(?i)(aws_(?:secret_access_key|session_token|access_key_id)['\"]?\s*[:=]\s*['\"]?)"
                    r"[^'\"\s,}]+"), r"\1[REDACTED]"),
        (re.compile(r"Credential=[^/\s,'\"]+"), "Credential=[REDACTED]"),
        (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{12,}\b"), "[REDACTED]"),
        (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]+"), "[REDACTED]"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a malformed record is the logging module's problem
            return True
        redacted = message
        for pattern, replacement in self.PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def _configure_logging(debug: bool) -> None:
    """Keep framework logging out of product output.

    Strands logs node and graph failures at ERROR, so a deliberate refusal -
    an oversized document, a scanned PDF - printed a stack of "node failed /
    graph execution failed" lines underneath Recheck's own one-line
    explanation. The refusal is the product behaving correctly; the framework
    trace is noise that makes it look like a crash. --debug restores it, but
    never the HTTP and signing logs that carry credentials.
    """
    logging.basicConfig(level=logging.DEBUG if debug else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not debug:
        for name in ("strands", "botocore", "urllib3", "httpx", "anthropic"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
    else:
        for name in CREDENTIAL_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RedactCredentials) for f in handler.filters):
            handler.addFilter(RedactCredentials())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "debug", False))
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    except (OSError, CaseBusy, CaseCorrupt) as exc:
        # A store that cannot be written, a case another process holds, a
        # case file that cannot be trusted: no result, and no traceback.
        print(f"[recheck] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    except Exception as exc:  # provider misconfiguration, unreadable case, etc.
        from recheck.models.factory import ProviderError

        if isinstance(exc, ProviderError):
            print(f"[recheck] {exc}", file=sys.stderr)
            return EXIT_CANNOT_PROCEED
        raise


if __name__ == "__main__":
    raise SystemExit(main())
