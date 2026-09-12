"""Recheck command line.

    recheck audit     <letter> --case ID [--scripted | --model PROVIDER]
    recheck resume    --case ID --answer "0=left,1=right"
    recheck show      --case ID
    recheck preflight [--model PROVIDER]

The two-command shape is not a convenience. `audit` may stop at an
interrupt and exit; `resume` is a DIFFERENT PROCESS that restores state from
disk. That is the whole point, so it is the default way to drive the tool
rather than a special mode.

Exit codes are meaningful, because the demo depends on them:
    0  completed
    2  awaiting human input (an interrupt was raised and persisted)
    3  could not proceed (unparsable document, invalid answer, bad case)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys

from strands.multiagent.base import Status

from recheck.case import CaseCorrupt, CaseStore
from recheck.graph import (
    INTERRUPT_ID,
    DocumentTooLarge,
    ScannedDocument,
    build_graph,
)
from recheck.report import render

EXIT_OK = 0
EXIT_AWAITING_HUMAN = 2
EXIT_CANNOT_PROCEED = 3

SCRIPTED_NOTE = (
    "zero-model path: classifications are supplied by ScriptedModel, which drives the real "
    "Strands structured-output code path with no network and no inference cost"
)


def _scripted_factory(fixture: pathlib.Path | None):
    """Build an agent whose model replays a committed classification fixture."""
    from strands import Agent

    from recheck.classify import SYSTEM_PROMPT
    from recheck.models.scripted import ScriptedModel

    if fixture is None or not fixture.exists():
        payload = {"classifications": []}
    else:
        payload = json.loads(fixture.read_text(encoding="utf-8"))

    def make():
        # callback_handler=None suppresses Strands' default stdout handler,
        # which otherwise prints "Tool #1: ClassificationBatch" into the
        # middle of the report. Framework chatter is not product output.
        return Agent(
            model=ScriptedModel(payload=payload),
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,
        )

    return make


def _resolve_factory(args) -> tuple[object | None, str]:
    if args.scripted:
        fixture = pathlib.Path(args.classifications) if args.classifications else None
        return _scripted_factory(fixture), SCRIPTED_NOTE
    if getattr(args, "model", None):
        from recheck.models.factory import build_agent_factory

        factory = build_agent_factory(args.model)
        config = getattr(factory, "config", None)
        detail = config.describe() if config is not None else args.model
        return factory, f"LIVE model provider: {detail}"
    return None, (
        "no classifier configured: conditions outside the deterministic lexicon will be "
        "routed to human review rather than guessed"
    )


def _print_interrupt(result) -> None:
    for interrupt in result.interrupts:
        reason = interrupt.reason or {}
        print()
        print("=" * 74)
        print("HUMAN INPUT REQUIRED")
        print("=" * 74)
        print(reason.get("question", "input required"))
        print()
        for item in reason.get("conditions", []):
            print(f"  [{item['index']}] {item['condition']}  ({item['percent']}%)")
            print(f"        extremity group: {item['extremity_group']}")
            print(f"        why asking: {item['why']}")
        print()
        print(f"  accepted values: {', '.join(reason.get('accepted_values', []))}")
        print(f"  note: {reason.get('note', '')}")
        print()
        print("This process is now exiting. State has been persisted to disk.")
        print("Resume in a NEW process with:")
        indices = [str(i["index"]) for i in reason.get("conditions", [])]
        example = ",".join(f"{i}=left" for i in indices) or "0=left"
        print(f"    python -m recheck.cli resume --case <ID> --answer \"{example}\"")
        print("=" * 74)


def cmd_audit(args) -> int:
    store = CaseStore(args.store)
    source = pathlib.Path(args.letter)
    if not source.exists():
        print(f"no such document: {source}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    factory, note = _resolve_factory(args)
    print(f"[recheck] {note}")

    graph = build_graph(store, args.case, str(source), factory)
    try:
        result = asyncio.run(graph.invoke_async(f"audit {source.name}"))
    except (ScannedDocument, DocumentTooLarge) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    if result.status == Status.INTERRUPTED:
        state_path = store.dir_for(args.case) / "graph_state.json"
        state_path.write_text(json.dumps(graph.serialize_state()), encoding="utf-8")
        _print_interrupt(result)
        return EXIT_AWAITING_HUMAN

    if result.status == Status.FAILED:
        print("[recheck] could not proceed; see the trace below", file=sys.stderr)
        _show(store, args.case)
        return EXIT_CANNOT_PROCEED

    problem = _completed_cleanly(store, args.case)
    if problem:
        print(f"[recheck] {problem}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    _show(store, args.case)
    return EXIT_OK


def cmd_resume(args) -> int:
    store = CaseStore(args.store)
    try:
        case = store.load(args.case)
    except (FileNotFoundError, CaseCorrupt) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    state_path = store.dir_for(args.case) / "graph_state.json"
    if not state_path.exists():
        print(
            f"[recheck] case {args.case} has no persisted graph state; nothing to resume",
            file=sys.stderr,
        )
        return EXIT_CANNOT_PROCEED

    factory, note = _resolve_factory(args)
    print(f"[recheck] resuming case {args.case} in a fresh process (pid {__import__('os').getpid()})")
    print(f"[recheck] {note}")

    graph = build_graph(store, args.case, case.source_path, factory)

    # The persisted graph state is untrusted input like anything else on disk.
    # Handing it straight to the framework turned a JSON array into an
    # unhandled AttributeError from inside deserialize_state - a stack trace
    # instead of an explanation. Validate the shape, then treat any
    # deserialisation failure as a corrupt case rather than a crash.
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[recheck] persisted graph state for {args.case} is not valid JSON: {exc}",
              file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    if not isinstance(payload, dict):
        print(f"[recheck] persisted graph state for {args.case} is a "
              f"{type(payload).__name__}, expected an object. The case is corrupt; "
              f"delete it and re-run the audit.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    try:
        graph.deserialize_state(payload)
    except Exception as exc:  # noqa: BLE001 - any failure here means a corrupt case
        print(f"[recheck] persisted graph state for {args.case} could not be restored "
              f"({type(exc).__name__}). The case is corrupt; delete it and re-run the "
              f"audit.", file=sys.stderr)
        return EXIT_CANNOT_PROCEED

    result = asyncio.run(
        graph.invoke_async(
            [{"interruptResponse": {"interruptId": INTERRUPT_ID, "response": args.answer}}]
        )
    )
    if result.status == Status.FAILED:
        _show(store, args.case)
        return EXIT_CANNOT_PROCEED
    if result.status == Status.INTERRUPTED:
        state_path.write_text(json.dumps(graph.serialize_state()), encoding="utf-8")
        _print_interrupt(result)
        return EXIT_AWAITING_HUMAN

    print(f"[recheck] execution order: {[n.node_id for n in result.execution_order]}")
    problem = _completed_cleanly(store, args.case)
    if problem:
        print(f"[recheck] {problem}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    _show(store, args.case)
    return EXIT_OK


def cmd_preflight(args) -> int:
    """Validate provider configuration without making an inference call."""
    from recheck.models.factory import (
        ProviderNotConfigured,
        load_config,
        preflight,
    )

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
        if not check.ok:
            failed += 1
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
        _show(store, args.case)
    except (FileNotFoundError, CaseCorrupt) as exc:
        print(f"[recheck] {exc}", file=sys.stderr)
        return EXIT_CANNOT_PROCEED
    return EXIT_OK


def _completed_cleanly(store: CaseStore, case_id: str) -> str | None:
    """Post-condition for a run that claims success. Returns a reason on failure.

    A graph can report COMPLETED while having executed nothing - a corrupted
    graph_state.json deserialises into a state with no pending work, the run
    is a silent no-op, and the exit code says success. That is worse than a
    crash, because the operator believes the audit ran.

    So success is defined by the PRODUCT outcome, not the framework's status:
    the case must be complete and must carry a recomputed degree.
    """
    try:
        case = store.load(case_id)
    except Exception as exc:  # noqa: BLE001 - any unreadable case is a failure here
        return f"case could not be re-read after the run: {exc}"
    if case.status != "complete":
        return (
            f"the run reported success but the case status is {case.status!r}. "
            f"Nothing was computed. If this case was resumed, its persisted state "
            f"may be corrupt - delete it and re-run the audit."
        )
    if case.recomputed_degree is None:
        return "the run reported success but produced no recomputed evaluation"
    return None


def _show(store: CaseStore, case_id: str) -> None:
    print()
    print(render(store.load(case_id)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recheck", description=__doc__)
    parser.add_argument("--store", default="runs", help="case store directory")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show framework logging (Strands node/graph internals), normally suppressed",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_flags(p):
        p.add_argument("--scripted", action="store_true",
                       help="use ScriptedModel: no network, no cost, real Strands code path")
        p.add_argument("--classifications", default=None,
                       help="JSON fixture of classifications for --scripted")
        p.add_argument("--model", default=None,
                       help="live provider id (e.g. bedrock, anthropic, ollama)")

    audit = sub.add_parser("audit", help="audit a decision letter")
    audit.add_argument("letter")
    audit.add_argument("--case", required=True)
    add_model_flags(audit)
    audit.set_defaults(func=cmd_audit)

    resume = sub.add_parser("resume", help="resume an interrupted case in a fresh process")
    resume.add_argument("--case", required=True)
    resume.add_argument("--answer", required=True,
                        help='laterality only, e.g. "0=left,1=right"')
    add_model_flags(resume)
    resume.set_defaults(func=cmd_resume)

    show = sub.add_parser("show", help="print a stored case report")
    show.add_argument("--case", required=True)
    show.set_defaults(func=cmd_show)

    pre = sub.add_parser(
        "preflight",
        help="validate live-provider configuration WITHOUT making an inference call",
    )
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
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not debug:
        for name in ("strands", "strands.multiagent", "strands.event_loop", "botocore", "urllib3"):
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
