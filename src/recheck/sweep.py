"""Caseload sweep: audit a stack of documents unattended, surface only decisions.

This is the product's actual shape, and the reason an orchestrated workflow
earns its place rather than a script.

The user is not a veteran holding one letter. It is a County Veterans Service
Officer or an accredited representative with a stack of them. A tool that
audits one document interactively demonstrates the machinery; a tool that
works through the stack unattended, completes everything it can resolve on
its own, and reports only the cases where a human must supply a fact the
document does not contain - that is the product.

Each document becomes its own case with its own durable state, so the ones
that need a decision can be answered later, in any order, by a different
process. That is why interrupt-and-resume is load-bearing here in a way it is
not for a single interactive run.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

from recheck.case import CaseStore
from recheck.provenance import Actor

EXIT_OK = 0
EXIT_AWAITING_HUMAN = 2
EXIT_CANNOT_PROCEED = 3

WIDTH = 78
BAR = "=" * WIDTH
THIN = "-" * WIDTH

DOCUMENT_SUFFIXES = (".txt", ".pdf")


@dataclass
class Outcome:
    """The result of auditing one document, in product terms rather than
    framework terms."""

    COMPLETE = "complete"
    AWAITING_HUMAN = "awaiting_human"
    FAILED = "failed"

    case_id: str
    state: str
    detail: str = ""
    interrupt: object | None = None


def case_id_for(path: pathlib.Path) -> str:
    """A filesystem-safe case id derived from the document's name.

    The store rejects anything that could escape its directory, so the id is
    sanitised here rather than relying on the document being well named.
    """
    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in path.stem)
    return stem[:64] or "case"


def documents_in(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    )


def sweep(
    store: CaseStore,
    directory: pathlib.Path,
    factory,
    run_audit,
    *,
    show_progress: bool = True,
) -> tuple[list[Outcome], int]:
    """Audit every document in `directory`. One bad document never stops the sweep.

    `run_audit` is injected rather than imported so the CLI keeps a single
    audit code path and this module stays testable in isolation.
    """
    outcomes: list[Outcome] = []
    for path in documents_in(directory):
        outcome = run_audit(store, case_id_for(path), path, factory)
        outcomes.append(outcome)
        if show_progress:
            mark = {
                Outcome.COMPLETE: ".",
                Outcome.AWAITING_HUMAN: "?",
                Outcome.FAILED: "!",
            }[outcome.state]
            sys.stdout.write(mark)
            sys.stdout.flush()
    if show_progress and outcomes:
        print()
        print()
    return outcomes, triage_exit_code(outcomes)


def triage_exit_code(outcomes: list[Outcome]) -> int:
    if any(o.state == Outcome.AWAITING_HUMAN for o in outcomes):
        return EXIT_AWAITING_HUMAN
    return EXIT_OK


def render_triage(store: CaseStore, outcomes: list[Outcome]) -> str:
    """The triage report. This is what the operator actually reads."""
    agree: list[tuple[Outcome, object]] = []
    discrepant: list[tuple[Outcome, object]] = []
    awaiting: list[tuple[Outcome, object]] = []
    failed: list[Outcome] = []
    totals = {Actor.DETERMINISTIC: 0, Actor.AI: 0, Actor.HUMAN: 0}

    for outcome in outcomes:
        if outcome.state == Outcome.FAILED:
            failed.append(outcome)
            continue
        try:
            case = store.load(outcome.case_id)
        except Exception as exc:  # noqa: BLE001
            failed.append(Outcome(outcome.case_id, Outcome.FAILED, f"case unreadable: {exc}"))
            continue
        trace = case.load_trace()
        for actor in totals:
            totals[actor] += len(trace.by_actor(actor))
        if outcome.state == Outcome.AWAITING_HUMAN:
            awaiting.append((outcome, case))
        elif case.recomputed_degree == case.stated_combined:
            agree.append((outcome, case))
        else:
            discrepant.append((outcome, case))

    out: list[str] = [BAR, f"CASELOAD TRIAGE  -  {len(outcomes)} document(s)", BAR]

    out.append("")
    out.append(_heading("NO ACTION NEEDED", len(agree)))
    out.append("     the recomputed evaluation agrees with the one stated")

    out.append("")
    out.append(_heading("POTENTIAL DISCREPANCY - review recommended", len(discrepant)))
    if discrepant:
        out.append("     questions for an accredited representative, not findings of error")
    for outcome, case in discrepant:
        delta = (case.recomputed_degree or 0) - (case.stated_combined or 0)
        factor = "4.26 applied" if case.bilateral_applied else "4.26 not applicable"
        out.append(
            f"     {outcome.case_id:<12} stated {case.stated_combined}%"
            f"   recomputed {case.recomputed_degree}%   ({delta:+d}, {factor})"
        )

    out.append("")
    out.append(_heading("AWAITING YOUR DECISION", len(awaiting)))
    for outcome, case in awaiting:
        unresolved = [d for d in case.load_decisions() if d.needs_human]
        out.append(f"     {outcome.case_id:<12} {len(unresolved)} condition(s) need a side")
        for decision in unresolved[:2]:
            out.append(f"                  - {decision.condition[:54]}")

    out.append("")
    out.append(_heading("COULD NOT PROCEED", len(failed)))
    for outcome in failed:
        out.append(f"     {outcome.case_id:<12} {outcome.detail[:56]}")

    out.append("")
    out.append(THIN)
    out.append(
        f"ownership across the sweep: {totals[Actor.DETERMINISTIC]} deterministic, "
        f"{totals[Actor.AI]} AI, {totals[Actor.HUMAN]} human decision(s)"
    )
    resolved = len(outcomes) - len(awaiting)
    out.append(
        f"{len(awaiting)} of {len(outcomes)} document(s) need a human. "
        f"The other {resolved} were resolved without one."
    )
    if awaiting:
        first = awaiting[0][0].case_id
        out.append("")
        out.append(f'answer one with:  recheck resume --case {first} --answer "<index>=left,..."')
    out.append(BAR)
    return "\n".join(out)


def _heading(label: str, count: int) -> str:
    pad = max(1, WIDTH - 4 - len(label) - len(str(count)) - 2)
    return f"  {label}{' ' * pad}{count}"
