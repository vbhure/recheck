"""Caseload sweep: audit a stack of documents unattended, surface only decisions.

The user is not a veteran holding one letter. It is a County Veterans Service
Officer or an accredited representative with a stack of them. A tool that
audits one document interactively demonstrates the machinery; a tool that
works through the stack unattended, finishes everything it can establish on
its own, and brings back only the questions whose answers change a rating -
that is the product.

Each document becomes its own case with its own durable state, so a question
can be answered later, in any order, by a different process. Running the
sweep again does not redo finished work: a case already on file is reported
as it stands, which makes the sweep the caseload's status view as well.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

from recheck.case import CaseCorrupt, CaseStore, _WINDOWS_RESERVED
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
    UNDETERMINED = "undetermined"
    FAILED = "failed"

    case_id: str
    state: str
    detail: str = ""
    interrupt: object | None = None
    reused: bool = False


def case_id_for(path: pathlib.Path) -> str:
    """A filesystem-safe case id derived from the document's name.

    The store rejects anything that could escape its directory, so the id is
    sanitised here rather than relying on the document being well named.
    """
    stem = "".join(c if (c.isascii() and (c.isalnum() or c in "-_")) else "_" for c in path.stem)
    stem = stem[:64] or "case"
    if stem.lower() in _WINDOWS_RESERVED:
        stem = f"case_{stem}"
    return stem


def documents_in(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    )


def outcome_from_case(store: CaseStore, case_id: str, *, reused: bool = False) -> Outcome:
    """Classify a stored case into a triage outcome."""
    try:
        case = store.load(case_id)
    except (FileNotFoundError, CaseCorrupt) as exc:
        return Outcome(case_id, Outcome.FAILED, str(exc), reused=reused)
    state = {
        "complete": Outcome.COMPLETE,
        "awaiting_human": Outcome.AWAITING_HUMAN,
        "undetermined": Outcome.UNDETERMINED,
    }.get(case.status, Outcome.FAILED)
    detail = ""
    if state == Outcome.FAILED:
        if case.status == "unparsed":
            detail = case.extraction_failure() or "no assigned evaluations or no combined evaluation statement found"
        else:
            detail = f"the run stopped with the case in state {case.status!r}"
    return Outcome(case_id, state, detail, reused=reused)


def sweep(
    store: CaseStore,
    directory: pathlib.Path,
    factory,
    run_audit,
    *,
    fresh: bool = False,
    show_progress: bool = True,
) -> tuple[list[Outcome], int]:
    """Audit every document in `directory`. One bad document never stops the sweep.

    `run_audit(store, case_id, path, factory)` is injected so the CLI keeps a
    single audit code path and this module stays testable in isolation.
    """
    outcomes: list[Outcome] = []
    claimed: dict[str, pathlib.Path] = {}
    for path in documents_in(directory):
        case_id = case_id_for(path)
        if case_id in claimed:
            outcome = Outcome(case_id, Outcome.FAILED,
                              f"{path.name} and {claimed[case_id].name} map to the same case id; rename one")
        else:
            claimed[case_id] = path
            outcome = _audit_or_reuse(store, case_id, path, factory, run_audit, fresh)
        outcomes.append(outcome)
        if show_progress:
            mark = {Outcome.COMPLETE: ".", Outcome.AWAITING_HUMAN: "?",
                    Outcome.UNDETERMINED: "~", Outcome.FAILED: "x"}[outcome.state]
            sys.stdout.write(mark)
            sys.stdout.flush()
    if show_progress and outcomes:
        print("   (. finished   ? needs an answer   ~ undetermined   x could not proceed)")
        print()
    return outcomes, triage_exit_code(outcomes)


def _audit_or_reuse(store, case_id, path, factory, run_audit, fresh) -> Outcome:
    if store.exists(case_id):
        try:
            existing = store.load(case_id)
        except CaseCorrupt:
            existing = None
        same_document = existing is not None and _same_path(existing.source_path, path)
        if existing is not None and not same_document and not fresh:
            return Outcome(case_id, Outcome.FAILED,
                           f"case id already used by {pathlib.Path(existing.source_path).name}; rename or use --fresh")
        if fresh or existing is None:
            store.discard(case_id)
        else:
            return outcome_from_case(store, case_id, reused=True)
    return run_audit(store, case_id, path, factory)


def _same_path(recorded: str, path: pathlib.Path) -> bool:
    try:
        return pathlib.Path(recorded).resolve() == path.resolve()
    except OSError:
        return False


def triage_exit_code(outcomes: list[Outcome]) -> int:
    """2 if any case waits on an answer; else 3 if any produced no result; else 0."""
    if any(o.state == Outcome.AWAITING_HUMAN for o in outcomes):
        return EXIT_AWAITING_HUMAN
    if any(o.state in (Outcome.FAILED, Outcome.UNDETERMINED) for o in outcomes):
        return EXIT_CANNOT_PROCEED
    return EXIT_OK


def render_triage(store: CaseStore, outcomes: list[Outcome], *, store_flag: str = "") -> str:
    """The triage report. This is what the operator actually reads."""
    from recheck.graph import accepted_answers

    agree, discrepant, awaiting, undetermined, failed = [], [], [], [], []
    groups = {"DETERMINISTIC": 0, "AI": 0, "HUMAN": 0, "unknown": 0}
    sides_from_letter = 0
    immaterial_cases = 0
    reused = sum(1 for o in outcomes if o.reused)

    for outcome in outcomes:
        if outcome.state == Outcome.FAILED:
            failed.append((outcome, None))
            continue
        try:
            case = store.load(outcome.case_id)
        except (FileNotFoundError, CaseCorrupt) as exc:
            failed.append((Outcome(outcome.case_id, Outcome.FAILED, f"case unreadable: {exc}"), None))
            continue
        for d in case.load_decisions():
            groups[d.group_by.value if d.group_by else "unknown"] += 1
            if d.side_by is Actor.DETERMINISTIC:
                sides_from_letter += 1
        if case.immaterial_unknowns:
            immaterial_cases += 1
        bucket = {Outcome.AWAITING_HUMAN: awaiting, Outcome.UNDETERMINED: undetermined}.get(outcome.state)
        if bucket is None:
            bucket = agree if case.recomputed_degree == case.stated_combined else discrepant
        bucket.append((outcome, case))

    out: list[str] = [BAR, f"CASELOAD TRIAGE  -  {len(outcomes)} document(s)", BAR, ""]

    out.append(_heading("NO DISCREPANCY FOUND", len(agree)))
    out.append("")
    out.append(_heading("POTENTIAL DISCREPANCY - review recommended", len(discrepant)))
    if discrepant:
        out.append("     questions to raise in review, not findings of error")
    for higher in (True, False):
        rows = [(o, c) for o, c in discrepant if ((c.recomputed_degree or 0) > (c.stated_combined or 0)) == higher]
        if not rows:
            continue
        out.append("     recomputed HIGHER than the letter" if higher else
                   "     recomputed LOWER than the letter - raising it could prompt a downward review")
        for outcome, case in rows:
            delta = (case.recomputed_degree or 0) - (case.stated_combined or 0)
            factor = "4.26 applied" if case.bilateral_applied else "4.26 not applied"
            out.append(f"       {outcome.case_id:<12} stated {case.stated_combined}%   "
                       f"recomputed {case.recomputed_degree}%   ({delta:+d}, {factor})")

    out.append("")
    out.append(_heading("NEEDS YOUR ANSWER", len(awaiting)))
    for outcome, case in awaiting:
        decisions = case.load_decisions()
        unknown = [(i, d) for i, d in enumerate(decisions) if d.missing]
        out.append(f"     {outcome.case_id:<12} could be {_or(case.possible_degrees)}; "
                   f"letter states {case.stated_combined}%")
        for i, d in unknown:
            out.append(f"       [{i}] {d.condition[:44]:<44} {' + '.join(d.missing)}?")
        placeholder = ",".join(f"{i}=<{'|'.join(accepted_answers(d))}>" for i, d in unknown)
        out.append(f"       recheck{store_flag} resume --case {outcome.case_id} --answer \"{placeholder}\"")

    out.append("")
    out.append(_heading("UNDETERMINED - not computed", len(undetermined)))
    for outcome, case in undetermined:
        out.append(f"     {outcome.case_id:<12} could be {_or(case.possible_degrees)}: "
                   f"{(case.undetermined_reason or '')[:44]}")

    out.append("")
    out.append(_heading("COULD NOT PROCEED", len(failed)))
    for outcome, _ in failed:
        out.append(f"     {outcome.case_id:<12} {outcome.detail[:60]}")

    out += ["", THIN]
    out.append(f"{len(outcomes)} documents: {len(agree)} no discrepancy, {len(discrepant)} to review, "
               f"{len(awaiting)} waiting on an answer, {len(undetermined)} undetermined, "
               f"{len(failed)} could not proceed")
    out.append(f"questions asked: {len(awaiting)}.  letters with unknown facts that could not change "
               f"the rating, so nobody was asked: {immaterial_cases}")
    out.append(f"extremity groups: {groups['DETERMINISTIC']} lexicon, {groups['AI']} AI, "
               f"{groups['HUMAN']} reviewer, {groups['unknown']} unknown.  sides read from the letters: "
               f"{sides_from_letter}.  arithmetic: all deterministic")
    if reused:
        out.append(f"{reused} case(s) were already on file and are shown as they stand; "
                   f"add --fresh to re-audit them")
    out.append(BAR)
    return "\n".join(out)


def _or(values) -> str:
    items = [f"{v}%" for v in values]
    if not items:
        return "?"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _heading(label: str, count: int) -> str:
    pad = max(1, WIDTH - 4 - len(label) - len(str(count)) - 2)
    return f"  {label}{' ' * pad}{count}"
