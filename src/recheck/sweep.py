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

from recheck.case import RULE_426D_ACTION, Case, CaseBusy, CaseCorrupt, CaseStore, _WINDOWS_RESERVED
from recheck.provenance import Actor, wrap

EXIT_OK = 0
EXIT_AWAITING_HUMAN = 2
EXIT_CANNOT_PROCEED = 3

WIDTH = 78
BAR = "=" * WIDTH
THIN = "-" * WIDTH

DOCUMENT_SUFFIXES = (".txt", ".pdf")

#: trace action recorded when a reviewer was asked; counts questions asked, open or answered
QUESTION_ACTION = "Question for a reviewer"


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
    #: reviewer answers a --fresh re-audit threw away, as they were on file
    discarded_answers: dict | None = None


def case_id_for(path: pathlib.Path) -> str:
    """A filesystem-safe case id derived from the document's name.

    The store rejects anything that could escape its directory, so the id is
    sanitised here rather than relying on the document being well named. An
    id may not start with "-": a file named --help.txt became the printed
    command `resume --case --help`, which fails when pasted.
    """
    stem = "".join(c if (c.isascii() and (c.isalnum() or c in "-_")) else "_" for c in path.stem)
    stem = stem[:64] or "case"
    if stem.lower() in _WINDOWS_RESERVED or stem.startswith("-"):
        stem = f"case_{stem}"[:64]
    return stem


def documents_in(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    )


#: Short enough for a triage row.
QUESTION_LOST = "question not in its Strands session; re-audit with --fresh"
LETTER_CHANGED = "letter changed since it was audited; re-audit with --fresh"


def question_is_open(store: CaseStore, case: Case) -> bool:
    """Does the case's Strands session really hold its question?

    A run killed between writing case.json and writing the session left a
    case "awaiting_human" with no question in the session. The sweep's
    status view and `show` still printed NEEDS YOUR ANSWER and a resume
    command that was always refused. A missing session directory is checked
    first, because building a graph over it would create an empty session.

    An activated interrupt is not enough either: a run that raised part-way
    leaves one with no node waiting on it (the graph is not INTERRUPTED), and
    no answer can reach it. Such a stranded case was shown as NEEDS YOUR
    ANSWER with a resume command that could never work.
    """
    from recheck.graph import build_graph

    if not store.session_dir(case.case_id).is_dir():
        return False
    try:
        return open_question(build_graph(store, case.case_id, case.source_path, None), case.case_id) is not None
    except Exception:  # noqa: BLE001 - the persisted session is untrusted input
        return False


def open_question(graph, case_id: str):
    """The case's interrupt, if the restored graph is really waiting on it; else None."""
    from strands.multiagent.base import Status

    from recheck.graph import interrupt_id, outstanding_interrupt

    pending = outstanding_interrupt(graph)
    state = getattr(graph, "state", None)
    waiting = state is not None and state.status == Status.INTERRUPTED and bool(state.interrupted_nodes)
    return pending if waiting and pending is not None and pending.id == interrupt_id(case_id) else None


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
    if state == Outcome.AWAITING_HUMAN and reused and not question_is_open(store, case):
        state, detail = Outcome.FAILED, QUESTION_LOST
    elif state == Outcome.FAILED:
        if case.status == "unparsed":
            detail = case.extraction_failure() or "no assigned evaluations or no combined evaluation statement found"
        elif case.status == "ready":
            # A run killed after the answers were accepted but before compute.
            detail = "answers on file, arithmetic not run; run resume to finish"
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
    # Keyed ignoring letter case: on Windows and macOS "PAIR" and "pair" are
    # one case directory, and PAIR.txt and pair.pdf in one folder took turns
    # overwriting it - the second discarded the first's answered case.
    claimed: dict[str, tuple[str, pathlib.Path]] = {}
    for path in documents_in(directory):
        case_id = case_id_for(path)
        if case_id.casefold() in claimed:
            other_id, other = claimed[case_id.casefold()]
            same = "the same case id" if other_id == case_id else "the same case id, ignoring letter case"
            outcome = Outcome(case_id, Outcome.FAILED, f"{path.name} and {other.name} map to {same}; rename one")
        else:
            claimed[case_id.casefold()] = (case_id, path)
            try:
                # A resume or another sweep acting on this case at the same time
                # corrupted it; the sweep leaves a busy case alone.
                with store.lock(case_id):
                    outcome = _audit_or_reuse(store, case_id, path, factory, run_audit, fresh)
            except CaseBusy as exc:
                outcome = Outcome(case_id, Outcome.FAILED, str(exc))
            except OSError as exc:
                outcome = Outcome(case_id, Outcome.FAILED, f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - one bad case must not stop the sweep
                outcome = Outcome(case_id, Outcome.FAILED, f"{type(exc).__name__}: {exc}")
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
        except CaseCorrupt as exc:
            if not fresh:
                # Never discarded without --fresh. A case this build cannot
                # load may still hold a reviewer's answer - a stricter check,
                # or a name collision, made an answered case look corrupt, and
                # the sweep deleted it and asked the question again.
                return Outcome(case_id, Outcome.FAILED,
                               f"{exc}. Nothing was deleted; re-audit with --fresh", reused=True)
            existing = None
        same_document = existing is not None and _same_path(existing.source_path, path)
        if existing is not None and not same_document and not fresh:
            return Outcome(case_id, Outcome.FAILED,
                           f"case id already used by {pathlib.Path(existing.source_path).name}; rename or use --fresh")
        if fresh or existing is None:
            try:
                store.discard(case_id)
            except OSError as exc:
                return Outcome(case_id, Outcome.FAILED, f"could not discard the case on file: {exc}")
            if existing is not None and existing.human_answers:
                # Said, not silent: `sweep --fresh` re-audits every case, and
                # an answered case came back as an open question with nothing
                # to show the reviewer's answer had ever been given.
                outcome = run_audit(store, case_id, path, factory)
                outcome.discarded_answers = dict(existing.human_answers)
                return outcome
        else:
            # The case on file describes the letter as it was audited. If the
            # letter has changed since, its figures describe a document that
            # no longer exists; report that rather than the stale result.
            if existing.letter_changed():
                return Outcome(case_id, Outcome.FAILED, LETTER_CHANGED, reused=True)
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


def _discrepancy_note(case: Case) -> str:
    """The note on a triage row: the difference, what 4.26 did, and what it rests on."""
    delta = (case.recomputed_degree or 0) - (case.stated_combined or 0)
    if case.has_trace_action(RULE_426D_ACTION):
        # "4.26 not applied" read as VA having skipped the factor, when the
        # letter's figure is what the prior rule gives before April 16, 2023.
        factor = "4.26(d) exception - check decision period"
    else:
        factor = "4.26 applied" if case.bilateral_applied else "4.26 not applied"
    ai = [i for i, d in enumerate(case.load_decisions()) if d.group_by is Actor.AI]
    uses_ai = "; AI groups " + " ".join(f"[{i}]" for i in ai) if ai else ""
    return f"({delta:+d}, {factor}{uses_ai})"


def render_triage(store: CaseStore, outcomes: list[Outcome], *, store_flag: str = "", store_note: str = "") -> str:
    """The triage report. This is what the operator actually reads.

    Each case is rendered on its own: a case that cannot be read or rendered
    is listed under COULD NOT PROCEED. A type-confused field in one case.json
    once raised out of here and the sweep printed no triage at all, for any
    document.
    """
    from recheck.graph import accepted_answers, asked

    agree, discrepant, awaiting, undetermined, failed = [], [], [], [], []
    groups = {"DETERMINISTIC": 0, "AI": 0, "HUMAN": 0, "unknown": 0}
    sides_from_letter = 0
    immaterial_cases = 0
    questions_asked = 0
    reused = sum(1 for o in outcomes if o.reused)
    answered_on_file = 0

    for outcome in outcomes:
        if outcome.state == Outcome.FAILED:
            failed.append((outcome, _detail_rows(outcome.case_id, outcome.detail)))
            if outcome.reused:
                # A row that says "re-audit with --fresh" (letter changed,
                # question lost) can hold a reviewer's answers too.
                try:
                    answered_on_file += bool(store.load(outcome.case_id).human_answers)
                except Exception:  # noqa: BLE001 - a case that cannot be loaded is already listed
                    pass
            continue
        try:
            case = store.load(outcome.case_id)
            decisions = case.load_decisions()
            counts = [d.group_by.value if d.group_by else "unknown" for d in decisions]
            sides = sum(1 for d in decisions if d.side_by is Actor.DETERMINISTIC)
            if outcome.state == Outcome.AWAITING_HUMAN:
                bucket = awaiting
                unknown = [(i, decisions[i]) for i in asked(decisions)]
                placeholder = ",".join(f"{i}=<{'|'.join(accepted_answers(d))}>" for i, d in unknown)
                lines = [f"     {outcome.case_id:<12} could be {_or(case.possible_degrees)}; "
                         f"letter states {case.stated_combined}%"]
                lines += [f"       [{i}] {d.condition[:44]:<44} {' + '.join(d.missing)}?" for i, d in unknown]
                lines.append(f"       recheck{store_flag} resume --case {outcome.case_id} --answer \"{placeholder}\"")
            elif outcome.state == Outcome.UNDETERMINED:
                bucket = undetermined
                # The reason is wrapped, not cut: it used to stop mid-word at 44 characters.
                head = (f"could be {_or(case.possible_degrees)}:" if case.possible_degrees else "not computed:")
                lines = [f"     {outcome.case_id:<12} {head}"]
                lines += [f"     {'':<12} {part}" for part in wrap(case.undetermined_reason or "", 60)]
            elif case.recomputed_degree == case.stated_combined:
                bucket, lines = agree, []
            else:
                bucket = discrepant
                lines = [f"       {outcome.case_id:<12} stated {case.stated_combined}%   "
                         f"recomputed {case.recomputed_degree}%   {_discrepancy_note(case)}"]
        except Exception as exc:  # noqa: BLE001 - one bad case must not cost the operator the whole triage
            failed.append((outcome, _detail_rows(outcome.case_id, f"case unreadable: {exc}")))
            continue
        for key in counts:
            groups[key] += 1
        sides_from_letter += sides
        immaterial_cases += bool(case.immaterial_unknowns)
        questions_asked += case.has_trace_action(QUESTION_ACTION)
        answered_on_file += bool(outcome.reused and case.human_answers)
        bucket.append((case, lines))

    out: list[str] = [BAR, f"CASELOAD TRIAGE  -  {len(outcomes)} document(s)", BAR, ""]

    out.append(_heading("NO DISCREPANCY FOUND", len(agree)))
    out.append("")
    out.append(_heading("POTENTIAL DISCREPANCY - review recommended", len(discrepant)))
    if discrepant:
        out.append("     questions to raise in review, not findings of error")
    for higher in (True, False):
        rows = [(c, lines) for c, lines in discrepant
                if ((c.recomputed_degree or 0) > (c.stated_combined or 0)) == higher]
        if not rows:
            continue
        out.append("     recomputed HIGHER than the letter" if higher else
                   "     recomputed LOWER than the letter - raising it could prompt a downward review")
        for _, lines in rows:
            out += lines

    out.append("")
    out.append(_heading("NEEDS YOUR ANSWER", len(awaiting)))
    for _, lines in awaiting:
        out += lines
    if awaiting and store_note:
        out.append(f"     {store_note}")

    out.append("")
    out.append(_heading("UNDETERMINED - not computed", len(undetermined)))
    for _, lines in undetermined:
        out += lines

    out.append("")
    out.append(_heading("COULD NOT PROCEED", len(failed)))
    for _, lines in failed:
        out += lines

    out += ["", THIN]
    out.append(f"{len(outcomes)} documents: {len(agree)} no discrepancy, {len(discrepant)} to review, "
               f"{len(awaiting)} waiting on an answer, {len(undetermined)} undetermined, "
               f"{len(failed)} could not proceed")
    # Asked, not open: a question answered since still counts. This line
    # used to print the number still waiting under the label "asked".
    out.append(f"questions asked: {questions_asked}.  letters with unknown facts that could not change "
               f"the rating, so nobody was asked: {immaterial_cases}")
    out.append(f"extremity groups: {groups['DETERMINISTIC']} lexicon, {groups['AI']} AI, "
               f"{groups['HUMAN']} reviewer, {groups['unknown']} unknown.  sides read from the letters: "
               f"{sides_from_letter}.  arithmetic: all deterministic")
    if reused:
        out.append(f"{reused} case(s) were already on file and are shown as they stand; "
                   f"add --fresh to re-audit them"
                   + (f" (this discards the reviewer answers on file for {answered_on_file} case(s))"
                      if answered_on_file else ""))
    discarded = [o for o in outcomes if o.discarded_answers]
    if discarded:
        out.append(f"--fresh discarded the reviewer answers on file for {len(discarded)} case(s): "
                   + "; ".join(f"{o.case_id} ({','.join(f'{k}={v}' for k, v in o.discarded_answers.items())})"
                               for o in discarded))
    out.append(BAR)
    return "\n".join(out)


def _detail_rows(case_id: str, detail: str) -> list[str]:
    """A COULD NOT PROCEED row, wrapped rather than cut.

    Cut at 60 characters, the reason a case could not be used lost its end -
    and the end is where the way forward ("re-audit with --fresh") is.
    """
    first, *rest = wrap(detail, 60)
    return [f"     {case_id:<12} {first}"] + [f"     {'':<12} {line}" for line in rest]


def _or(values) -> str:
    items = [f"{v}%" for v in values]
    if not items:
        return "?"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _heading(label: str, count: int) -> str:
    pad = max(1, WIDTH - 4 - len(label) - len(str(count)) - 2)
    return f"  {label}{' ' * pad}{count}"
