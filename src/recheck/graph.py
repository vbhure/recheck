"""The Strands Graph: one per document.

    extract --(gate)--> classify --> assess --(gate)--> compute
                                       |
                                       v
                 INTERRUPT - only when an answer would change the rating

Why a Graph rather than a function call chain. The run must be able to STOP
at `assess`, persist, and be continued by a different process later - hours
later, by a different person - and every step must be observable. Strands
provides the pieces: an Interrupt raised from a node, a FileSessionManager
that persists and restores the graph (including the outstanding interrupt),
conditional edges, and node hooks.

Node ownership, enforced by construction:
  extract   DETERMINISTIC   file to ratings + stated combined value
  classify  AI (for the gap) extremity group only, for terms the lexicon
                             abstains on; sides are read by deterministic code
  assess    DETERMINISTIC   which unknown facts could change the rating, and
                             validation of any human answer
  compute   DETERMINISTIC   38 CFR 4.25 / 4.26 and the comparison

Two edges are gates. Returning Status.FAILED from a node does not stop
downstream nodes in this SDK version, so the topology - not convention - is
what stops a failed extraction reaching classification, or an open question
or undetermined case reaching the arithmetic.

Gate conditions must be STABLE. Strands re-evaluates them when it persists
the session, to work out where a resumed run continues. A condition that
flips once its target has run (for example "status is classified", which
stops being true when assess asks a question) empties the resume frontier,
and the resumed run silently does nothing. So the extraction gate reads the
extract node's own result from the Strands GraphState, and the compute gate
accepts only states from which computing is correct and which never revert:
"ready" and "complete".

Persistence has one owner. FileSessionManager restores graph state,
including an outstanding interrupt, when the graph is built; the case file
holds the domain facts. There is no second hand-written copy of graph state.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Sequence

from strands.hooks import BeforeNodeCallEvent, HookProvider, HookRegistry
from strands.interrupt import Interrupt
from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, Status
from strands.session.file_session_manager import FileSessionManager

from recheck.case import Case, CaseStore
from recheck.classify import AgentFactory, Decision, classify_async
from recheck.extract.deterministic import ExtractedRating, parse
from recheck.materiality import Materiality, assess as assess_materiality, evaluate_established
from recheck.provenance import Actor, Trace

NODE_ORDER = ("extract", "classify", "assess", "compute")

# A rating decision is a handful of pages. These caps are generous by an order
# of magnitude and exist so that a hostile or accidental input cannot exhaust
# memory: an unbounded read of a multi-gigabyte file, or a PDF decompression
# bomb whose page count explodes on parse.
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 100


def interrupt_id(case_id: str) -> str:
    return f"recheck:{case_id}:assess"


class ScannedDocument(Exception):
    """The document is an image. Recheck refuses rather than guessing."""


class DocumentTooLarge(Exception):
    """The document exceeds the bounds of anything plausibly a decision letter."""


def read_document(path: str | pathlib.Path) -> str:
    """Load letter text. PDFs must carry a text layer; OCR is out of scope.

    Input is bounded. Documents are untrusted, and refusing an implausible
    one by name is better than dying of memory exhaustion halfway through.
    """
    p = pathlib.Path(path)
    size = p.stat().st_size  # raises FileNotFoundError for a missing path
    if size > MAX_DOCUMENT_BYTES:
        raise DocumentTooLarge(
            f"{p.name} is {size / 1024 / 1024:.1f} MB, over the "
            f"{MAX_DOCUMENT_BYTES / 1024 / 1024:.0f} MB limit. A rating decision is a "
            f"few pages; refusing rather than loading it."
        )

    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise DocumentTooLarge(
                f"{p.name} has {page_count} pages, over the {MAX_PDF_PAGES}-page limit."
            )
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ScannedDocument(
                f"{p.name} has no extractable text layer. This document requires OCR, "
                f"which Recheck deliberately does not perform. Supply a text-layer PDF "
                f"or a .txt transcript."
            )
        return text
    return p.read_text(encoding="utf-8")


def _done(status: Status = Status.COMPLETED) -> MultiAgentResult:
    """A node result. Domain output lives in the case file, not in node text."""
    return MultiAgentResult(status=status)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class ExtractNode(MultiAgentBase):
    """DETERMINISTIC. Document in, ratings out. No model, no network."""

    def __init__(self, store: CaseStore, case_id: str, source: str) -> None:
        super().__init__()
        self.id = "extract"
        self.store, self.case_id, self.source = store, case_id, source

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        text = read_document(self.source)
        extraction = parse(text)
        # The caller opens the case (recording the classifier); extraction
        # starts its facts and trace afresh.
        case = self.store.load(self.case_id)
        trace = Trace()
        if not extraction.ok:
            trace.add(Actor.DETERMINISTIC, "Extraction FAILED",
                      extraction.unparsed_reason or "no ratings found", value="cannot proceed")
            case.status = "unparsed"
            case.store_trace(trace)
            self.store.save(case)
            return _done(Status.FAILED)

        case.stated_combined = extraction.stated_combined
        case.ratings = [
            {
                "condition": r.condition,
                "percent": r.percent,
                "extremity_group": r.extremity_group,
                "source_line": r.source_line,
                "source_line_number": r.source_line_number,
            }
            for r in extraction.ratings
        ]
        trace.add(
            Actor.DETERMINISTIC,
            "Stated combined evaluation",
            f"as printed in {pathlib.Path(self.source).name}",
            value=f"{extraction.stated_combined}%",
            evidence="combined evaluation statement",
            rule="38 CFR 4.25 (value under review)",
        )
        case.status = "extracted"
        case.store_trace(trace)
        self.store.save(case)
        return _done()


class ClassifyNode(MultiAgentBase):
    """AI, but only for what the lexicon cannot reach. See recheck.classify."""

    def __init__(self, store: CaseStore, case_id: str, agent_factory: AgentFactory | None) -> None:
        super().__init__()
        self.id = "classify"
        self.store, self.case_id, self.agent_factory = store, case_id, agent_factory

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        case = self.store.load(self.case_id)
        ratings = [ExtractedRating(**r) for r in case.ratings]
        trace = case.load_trace()
        decisions = await classify_async(ratings, trace, self.agent_factory)
        case.store_decisions(decisions)
        case.store_trace(trace)
        case.status = "classified"
        self.store.save(case)
        return _done()


class AssessNode(MultiAgentBase):
    """DETERMINISTIC gatekeeper: decides whether a human is needed, validates answers.

    A human is interrupted only when enumerating every possible answer to the
    unknown facts yields more than one final degree. A human is asked only
    for the facts that are unknown - never for a fact the letter states, and
    never for a number.
    """

    def __init__(self, store: CaseStore, case_id: str) -> None:
        super().__init__()
        self.id = "assess"
        self.store, self.case_id = store, case_id

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        case = self.store.load(self.case_id)
        decisions = case.load_decisions()
        trace = case.load_trace()
        response = _interrupt_response(task)

        if response is None:
            return self._first_pass(case, decisions, trace)

        answers, problems = parse_answers(response, decisions)
        if problems:
            trace.add(Actor.DETERMINISTIC, "Human answer REJECTED", "; ".join(problems),
                      value="question still open")
            case.status = "awaiting_human"
            case.rejected_answer = "; ".join(problems)
            case.store_trace(trace)
            self.store.save(case)
            # The question stays open: re-raise the same interrupt so the case
            # can be answered again, rather than stranding it.
            return self._interrupt(case, decisions, assess_materiality(decisions))

        for index, (group, side) in sorted(answers.items()):
            d = decisions[index]
            parts = []
            if d.group_missing and group != "unknown":
                d.extremity_group, d.group_by = group, Actor.HUMAN
                d.confidence = None
                parts.append(f"extremity group {group}")
            if d.side_missing and side != "unknown":
                d.laterality, d.side_by = side, Actor.HUMAN
                parts.append(f"side {side}")
            trace.add(
                Actor.HUMAN,
                "Answer",
                f"[{index}] {d.condition}",
                value=", ".join(parts) if parts else "does not know",
                evidence="supplied by the reviewer; not stated in the letter",
            )
        case.human_answers = {str(k): "-".join(v) for k, v in sorted(answers.items())}
        case.rejected_answer = None
        case.store_decisions(decisions)
        return self._settle(case, decisions, trace, after_answers=True)

    def _first_pass(self, case: Case, decisions: list[Decision], trace: Trace) -> MultiAgentResult:
        return self._settle(case, decisions, trace, after_answers=False)

    def _settle(self, case: Case, decisions: list[Decision], trace: Trace, *, after_answers: bool) -> MultiAgentResult:
        m = assess_materiality(decisions)
        unknown = [(i, decisions[i]) for i in m.unknown]
        both = [d for d in decisions if d.laterality == "both" and d.extremity_group != "none"]

        if not unknown and not both:
            trace.add(Actor.DETERMINISTIC, "Assessment",
                      "every fact 4.25 and 4.26 need is established", value="ready to compute")
            return self._ready(case, trace)

        if m.settled:
            listed = "; ".join(f"[{i}] {d.condition}: {' and '.join(d.missing)}" for i, d in unknown)
            detail = (
                (f"unknown: {listed}. " if listed else "")
                + (f"a single evaluation names both sides ({both[0].condition}). " if both else "")
                + f"Every one of {len(m.by_answers) * (2 if both else 1)} possible combinations gives "
                f"{m.possible[0]}%, so no answer could change the rating."
            )
            trace.add(Actor.DETERMINISTIC, "Unknown facts cannot change the result", detail,
                      value="no question needed", rule="38 CFR 4.25, 4.26")
            case.immaterial_unknowns = [i for i, _ in unknown]
            return self._ready(case, trace)

        if m.answers_matter and not after_answers:
            case.status = "awaiting_human"
            case.possible_degrees = list(m.possible)
            trace.add(
                Actor.DETERMINISTIC,
                "Question for a reviewer",
                "; ".join(f"[{i}] {d.condition}: {' and '.join(d.missing)} not established" for i, d in unknown)
                + f". The answers lead to different ratings: {_or(m.possible)}.",
                value=f"{len(unknown)} fact(s) needed",
            )
            case.store_trace(trace)
            self.store.save(case)
            return self._interrupt(case, decisions, m)

        # Either the reviewer has answered and something that matters is still
        # unknown, or the only open question is one Recheck does not decide.
        reason = (
            "a single evaluation names both sides, and whether 4.26 includes it changes the result"
            if m.reading_matters and not m.answers_matter
            else "facts that change the result are still not established"
        )
        trace.add(Actor.DETERMINISTIC, "Result UNDETERMINED",
                  f"{reason}. Possible final degrees: {_or(m.possible)}.", value="not computed")
        case.status = "undetermined"
        case.undetermined_reason = reason
        case.possible_degrees = list(m.possible)
        case.store_trace(trace)
        self.store.save(case)
        return _done()

    def _ready(self, case: Case, trace: Trace) -> MultiAgentResult:
        case.status = "ready"
        case.store_trace(trace)
        self.store.save(case)
        return _done()

    def _interrupt(self, case: Case, decisions: list[Decision], m: Materiality) -> MultiAgentResult:
        conditions = []
        for i in m.unknown:
            d = decisions[i]
            outcomes = m.outcomes_for(i) if len(m.unknown) == 1 else {}
            conditions.append({
                "index": i,
                "condition": d.condition,
                "percent": d.percent,
                "evidence": d.evidence,
                "missing": list(d.missing),
                "known": {k: v for k, v in (("extremity group", d.extremity_group), ("side", d.laterality))
                          if v != "unknown"},
                "accepted": accepted_answers(d),
                "why": d.note or "not stated in the letter",
                "outcomes": {"-".join(k): list(v) for k, v in outcomes.items()},
            })
        return MultiAgentResult(
            status=Status.INTERRUPTED,
            interrupts=[
                Interrupt(
                    id=interrupt_id(case.case_id),
                    name="establish_facts",
                    reason={
                        "case": case.case_id,
                        "question": "Recheck needs facts the letter does not establish.",
                        "conditions": conditions,
                        "possible_results": list(m.possible),
                        "stated": case.stated_combined,
                        "rejected": case.rejected_answer,
                        "note": "Answer with facts only - a side or an extremity group. "
                                "Recheck never asks for a percentage or a combined evaluation.",
                    },
                )
            ],
        )


class ComputeNode(MultiAgentBase):
    """DETERMINISTIC. 38 CFR 4.25 / 4.26 and the comparison. No model, ever."""

    def __init__(self, store: CaseStore, case_id: str) -> None:
        super().__init__()
        self.id = "compute"
        self.store, self.case_id = store, case_id

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        case = self.store.load(self.case_id)
        decisions = case.load_decisions()
        trace = case.load_trace()
        evaluation = evaluate_established(decisions)

        if evaluation.bilateral_members:
            trace.add(
                Actor.DETERMINISTIC,
                "Bilateral factor applies",
                "compensable disabilities on both the left and right side of "
                + " and ".join(sorted({m.extremity + " extremities" for m in evaluation.bilateral_members}))
                + ": " + ", ".join(m.label() for m in evaluation.bilateral_members),
                value=f"{len(evaluation.bilateral_members)} disabilities in the factor",
                rule="38 CFR 4.26(a)-(c)",
            )
        for step in evaluation.steps:
            trace.add(Actor.DETERMINISTIC, "Arithmetic", step.detail, value=step.running_after, rule=step.rule)
        for note in evaluation.notes:
            trace.add(Actor.DETERMINISTIC, "Note", note, rule="38 CFR 4.26")
        trace.add(
            Actor.DETERMINISTIC,
            "Final degree of disability",
            f"combined value {evaluation.combined_value} converted to the nearest degree divisible "
            f"by 10; values ending in 5 are adjusted upward",
            value=f"{evaluation.final_degree}%",
            rule="38 CFR 4.25(a)",
        )
        case.recomputed_combined = evaluation.combined_value
        case.recomputed_degree = evaluation.final_degree
        case.bilateral_applied = evaluation.bilateral_applied
        case.bilateral_note = " ".join(evaluation.notes) or None
        case.alternative_degree = evaluation.alternative_final_degree
        case.possible_degrees = [evaluation.final_degree]
        case.status = "complete"
        case.store_trace(trace)
        self.store.save(case)
        return _done()


# ---------------------------------------------------------------------------
# Human answers
# ---------------------------------------------------------------------------

def accepted_answers(decision: Decision) -> list[str]:
    """The answers a reviewer may give for one condition - facts, never numbers."""
    if decision.group_missing and decision.side_missing:
        return ["upper-left", "upper-right", "lower-left", "lower-right", "upper", "lower", "none", "unknown"]
    if decision.group_missing:
        return ["upper", "lower", "none", "unknown"]
    if decision.side_missing:
        return ["left", "right", "unknown"]
    return []


def _interpret(value: str, decision: Decision) -> tuple[str, str]:
    """Map an accepted answer onto (group, side), keeping established facts."""
    group = decision.extremity_group if not decision.group_missing else "unknown"
    side = decision.laterality if not decision.side_missing else "unknown"
    if value == "unknown":
        return group, side
    if value in ("left", "right"):
        return group, value
    if value == "none":
        return "none", side
    if "-" in value:
        g, s = value.split("-", 1)
        return g, s
    return value, side  # "upper" / "lower"


def parse_answers(
    response: Any, decisions: Sequence[Decision]
) -> tuple[dict[int, tuple[str, str]], list[str]]:
    """Validate a reviewer's answers. Returns ({index: (group, side)}, problems).

    Accepts a mapping {"2": "left"} or the command-line form "2=left,3=upper-right".
    Every answer is checked against the case: the condition must exist, must
    have had something unknown, the value must be one of the facts that
    condition could take, and no condition may be answered twice. A fact the
    letter states cannot be overridden, because it is never asked. Numbers
    have no place in the grammar.
    """
    problems: list[str] = []
    pairs: list[tuple[str, str]] = []
    if isinstance(response, dict):
        pairs = [(str(k).strip(), str(v).strip().lower()) for k, v in response.items()]
    elif isinstance(response, str):
        clauses = [c.strip() for c in response.replace(";", ",").split(",") if c.strip()]
        for clause in clauses:
            if "=" not in clause:
                problems.append(f"cannot parse {clause!r}; expected index=answer")
                continue
            raw_index, raw_value = clause.split("=", 1)
            pairs.append((raw_index.strip(), raw_value.strip().lower()))
    else:
        return {}, ["answer must be text such as 2=left or a mapping of index to answer"]
    if not pairs and not problems:
        return {}, ["empty answer"]

    answers: dict[int, tuple[str, str]] = {}
    for raw_index, raw_value in pairs:
        if not raw_index.isdigit():
            problems.append(f"{raw_index!r} is not a condition index")
            continue
        index = int(raw_index)
        if index >= len(decisions):
            problems.append(f"no condition at index {index}")
            continue
        decision = decisions[index]
        allowed = accepted_answers(decision)
        if not allowed:
            stated = f"{decision.extremity_group}, side {decision.laterality}"
            problems.append(f"condition {index} was not asked about; its facts are established ({stated})")
            continue
        if index in answers:
            problems.append(f"condition {index} answered more than once")
            continue
        if raw_value not in allowed:
            problems.append(f"{raw_value!r} is not an accepted answer for condition {index} "
                            f"(accepted: {', '.join(allowed)})")
            continue
        answers[index] = _interpret(raw_value, decision)

    outstanding = [i for i, d in enumerate(decisions) if d.missing and i not in answers]
    if outstanding and not problems:
        problems.append(f"no answer supplied for condition(s) {outstanding}")
    return answers, problems


def _interrupt_response(task: Any) -> Any | None:
    if isinstance(task, list):
        for block in task:
            if isinstance(block, dict) and "interruptResponse" in block:
                return block["interruptResponse"].get("response")
    return None


def _or(values: Sequence[int]) -> str:
    items = [f"{v}%" for v in values]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


# ---------------------------------------------------------------------------
# Gates and observability
# ---------------------------------------------------------------------------

def _extraction_succeeded(state: Any, *, invocation_state: dict | None = None, **kwargs: Any) -> bool:
    """Gate extract -> classify on the extract node's own result in GraphState."""
    result = getattr(state, "results", {}).get("extract")
    return result is not None and result.status == Status.COMPLETED


COMPUTABLE_STATES = ("ready", "complete")


def _safe_to_compute(store: CaseStore, case_id: str):
    """Gate assess -> compute on committed case state.

    Fail-closed: an unreadable case, an open question, an undetermined case,
    a rejected answer - anything but a state in COMPUTABLE_STATES - blocks.
    """

    def condition(state: Any, *, invocation_state: dict | None = None, **kwargs: Any) -> bool:
        try:
            return store.load(case_id).status in COMPUTABLE_STATES
        except Exception:  # noqa: BLE001 - unreadable means blocked
            return False

    condition.__name__ = "safe_to_compute"
    return condition


class NodeTimeline(HookProvider):
    """Records which process ran each graph node, via a Strands node hook.

    The persisted timeline is how a report can show - rather than assert -
    that a case was interrupted in one process and finished in another.
    BeforeNodeCallEvent is used because Strands does not emit
    AfterNodeCallEvent for a node that raised an interrupt.
    """

    def __init__(self, store: CaseStore, case_id: str) -> None:
        self.store, self.case_id = store, case_id

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeNodeCallEvent, self._before)

    def _before(self, event: BeforeNodeCallEvent) -> None:
        try:
            case = self.store.load(self.case_id)
        except Exception:  # noqa: BLE001 - never let observability break a run
            return
        case.timeline.append({"node": event.node_id, "pid": os.getpid()})
        self.store.save(case)


def open_case(store: CaseStore, case_id: str, source: str, classifier: str = "none") -> Case:
    """Create the case a new audit writes into. Callers discard any old one first."""
    case = Case(case_id, str(source), classifier=classifier, status="open")
    store.save(case)
    return case


def build_graph(store: CaseStore, case_id: str, source: str, agent_factory: AgentFactory | None = None):
    """Assemble the graph for one document.

    FileSessionManager restores any persisted graph state - including an
    outstanding interrupt - as the graph is built.
    """
    builder = GraphBuilder()
    builder.add_node(ExtractNode(store, case_id, source), "extract")
    builder.add_node(ClassifyNode(store, case_id, agent_factory), "classify")
    builder.add_node(AssessNode(store, case_id), "assess")
    builder.add_node(ComputeNode(store, case_id), "compute")
    builder.add_edge("extract", "classify", condition=_extraction_succeeded)
    builder.add_edge("classify", "assess")
    builder.add_edge("assess", "compute", condition=_safe_to_compute(store, case_id))
    builder.set_entry_point("extract")
    builder.set_graph_id("recheck")
    builder.set_hook_providers([NodeTimeline(store, case_id)])
    builder.set_session_manager(
        FileSessionManager(session_id=case_id, storage_dir=str(store.session_dir(case_id)))
    )
    return builder.build()


def outstanding_interrupt(graph: Any) -> Interrupt | None:
    """The interrupt a restored graph is waiting on, if any.

    Read from the graph Strands restored from the session - not from a
    file-exists check - so a case whose session was lost cannot be resumed.
    """
    state = getattr(graph, "_interrupt_state", None)
    if state is None or not state.activated:
        return None
    return next(iter(state.interrupts.values()), None)
