"""The Strands Graph: the orchestration boundary.

    ingest -> extract -> classify -> assess -> compute -> report
                                       |
                                  (interrupt)
                                  human supplies laterality only

Why a Graph rather than a function call chain. Two properties are needed
that a plain call chain does not give: the run must be able to STOP at
`assess`, persist, and be resumed by a different process hours later; and
every transition must be individually observable so the ownership boundary
can be shown rather than asserted. Strands provides both - interrupts with
serialised state, and a node topology with an inspectable execution order.

Node ownership, enforced by construction:
  ingest    DETERMINISTIC   file to text
  extract   DETERMINISTIC   text to ratings + stated combined value
  classify  AI (for the gap) extremity group and laterality
  assess    DETERMINISTIC   is this safe to compute, or must a human decide
  compute   DETERMINISTIC   38 CFR 4.25 / 4.26
  report    DETERMINISTIC   evidence-backed comparison

The model appears in exactly one node, cannot reach `compute`, and cannot
express a percentage. The human answers exactly one kind of question - which
side a condition is on - and that answer is validated against the document
and against 4.26 before any arithmetic uses it.
"""

from __future__ import annotations

import pathlib
from typing import Any, Sequence

from strands.agent.agent_result import AgentResult
from strands.interrupt import Interrupt
from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status
from strands.session.file_session_manager import FileSessionManager
from strands.telemetry.metrics import EventLoopMetrics

from recheck.case import Case, CaseStore
from recheck.classify import AgentFactory, Decision, classify
from recheck.extract.deterministic import parse
from recheck.provenance import Actor, Trace

INTERRUPT_NAME = "confirm_laterality"
INTERRUPT_ID = "laterality-1"
VALID_SIDES = ("left", "right", "unknown")


def _node_result(node_id: str, text: str, status: Status = Status.COMPLETED) -> MultiAgentResult:
    """Wrap a plain string as a MultiAgentResult.

    The nesting is not optional: putting a bare string in NodeResult.result
    makes session serialisation fail with
    AttributeError: 'str' object has no attribute 'to_dict'.
    """
    agent_result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=EventLoopMetrics(),
        state={},
    )
    return MultiAgentResult(
        status=status, results={node_id: NodeResult(result=agent_result, status=status)}
    )


def _resume_answer(task: Any) -> str | None:
    """Extract a human response from a resumed invocation, if present."""
    if isinstance(task, list):
        for block in task:
            if isinstance(block, dict) and "interruptResponse" in block:
                return str(block["interruptResponse"].get("response", ""))
    return None


# A rating decision is a handful of pages. These caps are generous by an order
# of magnitude and exist so that a hostile or accidental input cannot exhaust
# memory: an unbounded read of a multi-gigabyte file, or a PDF decompression
# bomb whose page count explodes on parse.
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 100


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


class ScannedDocument(Exception):
    """The document is an image. Recheck refuses rather than guessing."""


class DocumentTooLarge(Exception):
    """The document exceeds the bounds of anything plausibly a decision letter."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class IngestExtractNode(MultiAgentBase):
    """DETERMINISTIC. Document in, ratings out. No model, no network."""

    def __init__(self, store: CaseStore, case_id: str, source: str) -> None:
        super().__init__()
        self.id = "extract"
        self.store = store
        self.case_id = case_id
        self.source = source

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        if _resume_answer(task) is not None:
            # Resuming: extraction already happened in the earlier process.
            return _node_result(self.id, "extraction restored from case file")

        text = read_document(self.source)
        extraction = parse(text)
        trace = Trace()
        if not extraction.ok:
            trace.add(
                Actor.DETERMINISTIC,
                "Extraction FAILED",
                extraction.unparsed_reason or "no ratings found",
                value="cannot proceed",
            )
            case = Case(self.case_id, str(self.source), status="unparsed")
            case.store_trace(trace)
            self.store.save(case)
            return _node_result(self.id, f"extraction failed: {extraction.unparsed_reason}", Status.FAILED)

        case = Case(
            case_id=self.case_id,
            source_path=str(self.source),
            stated_combined=extraction.stated_combined,
            ratings=[
                {
                    "condition": r.condition,
                    "percent": r.percent,
                    "laterality": r.laterality,
                    "extremity_group": r.extremity_group,
                    "source_line": r.source_line,
                    "source_line_number": r.source_line_number,
                }
                for r in extraction.ratings
            ],
        )
        trace.add(
            Actor.DETERMINISTIC,
            "Stated combined evaluation",
            f"as printed in {pathlib.Path(self.source).name}",
            value=f"{extraction.stated_combined}%",
            evidence="combined evaluation statement",
            rule="38 CFR 4.25 (value under review)",
        )
        case.store_trace(trace)
        self.store.save(case)
        return _node_result(self.id, f"extracted {len(extraction.ratings)} ratings")


class ClassifyNode(MultiAgentBase):
    """AI, but only for what the lexicon cannot reach. See recheck.classify."""

    def __init__(self, store: CaseStore, case_id: str, agent_factory: AgentFactory | None) -> None:
        super().__init__()
        self.id = "classify"
        self.store = store
        self.case_id = case_id
        self.agent_factory = agent_factory

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        if _resume_answer(task) is not None:
            return _node_result(self.id, "classifications restored from case file")

        case = self.store.load(self.case_id)
        from recheck.extract.deterministic import ExtractedRating

        ratings = [ExtractedRating(**r) for r in case.ratings]
        trace = case.load_trace()
        decisions = classify(ratings, trace, self.agent_factory)
        case.store_decisions(decisions)
        case.store_trace(trace)
        self.store.save(case)
        ai_count = sum(1 for d in decisions if d.decided_by is Actor.AI)
        return _node_result(self.id, f"classified {len(decisions)} conditions ({ai_count} by model)")


class AssessNode(MultiAgentBase):
    """DETERMINISTIC gatekeeper. Raises the interrupt; validates the answer.

    This node decides whether the case is safe to compute. It is the only
    place a human is asked anything, and the only thing it asks for is a
    side. It never asks a human for a percentage or a combined value.
    """

    def __init__(self, store: CaseStore, case_id: str) -> None:
        super().__init__()
        self.id = "assess"
        self.store = store
        self.case_id = case_id

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        case = self.store.load(self.case_id)
        decisions = case.load_decisions()
        trace = case.load_trace()
        answer = _resume_answer(task)

        if answer is None:
            unresolved = [(i, d) for i, d in enumerate(decisions) if d.needs_human]
            if not unresolved:
                trace.add(
                    Actor.DETERMINISTIC,
                    "Ambiguity assessment",
                    "every condition resolved without human input; no interrupt raised",
                    value="clear to compute",
                )
                case.store_trace(trace)
                self.store.save(case)
                return _node_result(self.id, "no human input required")

            trace.add(
                Actor.DETERMINISTIC,
                "Ambiguity assessment",
                "; ".join(f"[{i}] {d.condition}: {d.reason}" for i, d in unresolved),
                value=f"{len(unresolved)} condition(s) need a human",
            )
            case.store_trace(trace)
            case.status = "awaiting_human"
            self.store.save(case)
            return MultiAgentResult(
                status=Status.INTERRUPTED,
                results={},
                interrupts=[
                    Interrupt(
                        id=INTERRUPT_ID,
                        name=INTERRUPT_NAME,
                        reason={
                            "question": "Which side is each of these conditions on?",
                            "conditions": [
                                {"index": i, "condition": d.condition, "percent": d.percent,
                                 "extremity_group": d.extremity_group, "why": d.reason}
                                for i, d in unresolved
                            ],
                            "accepted_values": list(VALID_SIDES),
                            "note": "Recheck asks only for laterality. It never asks a human "
                                    "for a percentage or a combined evaluation.",
                        },
                    )
                ],
            )

        # Resuming with a human answer. Validate before it touches anything.
        sides, problems = parse_sides(answer, decisions)
        if problems:
            trace.add(
                Actor.DETERMINISTIC,
                "Human answer REJECTED",
                "; ".join(problems),
                value="cannot proceed",
            )
            case.store_trace(trace)
            case.status = "invalid_answer"
            self.store.save(case)
            return _node_result(self.id, "invalid human answer: " + "; ".join(problems), Status.FAILED)

        for index, side in sides.items():
            decision = decisions[index]
            trace.add(
                Actor.HUMAN,
                "Laterality confirmed",
                f"{decision.condition}",
                value=side,
                evidence="human confirmation, not present in the document",
            )
            decisions[index] = Decision(
                condition=decision.condition,
                percent=decision.percent,
                extremity_group=decision.extremity_group,
                laterality=side,
                decided_by=Actor.HUMAN,
                confidence=None,
                needs_human=(side == "unknown"),
                reason=None if side != "unknown" else "human could not determine the side",
                evidence=decision.evidence,
            )
        case.human_answers = {str(k): v for k, v in sides.items()}
        case.store_decisions(decisions)
        case.store_trace(trace)
        case.status = "resumed"
        self.store.save(case)
        return _node_result(self.id, f"human resolved {len(sides)} condition(s)")


class ComputeReportNode(MultiAgentBase):
    """DETERMINISTIC. 38 CFR 4.25 / 4.26 and the comparison. No model, ever."""

    def __init__(self, store: CaseStore, case_id: str) -> None:
        super().__init__()
        self.id = "compute"
        self.store = store
        self.case_id = case_id

    async def invoke_async(self, task: Any, invocation_state: dict | None = None, **kw: Any) -> MultiAgentResult:
        from recheck.cfr.rating import evaluate

        case = self.store.load(self.case_id)
        decisions = case.load_decisions()
        trace = case.load_trace()

        percents = [d.percent for d in decisions]
        pair = choose_bilateral_pair(decisions)
        if pair is not None:
            i, j = pair
            trace.add(
                Actor.DETERMINISTIC,
                "Bilateral pair identified",
                f"{decisions[i].condition} ({decisions[i].laterality}) and "
                f"{decisions[j].condition} ({decisions[j].laterality}): same extremity group, "
                f"opposite sides, both compensable",
                value=f"{decisions[i].percent}% + {decisions[j].percent}%",
                rule="38 CFR 4.26(a), 4.26(c)",
            )
            evaluation = evaluate(percents, bilateral_pair=[decisions[i].percent, decisions[j].percent])
        else:
            trace.add(
                Actor.DETERMINISTIC,
                "Bilateral factor not applied",
                "no pair of compensable conditions in the same extremity group on opposite sides "
                "was established",
                value="4.26 not applicable",
                rule="38 CFR 4.26(c)",
            )
            evaluation = evaluate(percents)

        for step in evaluation.steps:
            trace.add(
                Actor.DETERMINISTIC,
                step.detail.split(":")[0][:60],
                step.detail,
                value=step.running_after,
                rule=step.rule,
            )
        trace.add(
            Actor.DETERMINISTIC,
            "Final degree of disability",
            "combined value converted to the nearest degree divisible by 10; values ending in 5 "
            "are adjusted upward",
            value=f"{evaluation.final_degree}%",
            rule="38 CFR 4.25(a)",
        )
        case.recomputed_combined = evaluation.combined_value
        case.recomputed_degree = evaluation.final_degree
        case.bilateral_applied = evaluation.bilateral_applied
        case.bilateral_note = " ".join(evaluation.notes) or None
        case.alternative_degree = evaluation.alternative_final_degree
        case.store_trace(trace)
        case.status = "complete"
        self.store.save(case)
        return _node_result(self.id, f"recomputed {evaluation.final_degree}%")


# ---------------------------------------------------------------------------
# Validation of the human answer
# ---------------------------------------------------------------------------

def parse_sides(
    answer: str, decisions: Sequence[Decision]
) -> tuple[dict[int, str], list[str]]:
    """Parse and validate a human laterality answer.

    Accepts "0=left,1=right". Every clause is checked against the case: the
    index must exist, it must be a condition that actually asked for input,
    and the value must be a permitted side. A human cannot supply a
    percentage here because the grammar has no place for one.
    """
    sides: dict[int, str] = {}
    problems: list[str] = []
    clauses = [c.strip() for c in answer.replace(";", ",").split(",") if c.strip()]
    if not clauses:
        return {}, ["empty answer"]

    for clause in clauses:
        if "=" not in clause:
            problems.append(f"cannot parse {clause!r}; expected index=side")
            continue
        raw_index, raw_side = clause.split("=", 1)
        raw_index, raw_side = raw_index.strip(), raw_side.strip().lower()
        if not raw_index.isdigit():
            problems.append(f"{raw_index!r} is not a condition index")
            continue
        index = int(raw_index)
        if index < 0 or index >= len(decisions):
            problems.append(f"no condition at index {index}")
            continue
        if not decisions[index].needs_human:
            problems.append(f"condition {index} did not require human input")
            continue
        if raw_side not in VALID_SIDES:
            problems.append(f"{raw_side!r} is not one of {VALID_SIDES}")
            continue
        sides[index] = raw_side

    outstanding = [i for i, d in enumerate(decisions) if d.needs_human and i not in sides]
    if outstanding and not problems:
        problems.append(f"no answer supplied for condition(s) {outstanding}")
    return sides, problems


def choose_bilateral_pair(decisions: Sequence[Decision]) -> tuple[int, int] | None:
    """Deterministically select a 4.26 pair from resolved decisions.

    Requirements, all from the regulation:
      - both conditions in the same extremity group (4.26(a))
      - opposite sides
      - both compensable, i.e. at least 10 percent (4.26(c))
      - neither still flagged for human review
    """
    eligible = [
        (i, d)
        for i, d in enumerate(decisions)
        if d.extremity_group in ("upper", "lower") and d.safe_for_pairing and d.percent >= 10
    ]
    for a_index, a in eligible:
        for b_index, b in eligible:
            if b_index <= a_index:
                continue
            if a.extremity_group != b.extremity_group:
                continue
            if {a.laterality, b.laterality} == {"left", "right"}:
                return a_index, b_index
    return None


# ---------------------------------------------------------------------------
# The safety gate between assessment and arithmetic
# ---------------------------------------------------------------------------

#: Case states in which arithmetic must NOT be produced.
BLOCKED_STATES = ("unparsed", "invalid_answer", "awaiting_human")


def _safe_to_compute(store: CaseStore, case_id: str):
    """Build the edge condition guarding `compute`.

    Fail-closed: any state that is not positively known to be computable
    blocks the edge, including a case file that cannot be read.
    """

    def condition(state: Any, *, invocation_state: dict | None = None, **kwargs: Any) -> bool:
        try:
            case = store.load(case_id)
        except Exception:
            return False
        return case.status not in BLOCKED_STATES

    condition.__name__ = "safe_to_compute"
    return condition


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(
    store: CaseStore,
    case_id: str,
    source: str,
    agent_factory: AgentFactory | None = None,
):
    """Assemble the graph.

    The session manager is attached to the ORCHESTRATOR only. Attaching one
    to each member node is not supported for multi-agent runs in this SDK
    version.
    """
    builder = GraphBuilder()
    builder.add_node(IngestExtractNode(store, case_id, source), "extract")
    builder.add_node(ClassifyNode(store, case_id, agent_factory), "classify")
    builder.add_node(AssessNode(store, case_id), "assess")
    builder.add_node(ComputeReportNode(store, case_id), "compute")
    builder.add_edge("extract", "classify")
    builder.add_edge("classify", "assess")
    # CONDITIONAL EDGE - a safety gate, not decoration.
    #
    # Returning Status.FAILED from `assess` does NOT stop downstream nodes in
    # this SDK version: a rejected human answer still reached `compute` and
    # produced arithmetic. That is the worst possible failure for this
    # product, so the gate is now part of the topology and is enforced by
    # reading committed state from disk rather than by in-memory convention.
    builder.add_edge("assess", "compute", condition=_safe_to_compute(store, case_id))
    builder.set_entry_point("extract")
    builder.set_max_node_executions(12)
    builder.set_session_manager(
        FileSessionManager(session_id=case_id, storage_dir=str(store.session_dir(case_id)))
    )
    return builder.build()
