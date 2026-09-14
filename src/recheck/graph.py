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

import hashlib
import os
import pathlib
from typing import Any, Sequence

from strands.hooks import BeforeNodeCallEvent, HookProvider, HookRegistry
from strands.interrupt import Interrupt
from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, Status
from strands.session.file_session_manager import FileSessionManager

from recheck.case import Case, CaseStore
from recheck.cfr.rating import COMPENSABLE_MINIMUM
from recheck.classify import AgentFactory, Decision, classify_async
from recheck.extract import pdf_text
from recheck.extract.deterministic import ExtractedRating, parse
from recheck.materiality import Materiality, assess as assess_materiality, evaluate_for_report
from recheck.provenance import Actor, Trace

NODE_ORDER = ("extract", "classify", "assess", "compute")

# A rating decision is a handful of pages. These caps are generous by an order
# of magnitude and exist so that a hostile or accidental input cannot exhaust
# memory: an unbounded read of a multi-gigabyte file, or a PDF decompression
# bomb whose page count explodes on parse.
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 100
# Bytes and pages do not bound the work. A 68 KB one-page PDF whose content
# stream inflates to 70 MB passed both caps, then pypdf and the parser grew
# to 5-6 GB over more than ten minutes. So the text a document may yield is
# capped (a hundred dense pages), and so is what any one PDF stream may
# inflate to - pypdf's own default is 75 MB per stream, far beyond a letter -
# and what all page content streams together may inflate to: pypdf spends
# about 3 s per MB parsing page content whether or not it draws any text, so
# a hundred pages each just under the per-stream cap ran for over 5 minutes.
MAX_DOCUMENT_CHARS = 500_000
MAX_PDF_STREAM_BYTES = 2_000_000
MAX_PDF_CONTENT_BYTES = 4_000_000
# And none of these bounds the time pypdf spends on a Form XObject invoked
# again and again: a 2.7 KB PDF took 95 s. PDF text is extracted in a child
# process that is killed at this budget. A real letter reads in well under a
# second, child start-up included.
MAX_PDF_SECONDS = 20.0


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
    return _document_text(p, _document_bytes(p))


def _too_large(name: str, size: int) -> DocumentTooLarge:
    return DocumentTooLarge(
        f"{name} is {size / 1024 / 1024:.1f} MB, over the "
        f"{MAX_DOCUMENT_BYTES / 1024 / 1024:.0f} MB limit. A rating decision is a "
        f"few pages; refusing rather than loading it."
    )


def _document_bytes(p: pathlib.Path) -> bytes:
    """The document's bytes, read once and never more than the cap.

    Text is decoded from these same bytes, and the case records their digest,
    so the digest describes exactly what was parsed - not a second read of a
    file that may have changed in between.
    """
    size = p.stat().st_size  # raises FileNotFoundError for a missing path
    if size > MAX_DOCUMENT_BYTES:
        raise _too_large(p.name, size)
    with p.open("rb") as handle:
        data = handle.read(MAX_DOCUMENT_BYTES + 1)  # the file may have grown since stat()
    if len(data) > MAX_DOCUMENT_BYTES:
        raise _too_large(p.name, len(data))
    return data


def _document_text(p: pathlib.Path, data: bytes) -> str:
    too_long = (f"{p.name} yields more than {MAX_DOCUMENT_CHARS:,} characters of text. A rating "
                f"decision is a few pages; refusing rather than parsing it.")
    if p.suffix.lower() == ".pdf":
        # pypdf runs in a child process with a wall-clock budget; see
        # recheck.extract.pdf_text for why no cap alone could bound it.
        limits = pdf_text.Limits(max_pages=MAX_PDF_PAGES, max_chars=MAX_DOCUMENT_CHARS,
                                 max_stream_bytes=MAX_PDF_STREAM_BYTES, max_content_bytes=MAX_PDF_CONTENT_BYTES)
        outcome = pdf_text.extract(data, p.name, limits, MAX_PDF_SECONDS)
        if outcome.kind == "timeout":
            raise DocumentTooLarge(
                f"{p.name} took more than {MAX_PDF_SECONDS:g} s to read. A rating decision is a few pages; "
                f"refusing rather than continuing to parse it."
            )
        if outcome.kind == "died":
            # An OSError, so the extract node records "could not be read".
            raise ChildProcessError(f"the PDF reader stopped without a result ({outcome.value})")
        if outcome.kind == "error":
            from pypdf.errors import PdfReadError, PyPdfError

            if isinstance(outcome.value, (PyPdfError, OSError)):
                raise outcome.value  # what pypdf raised, as in-process
            # pypdf also fails on a malformed file with KeyError (a Type0 font
            # without /DescendantFonts), NotImplementedError (an unknown
            # filter), RecursionError (deeply nested arrays) and others. Raised
            # as they were, they escaped the extract node: the case stayed
            # 'open' with no reason and read as a stopped run. Reading the PDF
            # failed; say so.
            raise PdfReadError(f"{type(outcome.value).__name__}: {outcome.value}") from outcome.value  # type: ignore[misc]
        if outcome.kind == "too_large":
            raise DocumentTooLarge(str(outcome.value))
        if outcome.kind == "invisible":
            raise ScannedDocument(str(outcome.value))
        # pypdf decodes a font's ToUnicode map as UTF-16 code units, so a
        # malformed map leaves a lone surrogate in the text. It is not a
        # character: the Strands session could not write a question naming the
        # condition (the case was left awaiting an answer nobody could give),
        # and a finished report could not be printed. Pairs are joined; a lone
        # one becomes U+FFFD, which no percentage or side is read from.
        text = str(outcome.value).encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        if not text.strip():
            raise ScannedDocument(
                f"{p.name} has no extractable text layer. This document requires OCR, "
                f"which Recheck deliberately does not perform. Supply a text-layer PDF "
                f"or a .txt transcript."
            )
        return text
    # Universal newlines, as Path.read_text gives: a bare CR is a line break.
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise DocumentTooLarge(too_long)
    return text


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
        # A refused document is recorded on the case, not only raised. Raised,
        # the reason ("no extractable text layer", "over the 8 MB limit")
        # reached the first run's output and nowhere else: the case stayed
        # "open", and a second sweep over the same store reported it as "the
        # run stopped with the case in state 'open'" - a different triage,
        # and one that reads like a crash.
        # Only READING the document is a refusal. A defect in parse() is a
        # defect in Recheck and must not be reported as "could not read the
        # letter", so it is left to surface.
        from pypdf.errors import PyPdfError

        path = pathlib.Path(self.source)
        data: bytes | None = None
        try:
            data = _document_bytes(path)
            text = _document_text(path, data)
        except (ScannedDocument, DocumentTooLarge) as exc:
            text, reason = None, str(exc)
        except (OSError, UnicodeDecodeError, PyPdfError) as exc:
            text, reason = None, f"{path.name} could not be read ({type(exc).__name__}: {exc})"
        extraction = parse(text) if text is not None else None
        if extraction is not None:
            reason = extraction.unparsed_reason or "no ratings found"
        # The caller opens the case (recording the classifier); extraction
        # starts its facts and trace afresh.
        case = self.store.load(self.case_id)
        # The digest of the bytes this case was read from - including a
        # letter that was refused - so resume, show and sweep can tell when
        # the letter on disk is no longer the one the case describes.
        case.document_sha256 = hashlib.sha256(data).hexdigest() if data is not None else None
        trace = Trace()
        if extraction is None or not extraction.ok:
            trace.add(Actor.DETERMINISTIC, "Extraction FAILED", reason, value="cannot proceed")
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

        refined = []
        # Only a condition that was asked can be refined into a follow-up: a
        # volunteered partial answer for a 0% one (never asked, see asked())
        # re-asked the unchanged question instead of ending the round.
        was_asked = set(asked(decisions))
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
            if parts and d.missing and index in was_asked:
                refined.append(index)
            trace.add(
                Actor.HUMAN,
                "Answer",
                f"[{index}] {d.condition}",
                value=", ".join(parts) if parts else "does not know",
                evidence="supplied by the reviewer; not stated in the letter",
            )
        # Answers accumulate across rounds. A follow-up question (see _settle)
        # asks only for what is still missing, so keeping this round's answers
        # alone would drop the earlier ones from the case and from the basis
        # the report gives for its figure.
        case.human_answers = {**case.human_answers,
                              **{str(k): "-".join(v) for k, v in sorted(answers.items())}}
        case.rejected_answer = None
        case.store_decisions(decisions)
        return self._settle(case, decisions, trace, after_answers=True, refined=bool(refined))

    def _first_pass(self, case: Case, decisions: list[Decision], trace: Trace) -> MultiAgentResult:
        return self._settle(case, decisions, trace, after_answers=False)

    def _settle(self, case: Case, decisions: list[Decision], trace: Trace, *, after_answers: bool,
                refined: bool = False) -> MultiAgentResult:
        """Route the case on what the unknown facts can do to the rating.

        `refined` means an answer this round established one fact of a
        condition and left another missing - the reviewer gave the group
        ("lower"), an answer the question itself offers, but not the side.
        Such a case is asked again for what remains. It used to go straight
        to UNDETERMINED, a terminal state: the reviewer who knew the side
        next could not give it without discarding the case.
        """
        m = assess_materiality(decisions)
        unknown = [(i, decisions[i]) for i in m.unknown]
        both = [d for d in decisions if d.laterality == "both" and d.extremity_group != "none"]

        if not m.exhaustive:
            # Too many possibilities to try them all, or more arm and leg
            # ratings than the 4.26(d) search is verified for - so there are
            # no possible degrees. The branches below were reached with that
            # empty set and phrasing it raised IndexError inside this node,
            # leaving the case stuck in 'classified'. Not sampled, and not
            # asked blind - nobody can say which answers would matter, or what
            # they would lead to. This comes before the every-fact-established
            # branch: thirteen arm and leg ratings with every side stated went
            # on to compute, where the engine's refusal raised instead.
            return _record_unenumerated(self.store, case, trace, m)

        if not unknown and not both:
            trace.add(Actor.DETERMINISTIC, "Assessment",
                      "every fact 4.25 and 4.26 need is established", value="ready to compute")
            return self._ready(case, trace)

        if m.settled:
            listed = "; ".join(f"[{i}] {d.condition}: {' and '.join(d.missing)}" for i, d in unknown)
            detail = (
                (f"unknown: {listed}. " if listed else "")
                + (f"a single evaluation names both sides ({both[0].condition}). " if both else "")
                + f"Every one of {m.combinations} possible combinations gives "
                f"{m.possible[0]}%, so no answer could change the rating."
            )
            trace.add(Actor.DETERMINISTIC, "Unknown facts cannot change the result", detail,
                      value="no question needed", rule="38 CFR 4.25, 4.26")
            case.immaterial_unknowns = [i for i, _ in unknown]
            return self._ready(case, trace)

        if m.answers_matter and (not after_answers or refined):
            case.status = "awaiting_human"
            case.possible_degrees = list(m.possible)
            questions = [(i, decisions[i]) for i in asked(decisions)]
            trace.add(
                Actor.DETERMINISTIC,
                "Question for a reviewer",
                "; ".join(f"[{i}] {d.condition}: {' and '.join(d.missing)} not established" for i, d in questions)
                + f". The answers lead to different ratings: {_or(m.possible)}.",
                value=f"{sum(len(d.missing) for _, d in questions)} fact(s) needed for {len(questions)} condition(s)",
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
        questions = asked(decisions)
        for i in questions:
            d = decisions[i]
            outcomes = m.outcomes_for(i) if len(questions) == 1 else {}
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
        trace = case.load_trace()
        if case.status not in COMPUTABLE_STATES:
            # The assess -> compute edge is the gate, but the persisted Strands
            # session names the node a resumed run continues at. A hand-edited
            # session naming "compute" ran the arithmetic with the question
            # still open and reported NO DISCREPANCY FOUND, exit 0. So this
            # node checks committed state for itself.
            trace.add(Actor.DETERMINISTIC, "Arithmetic REFUSED",
                      f"the case is {case.status!r}, not ready to compute", value="not computed")
            case.store_trace(trace)
            self.store.save(case)
            return _done(Status.FAILED)
        decisions = case.load_decisions()
        m = assess_materiality(decisions)
        if not m.exhaustive:
            # Assess never marks such a case ready, but this node checks for
            # itself (see above): a hand-edited 'ready' case with thirteen arm
            # and leg ratings raised the engine's ValueError here instead of
            # ending without a figure.
            return _record_unenumerated(self.store, case, trace, m)
        if not m.settled:
            # Assess never marks such a case ready. A hand-edited 'ready' case
            # whose facts leave the result open (an unknown fact that matters,
            # or a single both-sides evaluation M21-1 leaves open) was computed
            # and written 'complete' with one of its possible figures - a
            # result the load then refused. No figure is written.
            trace.add(Actor.DETERMINISTIC, "Arithmetic REFUSED",
                      f"the facts on file could change the result ({_or(m.possible)}); "
                      f"the case is not ready to compute", value="not computed")
            case.store_trace(trace)
            self.store.save(case)
            return _done(Status.FAILED)
        evaluation, assumed = evaluate_for_report(decisions)
        for index, (group, side) in sorted(assumed.items()):
            if group not in ("upper", "lower"):
                shown = "not an arm or leg"
            elif side == "both":
                shown = f"both {group} extremities"
            else:
                shown = f"the {side} {group} extremity"
            trace.add(
                Actor.DETERMINISTIC,
                "Arithmetic shown with an assumed fact",
                f"[{index}] {decisions[index].condition}: not established by the letter. Every possibility "
                f"gives the same final degree, so the arithmetic below is shown as if it were {shown}; "
                f"the result does not depend on it.",
                value="does not change the result",
                rule="38 CFR 4.25, 4.26",
            )

        if evaluation.excluded_under_426d:
            trace.add(
                Actor.DETERMINISTIC,
                "38 CFR 4.26(d) decides this result",
                "4.26(d) took effect April 16, 2023 (88 FR 22914). For a period before that date the "
                "prior rule applied the bilateral factor without exception, and VA has stated that "
                "evaluations under the prior rule were not in error (88 FR 89307).",
                value="check the period the decision covers",
                rule="38 CFR 4.26(d)",
            )
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


def _record_unenumerated(store: CaseStore, case: Case, trace: Trace, m: Materiality) -> MultiAgentResult:
    """End a case whose possibilities could not all be evaluated: UNDETERMINED, no figure.

    Shared by assess and compute so that both reach the same terminal state
    with the same reason. m.limit is a whole reason (materiality.assess
    writes it for the case file and the report).
    """
    trace.add(Actor.DETERMINISTIC, "Result UNDETERMINED",
              f"{m.limit}. Recheck neither samples the possibilities nor truncates the 4.26(d) search.",
              value="not computed")
    case.status = "undetermined"
    case.undetermined_reason = m.limit
    case.possible_degrees = []
    case.store_trace(trace)
    store.save(case)
    return _done()


# ---------------------------------------------------------------------------
# Human answers
# ---------------------------------------------------------------------------

def asked(decisions: Sequence[Decision]) -> list[int]:
    """The conditions a reviewer is asked about: those with a fact missing that could matter.

    A non-compensable (0%) evaluation is never part of the bilateral factor
    (38 CFR 4.26(c)), so no answer about it can change a rating. It used to
    be listed in the question and required in every answer: "1=left,2=right"
    for the two knees was refused because the 0% scar had no answer. It may
    still be answered; it is never required.
    """
    return [i for i, d in enumerate(decisions) if d.missing and d.percent >= COMPENSABLE_MINIMUM]


def accepted_answers(decision: Decision) -> list[str]:
    """The answers a reviewer may give for one condition - facts, never numbers."""
    if decision.group_missing and decision.side_missing:
        return ["upper-left", "upper-right", "upper-both", "lower-left", "lower-right", "lower-both",
                "upper", "lower", "none", "unknown"]
    if decision.group_missing:
        return ["upper", "lower", "none", "unknown"]
    if decision.side_missing:
        # "both": one evaluation covering both extremities, e.g. plantar
        # fasciitis rated "unilateral or bilateral" under DC 5269.
        return ["left", "right", "both", "unknown"]
    return []


def _interpret(value: str, decision: Decision) -> tuple[str, str]:
    """Map an accepted answer onto (group, side), keeping established facts."""
    group = decision.extremity_group if not decision.group_missing else "unknown"
    side = decision.laterality if not decision.side_missing else "unknown"
    if value == "unknown":
        return group, side
    if value in ("left", "right", "both"):
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
        # Only text is an answer. str() made a JSON null into "None", which
        # lowercases to the accepted answer "none": an integrator sending
        # null for "no answer" told Recheck the condition is not an arm or a
        # leg, and a figure was reported on a fact nobody supplied.
        for k, v in response.items():
            if isinstance(k, bool) or not isinstance(k, (str, int)):
                problems.append(f"{k!r} is not a condition index")
            elif not isinstance(v, str):
                problems.append(f"the answer for condition {k} must be text such as \"left\", "
                                f"not {type(v).__name__}")
            else:
                pairs.append((str(k).strip(), v.strip().lower()))
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
        # ASCII only: str.isdigit() is also true for "²", which int() then
        # refuses. That ValueError escaped the assess node, failed the graph
        # and left the case unable to accept a correct answer afterwards.
        if not (raw_index.isascii() and raw_index.isdigit()):
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

    outstanding = [i for i in asked(decisions) if i not in answers]
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
    if not items:
        # Never expected (an unenumerated case does not reach a caller), but
        # this runs inside a graph node, where an IndexError strands the case.
        return "not established"
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


# The Strands session id. The session directory already names the case
# (<store>/<case>/session), so the id need not repeat it. It used to be the
# case id, which put the id in the path twice - <case>\session\session_<case>
# \multi_agents\... - and on Windows a 64-character id, which the validator
# accepts and sweep derives from ordinary file names, overran the 260-character
# path limit under a store of about 60 characters.
SESSION_ID = "s"

# The longest name Strands writes under the session: FileSessionManager saves
# through tempfile.mkstemp(prefix=".strands_", suffix=".tmp"), eight random
# characters, in the multi-agent directory.
_STRANDS_TEMP_NAME = ".strands_" + "x" * 8 + ".tmp"

# Windows MAX_PATH is 260 including the terminating NUL.
_WINDOWS_MAX_PATH = 259


def deepest_session_path(store: CaseStore, case_id: str) -> str:
    """The longest path the graph writes for this case, as the OS will see it."""
    return os.path.join(
        os.path.abspath(store.session_dir(case_id)), f"session_{SESSION_ID}", "multi_agents",
        "multi_agent_recheck", _STRANDS_TEMP_NAME,
    )


def _path_limit() -> int | None:
    """The longest path this process can create, when the OS imposes one we can hit."""
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            if winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1:
                return None
    except OSError:
        pass
    return _WINDOWS_MAX_PATH


def open_case(store: CaseStore, case_id: str, source: str, classifier: str = "none") -> Case:
    """Create the case a new audit writes into. Callers discard any old one first.

    Refuses, before writing anything, a case whose session files would not
    fit the platform's path limit. Found out later, the session write failed
    part-way: the case was left 'open', the report blamed the letter ("could
    not establish enough facts") and a second sweep called it a stopped run.
    """
    limit = _path_limit()
    deepest = deepest_session_path(store, case_id)
    if limit is not None and len(deepest) > limit:
        raise ValueError(
            f"the store path and case id are too long for Windows: this case's session files need a "
            f"{len(deepest)}-character path and {limit} is the limit. Use a shorter --store or case id."
        )
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
        FileSessionManager(session_id=SESSION_ID, storage_dir=str(store.session_dir(case_id)))
    )
    return builder.build()


def outstanding_interrupt(graph: Any) -> Interrupt | None:
    """The interrupt a restored graph is waiting on, if any.

    Read from the graph Strands restored from the session - not from a
    file-exists check - so a case whose session was lost cannot be resumed.

    An activated interrupt is not enough. A resume that raised or was
    interrupted part-way (a locked case file, Ctrl+C) leaves the session with
    the interrupt still activated, already holding that attempt's response,
    but no node waiting on it. Strands resumes such a session by running
    nothing, so every later resume - correct answer or garbage - printed the
    same question again and changed nothing, forever. The graph must be
    INTERRUPTED with a node to continue at.
    """
    state = getattr(graph, "_interrupt_state", None)
    if state is None or not state.activated:
        return None
    graph_state = getattr(graph, "state", None)
    if graph_state is None or graph_state.status != Status.INTERRUPTED or not graph_state.interrupted_nodes:
        return None
    return next(iter(state.interrupts.values()), None)
