"""Deep review, human-in-the-loop lane: the reviewer's answer and the Strands session.

Each test fails on 12705f8, the code before its fix.

  HITL-1  resume accepted a session whose question was not waiting at assess.
          Edited to mark the interrupt as raised by a hook, the reviewer's typed
          answer never reached assess and an answer written into the session's
          task was used instead: NO DISCREPANCY FOUND on facts the reviewer did
          not give, exit 0. Other edits dropped the answer silently (exit 2) or
          ran compute twice and left the answered case unreadable.

(HITL-3, a fact volunteered for a 0% rating re-asking the question, was already
fixed on integ and is covered by tests/test_rt_verify.py RG11-REFINED-ZERO.)
"""

from __future__ import annotations

import json
import pathlib

import pytest

import recheck.graph as graph
from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    main,
    nodes_run,
)
from recheck.case import CaseStore

MISSING_SIDES = LETTERS / "05_missing_side.txt"  # [1] and [2] knees, sides unstated: 70% or 80%


def _edit_session(store_root: pathlib.Path, case_id: str, change) -> None:
    (path,) = CaseStore(store_root).session_dir(case_id).rglob("multi_agent.json")
    state = json.loads(path.read_text(encoding="utf-8"))
    change(state, state["_internal_state"]["interrupt_state"]["context"])
    path.write_text(json.dumps(state), encoding="utf-8")


# ==========================================================================
# HITL-1: the question must be waiting at assess, raised by assess
# ==========================================================================

def _raised_by_a_hook_with_an_answer_in_the_task(state, context):
    context["assess"]["from_hook"] = True
    state["current_task"] = [{"interruptResponse": {"interruptId": graph.interrupt_id("c"),
                                                    "response": {"1": "left", "2": "left"}}}]


def _routed_to_no_interrupt(state, context):
    context["assess"]["interrupt_ids"] = []


def _classify_left_to_run_again(state, context):
    context["completed_nodes"] = ["classify"]


def _compute_interrupted_beside_assess(state, context):
    state["interrupted_nodes"] = ["assess", "compute"]
    context["compute"] = dict(context["assess"])


@pytest.mark.parametrize("tamper", [
    _raised_by_a_hook_with_an_answer_in_the_task,
    _routed_to_no_interrupt,
    _classify_left_to_run_again,
    _compute_interrupted_beside_assess,
])
def test_a_session_whose_question_is_not_waiting_at_assess_is_not_resumed(tmp_path, tamper):
    store_root = tmp_path / "runs"
    assert main("audit", MISSING_SIDES, "--case", "c", store=store_root)[0] == EXIT_AWAITING_HUMAN
    _edit_session(store_root, "c", tamper)

    code, out, err = main("resume", "--case", "c", "--answer", "1=left,2=right", store=store_root)

    case = CaseStore(store_root).load("c")  # still readable: nothing ran twice
    assert code == EXIT_CANNOT_PROCEED, (code, err)
    assert "holds no open question" in err
    assert (case.status, case.recomputed_degree, case.human_answers) == ("awaiting_human", None, {})
    assert "NO DISCREPANCY FOUND" not in out and "POTENTIAL DISCREPANCY" not in out
    assert nodes_run(CaseStore(store_root), "c") == ["extract", "classify", "assess"]


def test_the_reviewers_typed_answer_is_what_a_genuine_session_uses(tmp_path):
    store_root = tmp_path / "runs"
    assert main("audit", MISSING_SIDES, "--case", "c", store=store_root)[0] == EXIT_AWAITING_HUMAN
    code, out, _ = main("resume", "--case", "c", "--answer", "1=up", store=store_root)
    assert code == EXIT_CANNOT_PROCEED and "not accepted" in out  # a rejection keeps the session resumable

    code, out, _ = main("resume", "--case", "c", "--answer", "1=left,2=right", store=store_root)
    case = CaseStore(store_root).load("c")
    assert code == EXIT_OK
    assert (case.recomputed_degree, case.human_answers) == (80, {"1": "lower-left", "2": "lower-right"})


def test_an_answer_written_into_a_genuinely_shaped_session_does_not_replace_the_typed_one(tmp_path):
    """Guard (passes before and after): with the session in the shape assess
    leaves it, Strands overwrites a pre-written response with the typed one."""
    store_root = tmp_path / "runs"
    assert main("audit", MISSING_SIDES, "--case", "c", store=store_root)[0] == EXIT_AWAITING_HUMAN
    written = {"interruptResponse": {"interruptId": graph.interrupt_id("c"), "response": {"1": "left", "2": "left"}}}

    def prefill(state, context):
        state["_internal_state"]["interrupt_state"]["interrupts"][graph.interrupt_id("c")]["response"] = {
            "1": "left", "2": "left"}
        context["responses"] = [written]
        state["current_task"] = [written]

    _edit_session(store_root, "c", prefill)
    code, _, err = main("resume", "--case", "c", "--answer", "1=left,2=right", store=store_root)
    case = CaseStore(store_root).load("c")
    assert code == EXIT_OK, err
    assert (case.recomputed_degree, case.human_answers) == (80, {"1": "lower-left", "2": "lower-right"})
