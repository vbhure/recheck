"""The gates between the nodes, and why they are shaped the way they are.

Returning Status.FAILED from a node does not stop downstream nodes in this
Strands version, so conditional edges do the stopping:

    extract --(the extract node's own result COMPLETED)--> classify
    assess  --(case status "ready" or "complete")--------> compute

Defects regression-locked here:

  G1  Fail-open gates. A rejected human answer once reached compute and
      produced a combined rating, because a FAILED status did not stop the
      graph. The compute gate reads committed case state and blocks every
      status except the two from which computing is correct.
  G2  Unstable gate conditions. Strands re-evaluates edge conditions when it
      persists a session, to work out where a resumed run continues. A
      condition that stops being true once its target has run ("status is
      classified", false as soon as assess asks) emptied the resume
      frontier, and the resumed run silently did nothing. The persisted
      session must name 'assess' as the node to continue at.
  G3  A hand-edited session naming 'compute' as the node to continue at ran
      the arithmetic with the question still open and reported NO
      DISCREPANCY FOUND, exit 0: the edge was bypassed, not evaluated.
      compute now checks committed state for itself.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from _support import (
    EXIT_CANNOT_PROCEED,
    actions,
    batch,
    item,
    main,
    nodes_run,
    run_audit,
    scripted,
    tabular_letter,
)
from recheck.case import STATUSES, Case, CaseStore
from recheck.graph import (
    COMPUTABLE_STATES,
    _extraction_succeeded,
    _safe_to_compute,
    build_graph,
)
from strands.multiagent.base import MultiAgentResult, Status

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]


def _gate(tmp_path, status: str) -> bool:
    store = CaseStore(tmp_path / "runs")
    # A "complete" case must carry the figures the engine gives for its facts
    # (here, none: 0%), or the store refuses it as corrupt.
    result = dict(recomputed_combined=0, recomputed_degree=0) if status == "complete" else {}
    store.save(Case(case_id="g", source_path="x.txt", status=status, **result))
    return _safe_to_compute(store, "g")(None)


# --------------------------------------------------------------------------
# G1: the compute gate
# --------------------------------------------------------------------------

def test_the_computable_states_are_exactly_ready_and_complete():
    assert COMPUTABLE_STATES == ("ready", "complete")
    assert set(COMPUTABLE_STATES) <= set(STATUSES)


@pytest.mark.parametrize("status", [s for s in STATUSES if s not in ("ready", "complete")])
def test_every_other_status_blocks_the_arithmetic(tmp_path, status):
    assert _gate(tmp_path, status) is False


@pytest.mark.parametrize("status", ["ready", "complete"])
def test_ready_and_complete_allow_the_arithmetic(tmp_path, status):
    assert _gate(tmp_path, status) is True


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", "null", json.dumps({"case_id": "g", "status": "ready"}),
     json.dumps({**Case("g", "x.txt", status="ready").__dict__, "schema_version": 1}),
     json.dumps({**Case("g", "x.txt").__dict__, "status": "resumed"})],
)
def test_an_unreadable_case_blocks_the_arithmetic(tmp_path, content):
    store = CaseStore(tmp_path / "runs")
    path = store.path_for("g")
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    assert _safe_to_compute(store, "g")(None) is False


def test_a_missing_case_blocks_the_arithmetic(tmp_path):
    assert _safe_to_compute(CaseStore(tmp_path / "runs"), "never-written")(None) is False


# --------------------------------------------------------------------------
# The extraction gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "results,expected",
    [
        ({"extract": SimpleNamespace(status=Status.COMPLETED)}, True),
        ({"extract": SimpleNamespace(status=Status.FAILED)}, False),
        ({"extract": SimpleNamespace(status=Status.INTERRUPTED)}, False),
        ({}, False),
    ],
)
def test_the_extraction_gate_reads_the_extract_nodes_own_result(results, expected):
    assert _extraction_succeeded(SimpleNamespace(results=results)) is expected


def test_a_failed_extraction_runs_nothing_downstream(tmp_path):
    letter = tmp_path / "unreadable.txt"
    letter.write_text("Dear veteran,\n\nThank you for your enquiry.\n", encoding="utf-8")
    factory = scripted(batch(item("anything", "upper")))
    store, _ = run_audit(tmp_path / "runs", "bad", letter, factory)
    case = store.load("bad")
    assert case.status == "unparsed"
    assert nodes_run(store, "bad") == ["extract"]
    assert factory.models == [], "classification must not run on a failed extraction"
    for action in ("Assessment", "Question for a reviewer", "Final degree of disability"):
        assert action not in actions(store, "bad")
    assert "Extraction FAILED" in actions(store, "bad")


def test_a_failed_extraction_exits_3_and_claims_nothing(tmp_path):
    letter = tmp_path / "unreadable.txt"
    letter.write_text("RATING DECISION\n  1. Tinnitus (DC 6260) ..... 10%\n", encoding="utf-8")
    code, out, err = main("audit", letter, "--case", "bad", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert "COULD NOT READ THE LETTER" in out
    assert "NO DISCREPANCY FOUND" not in out


# --------------------------------------------------------------------------
# G2: stable conditions and the resume frontier
# --------------------------------------------------------------------------

def _session_state(store: CaseStore, case_id: str) -> dict:
    (path,) = store.session_dir(case_id).rglob("multi_agent.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_interrupted_session_continues_at_assess(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    state = _session_state(store, "pair")
    assert state["status"] == "interrupted"
    assert "assess" in state["next_nodes_to_execute"]
    assert "compute" not in state["next_nodes_to_execute"]
    assert state["interrupted_nodes"] == ["assess"]


def test_the_extraction_gate_is_still_true_on_the_restored_graph(tmp_path):
    """Re-evaluated after the interrupt, the extraction gate must not flip."""
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    restored = build_graph(store, "pair", str(letter), None)
    assert _extraction_succeeded(restored.state) is True


def test_no_session_is_written_outside_the_strands_session_directory(tmp_path):
    """One owner for graph state: no hand-written graph_state.json beside it."""
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    case_dir = store.dir_for("pair")
    assert sorted(p.name for p in case_dir.iterdir()) == ["case.json", "session"]
    assert not list(case_dir.rglob("graph_state.json"))


# --------------------------------------------------------------------------
# G3: compute defends itself
# --------------------------------------------------------------------------

def test_a_session_edited_to_continue_at_compute_does_not_compute(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store_root = tmp_path / "runs"
    code, _, _ = main("audit", letter, "--case", "pair", store=store_root)
    assert code == 2
    store = CaseStore(store_root)
    (path,) = store.session_dir("pair").rglob("multi_agent.json")
    state = json.loads(path.read_text(encoding="utf-8"))
    state["next_nodes_to_execute"] = ["compute"]
    state["interrupted_nodes"] = ["compute"]
    state["completed_nodes"] = ["extract", "classify"]
    context = state["_internal_state"]["interrupt_state"]["context"]
    context["compute"] = context.pop("assess")
    path.write_text(json.dumps(state), encoding="utf-8")

    code, out, err = main("resume", "--case", "pair", "--answer", "2=left", store=store_root)
    case = store.load("pair")
    assert code == EXIT_CANNOT_PROCEED
    assert case.recomputed_degree is None
    assert case.status == "awaiting_human"
    assert "NO DISCREPANCY FOUND" not in out and "POTENTIAL DISCREPANCY" not in out
    # resume now refuses a session whose question is not waiting at assess
    # before running any node (graph.outstanding_interrupt). compute's own
    # check is still exercised: Strands is driven through the edited frontier
    # directly, as a caller that skips that check would.
    assert "compute" not in nodes_run(store, "pair")
    import asyncio

    restored = build_graph(store, "pair", str(letter), None)
    (pending,) = restored._interrupt_state.interrupts.values()
    asyncio.run(restored.invoke_async([{"interruptResponse": {"interruptId": pending.id, "response": {"2": "left"}}}]))
    case = store.load("pair")
    assert case.recomputed_degree is None
    assert case.status == "awaiting_human"
    # Unconditional: if Strands ever stopped honouring the edited frontier this
    # test must fail loudly rather than pass without exercising the self-check.
    assert nodes_run(store, "pair")[-1] == "compute"
    assert "Arithmetic REFUSED" in actions(store, "pair")


@pytest.mark.parametrize("status", ["open", "extracted", "classified", "awaiting_human", "undetermined", "unparsed"])
def test_the_compute_node_refuses_a_case_that_is_not_ready(tmp_path, status):
    import asyncio

    from recheck.graph import ComputeNode

    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    case = store.load("pair")
    case.status = status
    store.save(case)
    result: MultiAgentResult = asyncio.run(ComputeNode(store, "pair").invoke_async("compute"))
    after = store.load("pair")
    assert result.status == Status.FAILED
    assert after.recomputed_degree is None and after.status == status
    assert "Arithmetic REFUSED" in [e["action"] for e in after.trace]
