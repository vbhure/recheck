"""The conditional edge guarding the arithmetic.

Regression cover for a real defect: returning Status.FAILED from the assess
node did NOT stop downstream nodes, so a REJECTED human answer still reached
`compute` and produced a combined rating. The gate is now a conditional edge
on the graph and is enforced by reading committed state from disk.

The integration-level cover is in test_cross_process_resume.py, which asserts
that a rejected answer leaves recomputed_degree as None. These tests pin the
gate's own contract so the behaviour cannot be weakened without failing here.
"""

from __future__ import annotations

import pytest

from recheck.case import Case, CaseStore
from recheck.graph import BLOCKED_STATES, _safe_to_compute, choose_bilateral_pair
from recheck.classify import Decision
from recheck.provenance import Actor


def _gate(tmp_path, status: str):
    store = CaseStore(tmp_path / "runs")
    store.save(Case(case_id="g", source_path="x.txt", status=status))
    return _safe_to_compute(store, "g")(None)


@pytest.mark.parametrize("status", BLOCKED_STATES)
def test_blocked_states_prevent_arithmetic(tmp_path, status):
    assert _gate(tmp_path, status) is False


@pytest.mark.parametrize("status", ["open", "resumed", "complete"])
def test_computable_states_allow_arithmetic(tmp_path, status):
    assert _gate(tmp_path, status) is True


def test_gate_is_fail_closed_when_the_case_cannot_be_read(tmp_path):
    """An unreadable case must block, not default to computing."""
    store = CaseStore(tmp_path / "runs")
    assert _safe_to_compute(store, "never-written")(None) is False


def test_awaiting_human_is_a_blocked_state():
    """Explicit: the arithmetic may not run while a question is outstanding."""
    assert "awaiting_human" in BLOCKED_STATES
    assert "invalid_answer" in BLOCKED_STATES


# --------------------------------------------------------------------------
# Deterministic pair selection
# --------------------------------------------------------------------------

def _d(percent, group, side, *, needs_human=False, by=Actor.DETERMINISTIC):
    return Decision("c", percent, group, side, by, None, needs_human, None, None)


def test_pair_requires_opposite_sides():
    assert choose_bilateral_pair([_d(20, "lower", "left"), _d(10, "lower", "left")]) is None


def test_pair_requires_the_same_extremity_group():
    assert choose_bilateral_pair([_d(20, "upper", "left"), _d(10, "lower", "right")]) is None


def test_pair_requires_both_to_be_compensable():
    """4.26(c): a 0% rating is not a compensable disability."""
    assert choose_bilateral_pair([_d(20, "lower", "left"), _d(0, "lower", "right")]) is None


def test_pair_excludes_anything_still_awaiting_a_human():
    unresolved = [_d(20, "upper", "left", needs_human=True), _d(10, "upper", "right")]
    assert choose_bilateral_pair(unresolved) is None


def test_pair_is_found_when_every_condition_is_met():
    assert choose_bilateral_pair([_d(20, "upper", "left"), _d(10, "upper", "right")]) == (0, 1)


def test_non_extremity_conditions_are_never_paired():
    """Guards the contradiction case: group "none" must not pair even if a
    side somehow reached the decision."""
    assert choose_bilateral_pair([_d(60, "none", "left"), _d(10, "none", "right")]) is None
