"""Mutation-testing regressions: safety checks no existing test would miss.

Each test here was written against a mutation of 12705f8 that removed or
weakened one safety check and left the whole suite passing (see the review
lane "tests"). Every test fails with its mutation applied and passes
without it; the mutation is named in each docstring.
"""

from __future__ import annotations

import json

import pytest

from _support import UNLISTED, batch, item, run_audit, scripted, tabular_letter
from recheck import classify as classify_module
from recheck.case import CaseCorrupt
from recheck.classify import classify
from recheck.extract.deterministic import ExtractedRating
from recheck.provenance import Actor, Trace

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]


def _open_case(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    assert store.load("pair").status == "awaiting_human"
    return store


def _edit(tmp_path, change) -> None:
    path = tmp_path / "runs" / "pair" / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


# --------------------------------------------------------------------------
# case.json values are checked on a case with no result to re-derive
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [("extremity_group", "torso"), ("laterality", "north")])
def test_an_impossible_fact_on_an_open_case_is_refused(tmp_path, field, value):
    """Mutations load_group_value / load_side_value. The existing tamper test
    edits a COMPLETE case, where the re-derived figure no longer matches and
    the load refuses it anyway; the value check itself was never exercised.
    On an open case nothing is re-derived: a group of "torso" on the knee of
    unstated side made it neither asked nor paired, so the question lost it."""
    store = _open_case(tmp_path)
    _edit(tmp_path, lambda raw: raw["decisions"][2].update({field: value}))
    with pytest.raises(CaseCorrupt, match="impossible"):
        store.load("pair")


# --------------------------------------------------------------------------
# An answer that exists only after the budget is not used, whatever path it took
# --------------------------------------------------------------------------

class _Clock:
    """time.monotonic for recheck.classify: the deadline is read at 0, every later reading is past it."""

    def __init__(self) -> None:
        self.readings = 0

    def monotonic(self) -> float:
        self.readings += 1
        return 0.0 if self.readings == 1 else 1e9


def test_an_answer_returned_after_the_deadline_is_not_used_even_if_the_budget_wrapper_returns_it(monkeypatch):
    """Mutation late_answer. _ask_model checks the clock after _within_budget
    returns ("belt and braces ... an answer that exists only after the budget
    ran out is not used"). No test reached that check: every timeout test
    ends inside _within_budget. Here the wrapper hands back a valid, confident
    answer after the deadline has passed."""
    monkeypatch.setattr(classify_module, "time", _Clock())
    factory = scripted(batch(item(UNLISTED, "upper", 0.95)), timeout_s=5)
    trace = Trace()
    (decision,) = classify([ExtractedRating(UNLISTED, 20, "unrecognised", f"1. {UNLISTED}", 1)], trace, factory)
    assert decision.extremity_group == "unknown" and decision.group_by is None
    assert "did not answer within 5s" in (decision.note or "")
    assert not trace.by_actor(Actor.AI)
