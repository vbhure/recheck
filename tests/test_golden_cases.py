"""The golden scenarios: the engine's demo numbers, and the stable letters end to end.

If any of these change, the demo is wrong and must be re-shot. Letters 01-06
are stable fixtures; each is run through the whole graph with no classifier,
so every fact below comes from the letter or from arithmetic.
"""

from __future__ import annotations

import pytest

from _support import LETTERS, answer, run_audit
from recheck.cfr.combine import combine, final_degree
from recheck.cfr.rating import evaluate
from recheck.provenance import Actor

# A realistic and very common VA picture:
#   PTSD 60%, right knee 20%, left knee 10%, tinnitus 10%
HERO_RATINGS = [60, 20, 10, 10]
HERO_PAIR = [20, 10]


def test_golden_a_discrepancy_bilateral_raises_the_band():
    """GOLDEN A: stated 70%, recomputed 80%."""
    ev = evaluate(HERO_RATINGS, bilateral_pair=HERO_PAIR)
    assert ev.bilateral_subtotal_value == 31       # 28 combined, +10% = 30.8 -> 31
    assert ev.combined_value == 75                 # 60 -> 72 -> 75
    assert ev.final_degree == 80                   # 75 ends in 5, adjusts upward
    assert ev.bilateral_applied is True
    assert ev.alternative_final_degree == 70       # what you get ignoring 4.26


def test_golden_a_the_same_numbers_without_the_factor_give_seventy():
    """The letter's figure is not absurd - it is what you get without 4.26."""
    assert combine(HERO_RATINGS) == 74
    assert final_degree(74) == 70


def test_the_pairing_decides_the_outcome():
    """Same four ratings; which two are paired extremities moves the answer a
    full band. This is why a side the letter does not state can be worth a
    question - and why materiality checks before asking."""
    assert evaluate(HERO_RATINGS, bilateral_pair=[10, 10]).final_degree == 70
    assert evaluate(HERO_RATINGS, bilateral_pair=[20, 10]).final_degree == 80


def test_golden_c_no_discrepancy_case():
    """GOLDEN C: Recheck is not built to manufacture errors."""
    ev = evaluate([50, 30])
    assert (ev.combined_value, ev.final_degree, ev.bilateral_applied) == (65, 70, False)


def test_426d_exception_keeps_the_more_favorable_result():
    """80, 60, 20, 10 with a 10/20 pair: with the factor 90%, without 100%."""
    ev = evaluate([80, 60, 20, 10], bilateral_pair=[10, 20])
    assert ev.bilateral_applied is False
    assert ev.final_degree == 100
    assert ev.alternative_final_degree == 90
    assert any("4.26(d)" in n for n in ev.notes)


def test_426c_uncompensable_pair_does_not_get_the_factor():
    ev = evaluate([50, 30, 0], bilateral_pair=[30, 0])
    assert ev.bilateral_applied is False
    assert any("4.26(c)" in n for n in ev.notes)


def test_bilateral_pair_must_be_a_subset_of_the_ratings():
    with pytest.raises(ValueError):
        evaluate([50, 30], bilateral_pair=[20, 10])


def test_every_evaluation_carries_a_derivation():
    ev = evaluate(HERO_RATINGS, bilateral_pair=HERO_PAIR)
    assert len(ev.steps) >= 3
    assert all(s.rule.startswith("38 CFR") for s in ev.steps)
    assert ev.steps[-1].running_after == ev.combined_value


# --------------------------------------------------------------------------
# The stable letters, end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "letter,groups",
    [
        ("01_tabular.txt", "lower"),        # tabular layout
        ("02_prose.txt", "lower"),          # hard-wrapped prose
        ("03_history_trap.txt", "lower"),   # "currently evaluated as" and criteria percentages
        ("04_cross_bodypart.txt", "upper"),  # wrist and forearm are one extremity (4.26(a))
    ],
)
def test_the_hero_letters_recompute_to_80_from_facts_the_letter_states(tmp_path, letter, groups):
    store, result = run_audit(tmp_path / "runs", "g", LETTERS / letter)
    case = store.load("g")
    assert result.interrupts == []
    assert case.status == "complete"
    assert (case.stated_combined, case.recomputed_degree, case.bilateral_applied) == (70, 80, True)
    facts = [(d.percent, d.extremity_group, d.laterality) for d in case.load_decisions()]
    assert facts == [(60, "none", "unknown"), (20, groups, "right" if groups == "lower" else "left"),
                     (10, groups, "left" if groups == "lower" else "right"), (10, "none", "unknown")]
    decisions = case.load_decisions()
    assert all(d.group_by is Actor.DETERMINISTIC for d in decisions)
    assert all(d.side_by is Actor.DETERMINISTIC for d in decisions if d.extremity_group != "none")
    assert [s["node"] for s in case.timeline] == ["extract", "classify", "assess", "compute"]


def test_letter_05_asks_which_side_each_knee_is_on_and_finishes_with_the_answer(tmp_path):
    store, result = run_audit(tmp_path / "runs", "g", LETTERS / "05_missing_side.txt")
    case = store.load("g")
    assert case.status == "awaiting_human"
    assert case.possible_degrees == [70, 80]
    assert [c["index"] for c in result.interrupts[0].reason["conditions"]] == [1, 2]

    answer(store, "g", "1=left,2=right")
    case = store.load("g")
    assert (case.status, case.recomputed_degree) == ("complete", 80)
    assert [d.side_by for d in case.load_decisions()[1:3]] == [Actor.HUMAN, Actor.HUMAN]


def test_letter_05_answered_with_the_same_side_twice_agrees_with_the_letter(tmp_path):
    store, _ = run_audit(tmp_path / "runs", "g", LETTERS / "05_missing_side.txt")
    answer(store, "g", "1=left,2=left")
    case = store.load("g")
    assert (case.status, case.recomputed_degree, case.bilateral_applied) == ("complete", 70, False)


def test_letter_06_the_no_discrepancy_control(tmp_path):
    store, result = run_audit(tmp_path / "runs", "g", LETTERS / "06_agrees.txt")
    case = store.load("g")
    assert result.interrupts == []
    assert (case.status, case.stated_combined, case.recomputed_degree) == ("complete", 70, 70)
