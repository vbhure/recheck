"""The three golden demo scenarios, plus the 4.26(d) exception.

These are regression locks on the demo. If any of these change, the video is
wrong and must be re-shot.
"""
from recheck.cfr.combine import combine, final_degree
from recheck.cfr.rating import evaluate

# A realistic and very common VA picture:
#   PTSD                  60%
#   right knee            20%   (paired extremity)
#   left knee             10%   (paired extremity)
#   tinnitus              10%   (statutory maximum is 10)
HERO_RATINGS = [60, 20, 10, 10]
HERO_PAIR = [20, 10]


def test_golden_a_discrepancy_bilateral_raises_the_band():
    """GOLDEN A: stated 70%, recomputed 80%. A full band, for life."""
    ev = evaluate(HERO_RATINGS, bilateral_pair=HERO_PAIR)
    assert ev.bilateral_subtotal_value == 31       # 28 combined, +10% = 30.8 -> 31
    assert ev.combined_value == 75                 # 60 -> 72 -> 75
    assert ev.final_degree == 80                   # 75 ends in 5, adjusts upward
    assert ev.bilateral_applied is True
    assert ev.alternative_final_degree == 70       # what you get ignoring 4.26


def test_golden_a_the_same_numbers_without_the_factor_give_seventy():
    """The VA's answer is not absurd - it is what you get without 4.26."""
    assert combine(HERO_RATINGS) == 74
    assert final_degree(74) == 70


def test_the_pairing_choice_is_what_decides_the_outcome():
    """Same four ratings. Which two are 'paired extremities' moves the answer
    by a full band. This is the entire justification for asking a human.

    Pairing the two 10s is the regulation's own worked example -> 70%.
    Pairing the 20 with a 10 -> 80%.
    """
    as_two_tens = evaluate(HERO_RATINGS, bilateral_pair=[10, 10])
    as_twenty_and_ten = evaluate(HERO_RATINGS, bilateral_pair=[20, 10])
    assert as_two_tens.final_degree == 70
    assert as_twenty_and_ten.final_degree == 80
    assert as_two_tens.final_degree != as_twenty_and_ten.final_degree


def test_golden_c_no_discrepancy_case():
    """GOLDEN C: recomputation agrees with the letter.

    Proves Recheck is not built to manufacture errors - which is the first
    thing a skeptical reviewer will suspect.
    """
    ev = evaluate([50, 30])
    assert ev.combined_value == 65
    assert ev.final_degree == 70
    assert ev.bilateral_applied is False


def test_426d_exception_keeps_the_more_favorable_result():
    """4.26(d): applying the factor would LOWER the result, so it is excluded.

    ratings 80, 60, 20, 10 with a 10/20 bilateral pair:
      with the factor    -> subtotal 31, combined 94 -> 90%
      without the factor -> combined 95 -> 100%
    The regulation requires the result most favorable to the veteran.
    """
    ev = evaluate([80, 60, 20, 10], bilateral_pair=[10, 20])
    assert ev.bilateral_applied is False
    assert ev.final_degree == 100
    assert ev.alternative_final_degree == 90
    assert any("4.26(d)" in n for n in ev.notes)


def test_426c_uncompensable_pair_does_not_get_the_factor():
    """4.26(c): needs a compensable disability in EACH paired extremity."""
    ev = evaluate([50, 30, 0], bilateral_pair=[30, 0])
    assert ev.bilateral_applied is False
    assert any("4.26(c)" in n for n in ev.notes)


def test_bilateral_pair_must_be_a_subset_of_the_ratings():
    import pytest

    with pytest.raises(ValueError):
        evaluate([50, 30], bilateral_pair=[20, 10])


def test_every_evaluation_carries_a_derivation():
    """No number in a report may be unexplained (provenance requirement)."""
    ev = evaluate(HERO_RATINGS, bilateral_pair=HERO_PAIR)
    assert len(ev.steps) >= 3
    assert all(s.rule.startswith("38 CFR") for s in ev.steps)
    assert ev.steps[-1].running_after == ev.combined_value
