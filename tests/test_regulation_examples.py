"""The worked examples written into the regulations themselves.

These are the strongest tests in the suite: the government states both the
inputs and the expected intermediate values, so they pin not just the answer
but the ALGORITHM - including where rounding happens.
"""
from recheck.cfr.combine import bilateral_subtotal, combine, combine_step, final_degree
from recheck.cfr.rating import evaluate


def test_425_prose_example_60_and_30():
    """4.25: "a person having a 60 percent disability is considered 40 percent
    efficient... a further 30 percent disability... 28 percent efficiency
    altogether. The individual is thus 72 percent disabled."
    """
    assert combine([60, 30]) == 72


def test_425_prose_example_50_and_30_converts_to_70():
    """4.25(a): "the combined value will be found to be 65 percent, but the
    65 percent must be converted to 70 percent"."""
    assert combine([50, 30]) == 65
    assert final_degree(65) == 70


def test_425_prose_example_40_and_20_converts_to_50():
    """4.25(a): "the combined value is found to be 52 percent, but the 52
    percent must be converted to the nearest degree divisible by 10, which is
    50 percent"."""
    assert combine([40, 20]) == 52
    assert final_degree(52) == 50


def test_426_worked_example_full_chain():
    """4.26: "with disabilities evaluated at 60 percent, 20 percent, 10 percent
    and 10 percent (with the two 10 percent evaluations being bilateral
    disabilities), the order of severity would be 60, 21 and 20. The 60 and 21
    combine to 68 percent and the 68 and 20 combine to 74 percent, converted to
    70 percent as the final degree of disability."

    This single example pins every rounding decision in the engine.
    """
    # the two 10s combine to 19, plus 10% of 19 (1.9) = 20.9, stated as 21
    assert bilateral_subtotal([10, 10]) == 21
    # 60 and 21 combine to 68
    assert combine_step(60, 21) == 68
    # 68 and 20 combine to 74
    assert combine_step(68, 20) == 74
    # converted to 70
    assert final_degree(74) == 70

    ev = evaluate([60, 20, 10, 10], bilateral_pair=[10, 10])
    assert ev.bilateral_subtotal_value == 21
    assert ev.combined_value == 74
    assert ev.final_degree == 70
    assert ev.bilateral_applied is True


def test_426_factor_is_added_not_combined():
    """If the 10% were wrongly *combined* instead of added, 10+10 would give
    19 -> combine(19,10) = 27, not 21. Pin the difference explicitly."""
    assert bilateral_subtotal([10, 10]) == 21
    assert combine_step(19, 10) == 27
