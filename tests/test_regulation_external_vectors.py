"""Worked calculations published by VA and the Board, used as external oracles.

The regulations' own examples (tests/test_regulation_examples.py) pin the
two-rating case. These pin the cases the single-pair engine got wrong - more
than two bilateral disabilities, four extremities, and 4.26(d) - using
numbers someone other than this project wrote down. Each docstring quotes the
source; the expected values are the source's, not recomputed here.

Board of Veterans' Appeals decisions are not precedential. They are used for
their arithmetic, which is checked against the regulation text in each case.
"""

from recheck.cfr.combine import combine, final_degree
from recheck.cfr.rating import Paired, evaluate

L, R = "left", "right"


def test_bva_1312955_five_lower_extremity_ratings_in_one_bilateral_group():
    """BVA Citation Nr 1312955 (April 18, 2013), https://www.va.gov/vetapp13/Files2/1312955.txt

    Five lower-extremity disabilities - two separate left knee ratings, both
    hips, and a left lower extremity nerve rating - all go into ONE bilateral
    calculation: "10 percent of the combined evaluation of 41 (or 4.1) is
    added on to the combined evaluation (41 + 4.1 = 45.1, or 45 rounded
    down)". With the remaining 40, 30, 20 and 10 ratings the combined value is
    84, "rounded to 80".
    """
    paired = [
        Paired(10, "lower", L),  # left knee instability
        Paired(10, "lower", L),  # left knee scar / arthritis
        Paired(10, "lower", R),  # right hip
        Paired(10, "lower", L),  # left hip
        Paired(10, "lower", L),  # left lower extremity sensory dysfunction
    ]
    assert combine([10, 10, 10, 10, 10]) == 41
    ev = evaluate([40, 30, 20, 10] + [10] * 5, paired=paired)
    assert ev.bilateral_subtotal_value == 45
    assert ev.combined_value == 84
    assert ev.final_degree == 80


def test_bva_0815809_four_extremities():
    """BVA Citation Nr 0815809 (May 14, 2008), https://www.va.gov/vetapp08/files2/0815809.txt

    "the veteran is entitled to the bilateral factor since he is
    service-connected for disabilities involving both upper extremities and
    both lower extremities" - 20, 20, 10 and 10 combine to 48, plus 4.8
    (38 CFR 4.26(b): all four extremities, one factor).
    """
    paired = [Paired(20, "upper", L), Paired(20, "upper", R), Paired(10, "lower", L), Paired(10, "lower", R)]
    assert combine([20, 20, 10, 10]) == 48
    ev = evaluate([20, 20, 10, 10], paired=paired)
    assert ev.bilateral_subtotal_value == 53  # 48 + 4.8 = 52.8
    assert len(ev.bilateral_members) == 4


def test_federal_register_4_26_d_example_93_and_two_bilateral_tens():
    """88 FR 22915 (April 14, 2023), https://www.govinfo.gov/content/pkg/FR-2023-04-14/html/2023-07426.htm

    VA's own illustration of why 4.26(d) was added: under the prior rule, "93
    percent and 21 percent combine to 94.47, which is rounded to 94 and then
    adjusted downward to a final combined rating of 90 percent", whereas
    combining the two bilateral 10s separately reaches 100. Other ratings of
    90 and 30 combine to exactly 93.
    """
    assert combine([90, 30]) == 93
    ratings = [90, 30, 10, 10]
    prior_rule = combine([90, 30, 21])
    assert prior_rule == 94 and final_degree(prior_rule) == 90
    ev = evaluate(ratings, bilateral_pair=[10, 10])
    assert ev.final_degree == 100
    assert ev.bilateral_applied is False
    assert ev.alternative_final_degree == 90


def test_bva_1519449_single_bilateral_rating_with_both_knees():
    """BVA Citation Nr 1519449 (May 6, 2015), https://www.va.gov/vetapp15/Files3/1519449.txt

    "the 10 percent disability rating for bilateral feet is combined with the
    10 percent rating for right knee, resulting in a 19 percent rating. That
    19 percent combined with an additional 10 percent for the left knee
    results in a 27 percent rating", then 2.7 is added: 29.7, i.e. 30.
    A single evaluation covering both feet enters the calculation alongside
    separate right and left leg ratings (M21-1 V.iv.1.C.4.b).
    """
    paired = [Paired(10, "lower", "both"), Paired(10, "lower", R), Paired(10, "lower", L)]
    ev = evaluate([10, 10, 10], paired=paired)
    assert ev.bilateral_subtotal_value == 30
