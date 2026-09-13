"""38 CFR 4.26 applied to every bilateral disability, not one pair of them.

Regression cover for a real defect. The engine used to select the FIRST
left/right pair in an extremity group and ignore any further arm or leg
disabilities. 4.26 says "the ratings for the disabilities of the right and
left sides will be combined" - all of them. Across 750 three-leg-disability
combinations the single-pair engine produced a different final degree in
122, e.g. left knee 10%, left ankle 20%, right knee 30%: 50% instead of 60%.
A letter that stated the correct 60% would have been reported as a potential
discrepancy.

The oracle at the bottom is written independently of recheck.cfr.rating:
it enumerates subsets directly from the regulation's wording and uses only
Table I arithmetic, so the engine is checked against a second reading of the
rule rather than against itself.
"""

from __future__ import annotations

import itertools
from decimal import ROUND_HALF_UP, Decimal

import pytest

from recheck.cfr.combine import bilateral_subtotal, combine, final_degree
from recheck.cfr.rating import MAX_BILATERAL_MEMBERS, Paired, bilateral_group, evaluate

L, R = "left", "right"


def P(percent, extremity, side):
    return Paired(percent, extremity, side)


# --------------------------------------------------------------------------
# Which disabilities enter the factor
# --------------------------------------------------------------------------

def test_every_disability_of_a_qualifying_pair_enters_the_factor():
    """The defect: left knee 10, left ankle 20, right knee 30."""
    paired = [P(10, "lower", L), P(20, "lower", L), P(30, "lower", R)]
    ev = evaluate([10, 20, 30], paired=paired)
    # 30, 20, 10 combine to 50 (30 -> 44 -> 49.6 = 50); + 5.0 = 55
    assert ev.bilateral_subtotal_value == 55
    assert sorted(m.percent for m in ev.bilateral_members) == [10, 20, 30]
    assert ev.final_degree == 60
    assert ev.alternative_final_degree == 50


def test_the_old_single_pair_answer_is_no_longer_produced():
    paired = [P(10, "lower", L), P(20, "lower", L), P(30, "lower", R)]
    single_pair = evaluate([10, 20, 30], bilateral_pair=[10, 30]).final_degree
    assert single_pair == 50
    assert evaluate([10, 20, 30], paired=paired).final_degree == 60


def test_426a_upper_and_lower_extremities_as_a_whole():
    """4.26(a): "a compensable disability of the right thigh ... and one of the
    left foot ... the bilateral factor applies"."""
    group = bilateral_group([P(40, "lower", R), P(10, "lower", L)])
    assert len(group) == 2


def test_426b_four_extremities_form_one_group():
    """4.26(b): "combine the ratings of the disabilities affecting the 4
    extremities in the order of their individual severity and apply the
    bilateral factor by adding, not combining, 10 percent"."""
    paired = [P(10, "upper", L), P(10, "upper", R), P(10, "lower", L), P(10, "lower", R)]
    ev = evaluate([10, 10, 10, 10], paired=paired)
    # 10,10,10,10 combine to 34 (19, 27, 34); + 3.4 = 37 - ONE subtotal
    assert ev.bilateral_subtotal_value == 37
    assert len(ev.bilateral_members) == 4
    assert ev.steps[0].rule == "38 CFR 4.26(b)"


def test_an_unpaired_extremity_stays_out_of_the_factor():
    """Both legs affected, one arm: the arm is not bilateral."""
    group = bilateral_group([P(20, "lower", L), P(10, "lower", R), P(30, "upper", R)])
    assert {m.extremity for m in group} == {"lower"}


def test_426c_a_noncompensable_side_does_not_qualify():
    assert bilateral_group([P(30, "lower", L), P(0, "lower", R)]) == []
    ev = evaluate([50, 30, 0], paired=[P(30, "lower", L), P(0, "lower", R)])
    assert ev.bilateral_applied is False
    assert any("4.26(c)" in n for n in ev.notes)


def test_426c_a_noncompensable_disability_does_not_join_a_qualifying_group():
    group = bilateral_group([P(20, "lower", L), P(10, "lower", R), P(0, "lower", L)])
    assert sorted(m.percent for m in group) == [10, 20]


def test_same_side_disabilities_alone_never_qualify():
    assert bilateral_group([P(20, "lower", L), P(10, "lower", L), P(40, "upper", R)]) == []


# --------------------------------------------------------------------------
# 4.26(d): "one or more" may be removed
# --------------------------------------------------------------------------

def test_426d_partial_exclusion_when_it_is_strictly_more_favourable():
    """60 plus leg disabilities left 10, left 20, right 70.

    All three in the factor -> 90%. Leaving the left 10 out and combining it
    separately -> 100%. The regulation says "one or more", so the search must
    consider removing a single member, not only all-or-nothing.
    """
    paired = [P(10, "lower", L), P(20, "lower", L), P(70, "lower", R)]
    ev = evaluate([60, 10, 20, 70], paired=paired)
    assert ev.final_degree == 100
    assert ev.bilateral_applied is True
    assert [m.percent for m in ev.excluded_under_426d] == [10]
    assert ev.alternative_final_degree == 90
    assert any("4.26(d)" in n for n in ev.notes)


def test_426d_whole_factor_dropped_still_supported():
    ev = evaluate([80, 60, 20, 10], bilateral_pair=[10, 20])
    assert ev.bilateral_applied is False
    assert ev.final_degree == 100
    assert ev.alternative_final_degree == 90


def test_ties_keep_the_default_procedure():
    """Only a STRICTLY better result departs from putting every member in."""
    paired = [P(10, "lower", L), P(10, "lower", R)]
    ev = evaluate([10, 10], paired=paired)
    assert ev.bilateral_applied is True
    assert ev.excluded_under_426d == ()


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------

def test_subtotal_over_one_hundred_is_capped_not_a_crash():
    """80 and 60 combine to 92; adding 9.2 gives 101. Combining that with a
    further rating used to raise 'rating out of range 0-100: 101'."""
    assert bilateral_subtotal([80, 60]) == 100
    assert evaluate([80, 60, 10], bilateral_pair=[80, 60]).final_degree == 100


def test_paired_disabilities_must_be_among_the_ratings():
    with pytest.raises(ValueError):
        evaluate([50, 30], paired=[P(20, "lower", L), P(30, "lower", R)])


def test_cannot_pass_both_forms():
    with pytest.raises(ValueError):
        evaluate([10, 10], bilateral_pair=[10, 10], paired=[P(10, "lower", L), P(10, "lower", R)])


def test_refuses_rather_than_truncating_an_implausible_search():
    n = MAX_BILATERAL_MEMBERS + 2
    paired = [P(10, "lower", L if i % 2 else R) for i in range(n)]
    with pytest.raises(ValueError):
        evaluate([10] * n, paired=paired)


# --------------------------------------------------------------------------
# Independent oracle
# --------------------------------------------------------------------------

def _oracle(others, paired):
    """A second, deliberately naive reading of 4.26. Shares only Table I.

    Enumerate every subset of the compensable arm/leg disabilities that uses
    both sides of each extremity pair it touches (the empty subset = no
    factor). Default procedure = the largest qualifying set. Answer = the
    default, unless some subset is strictly more favourable (4.26(d)).
    """
    def degree_for(kept_idx):
        kept = [paired[i] for i in kept_idx]
        rest = [p[0] for i, p in enumerate(paired) if i not in kept_idx]
        if not kept:
            return final_degree(combine(others + rest))
        base = Decimal(combine([p[0] for p in kept]))
        sub = min(100, int((base * Decimal("1.1")).quantize(Decimal(1), rounding=ROUND_HALF_UP)))
        return final_degree(combine(others + rest + [sub]))

    def valid(kept_idx):
        kept = [paired[i] for i in kept_idx]
        if any(p[0] < 10 for p in kept):
            return False
        for ext in {p[1] for p in kept}:
            if {p[2] for p in kept if p[1] == ext} != {"left", "right"}:
                return False
        return True

    indices = range(len(paired))
    candidates = [s for n in range(len(paired) + 1) for s in itertools.combinations(indices, n)
                  if not s or valid(s)]
    # The default procedure is never worse than itself, so "the default unless
    # something is strictly better" is simply the best candidate.
    return max(degree_for(s) for s in candidates)


def _cases():
    shapes = [
        [("lower", L), ("lower", R)],
        [("lower", L), ("lower", L), ("lower", R)],
        [("upper", L), ("upper", R), ("lower", R)],
        [("upper", L), ("upper", R), ("lower", L), ("lower", R)],
        [("lower", L), ("lower", R), ("lower", R), ("upper", L)],
    ]
    values = [0, 10, 20, 40, 60, 80]
    others_options = [[], [30], [70, 50], [90]]
    out = []
    for shape_no, shape in enumerate(shapes):
        combos = itertools.product(values, repeat=len(shape))
        for combo_no, combo in enumerate(combos):
            if combo_no % (7 if len(shape) > 3 else 1):
                continue  # thin the four-member shapes; still hundreds of cases
            for others in others_options:
                out.append((others, [(v, e, s) for v, (e, s) in zip(combo, shape)]))
    return out


CASES = _cases()


def test_oracle_sample_is_substantial():
    assert len(CASES) > 1000


def test_engine_matches_an_independent_reading_of_426():
    mismatches = []
    for others, paired in CASES:
        ratings = others + [p[0] for p in paired]
        got = evaluate(ratings, paired=[Paired(*p) for p in paired]).final_degree
        want = _oracle(others, paired)
        if got != want:
            mismatches.append((others, paired, got, want))
    assert not mismatches, mismatches[:5]


def test_the_factor_never_makes_a_result_worse_than_ignoring_it():
    """4.26(d) guarantees the result is at least the no-factor result."""
    for others, paired in CASES:
        ratings = others + [p[0] for p in paired]
        with_rules = evaluate(ratings, paired=[Paired(*p) for p in paired]).final_degree
        assert with_rules >= final_degree(combine(ratings))
