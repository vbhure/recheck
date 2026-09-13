"""When a person is asked, and when a result is withheld.

recheck.materiality decides deterministically whether an unknown fact is
worth a reviewer's time: every possible answer is run through the verified
4.25/4.26 engine, and a question is raised only if the answers lead to
different final degrees.

Defects regression-locked here:

  M1  Questions nobody needed. At 0d018c9 any limb condition without a
      stated side interrupted the run - a single knee with nothing to pair
      with, or a 0% knee that 4.26(c) excludes anyway. The caseload's value
      rests on asking only when the answer changes the rating.
  M2  Unknown facts silently treated as "no". At 0d018c9 a term outside the
      lexicon, with no classifier, became extremity group "none"; answering
      "unknown" to the side question then reported NO DISCREPANCY FOUND.
      See test_cross_process_resume.py for the end-to-end lock.
  M3  Settled, but reported with a different number. A knee of unstated side
      that joins the bilateral group whichever side it is on was left out of
      the factor at compute time: the trace said "every possibility gives
      80%" and the report said 70%, a potential discrepancy against a letter
      that stated the correct 80%.
  M4  "Both sides" in one evaluation. Whether 4.26 includes such an
      evaluation is a question of rating practice Recheck does not decide;
      when it changes the result, the case is UNDETERMINED, never computed.
"""

from __future__ import annotations

import itertools

import pytest

from _support import (
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    UNLISTED,
    actions,
    main,
    nodes_run,
    require_lexicon_abstains,
    run_audit,
    tabular_letter,
)
from recheck.cfr.rating import Paired, evaluate
from recheck.classify import Decision
from recheck.materiality import (
    MAX_COMPLETIONS,
    assess,
    evaluate_established,
    evaluate_for_report,
    options_for,
)


def D(percent: int, group: str, side: str, condition: str = "c") -> Decision:
    return Decision(condition, percent, group, side, None, None, None, None, None)


# --------------------------------------------------------------------------
# The enumeration
# --------------------------------------------------------------------------

def test_options_cover_every_fact_a_condition_could_turn_out_to_have():
    # "both": a letter that omits the side may be rating one evaluation of
    # both extremities (DC 5269 plantar fasciitis is "unilateral or bilateral").
    assert options_for(D(10, "unknown", "unknown")) == [
        ("none", "unknown"), ("upper", "left"), ("upper", "right"), ("upper", "both"),
        ("lower", "left"), ("lower", "right"), ("lower", "both")]
    assert options_for(D(10, "unknown", "left")) == [("none", "left"), ("upper", "left"), ("lower", "left")]
    assert options_for(D(10, "lower", "unknown")) == [("lower", "left"), ("lower", "right"), ("lower", "both")]
    assert options_for(D(10, "lower", "right")) == [("lower", "right")]
    assert options_for(D(10, "none", "unknown")) == [("none", "unknown")]


def test_a_single_sideless_limb_with_nothing_to_pair_is_immaterial():
    """M1."""
    m = assess([D(60, "none", "unknown"), D(20, "lower", "unknown"), D(10, "none", "unknown")])
    assert m.unknown == (1,)
    assert m.settled and not m.answers_matter
    assert m.possible == (70,)


def test_a_zero_percent_sideless_limb_is_immaterial():
    """M1: 4.26(c) needs a compensable disability on each side."""
    m = assess([D(60, "none", "unknown"), D(20, "lower", "right"), D(0, "lower", "unknown"),
                D(10, "none", "unknown")])
    assert m.settled


def test_an_unknown_that_4_26d_would_exclude_anyway_is_immaterial():
    """80, 60 and a 20/10 leg pair: with the factor 90%, without it 100%. The
    regulation keeps the more favourable result, so the side is irrelevant."""
    m = assess([D(80, "none", "unknown"), D(60, "none", "unknown"), D(20, "lower", "right"),
                D(10, "lower", "unknown")])
    assert m.settled and m.possible == (100,)


def test_a_sideless_pair_is_material_and_each_answer_carries_its_outcome():
    m = assess([D(60, "none", "unknown"), D(20, "lower", "right"), D(10, "lower", "unknown"),
                D(10, "none", "unknown")])
    assert m.answers_matter and not m.settled
    assert m.possible == (70, 80)
    # both: M21-1 V.iv.1.C.4.b puts a single bilateral evaluation into the
    # factor when another compensable disability of the same pair is rated.
    assert m.outcomes_for(2) == {("lower", "left"): (80,), ("lower", "right"): (70,), ("lower", "both"): (80,)}
    assert m.outcomes_for(1) == {}, "a condition with nothing unknown has no options"


def test_an_unknown_group_is_material_when_being_a_leg_would_pair_it():
    m = assess([D(60, "none", "unknown"), D(20, "lower", "right"), D(10, "unknown", "unknown"),
                D(10, "none", "unknown")])
    assert m.answers_matter and m.possible == (70, 80)
    assert m.outcomes_for(2)[("lower", "left")] == (80,)
    assert m.outcomes_for(2)[("none", "unknown")] == (70,)


def test_a_single_both_sides_evaluation_whose_reading_matters():
    """M4: one 30% evaluation naming both knees, nothing else in the legs, and
    both arms rated (left 20, right 10) - M21-1's open case -> 50% or 60%.

    This used migraine 20% and the knees alone. M21-1 settles that case (no
    factor: 40%), and treating it as open was red-team finding ARITH-F6; see
    tests/test_rt_graph.py."""
    m = assess([D(20, "upper", "left"), D(10, "upper", "right"), D(30, "lower", "both")])
    assert m.unknown == ()
    assert m.reading_matters and not m.answers_matter
    assert m.possible == (50, 60)


def test_a_single_both_sides_evaluation_whose_reading_does_not_matter():
    m = assess([D(60, "none", "unknown"), D(30, "lower", "both")])
    assert m.settled and m.possible == (70,)


def test_enumeration_that_would_be_sampled_is_refused_instead():
    """Exhaustive or not used: too many unknowns means asking, not sampling."""
    decisions = [D(10, "unknown", "unknown") for _ in range(6)]
    assert 5 ** 6 > MAX_COMPLETIONS
    m = assess(decisions)
    assert not m.exhaustive
    assert not m.settled
    assert m.answers_matter


def test_established_facts_only_pair_what_is_established():
    decisions = [D(60, "none", "unknown"), D(20, "lower", "right"), D(10, "lower", "unknown"),
                 D(10, "none", "unknown")]
    assert evaluate_established(decisions).bilateral_applied is False
    assert evaluate_established([D(30, "lower", "both"), D(20, "none", "unknown")]).bilateral_applied is False


# --------------------------------------------------------------------------
# M3: the reported degree is the settled degree
# --------------------------------------------------------------------------

def test_a_side_that_cannot_change_the_result_is_reported_at_the_settled_degree():
    """M3, the concrete case: left knee 10, right knee 10, knee of unstated side 60."""
    decisions = [D(10, "none", "unknown"), D(10, "lower", "left"), D(10, "lower", "right"),
                 D(60, "lower", "unknown")]
    m = assess(decisions)
    assert m.settled and m.possible == (80,)
    assert evaluate_established(decisions).final_degree == 70, "the figure the old compute reported"
    evaluation, assumed = evaluate_for_report(decisions)
    assert evaluation.final_degree == 80
    assert set(assumed) == {3}


def test_when_established_facts_suffice_nothing_is_assumed():
    evaluation, assumed = evaluate_for_report(
        [D(60, "none", "unknown"), D(20, "lower", "unknown"), D(10, "none", "unknown")])
    assert assumed == {}
    assert evaluation.final_degree == 70


SHAPES = [
    [("none", "unknown"), ("lower", "left"), ("lower", "right"), ("lower", "unknown")],
    [("none", "unknown"), ("lower", "left"), ("unknown", "unknown"), ("lower", "unknown")],
    [("none", "unknown"), ("lower", "both"), ("lower", "unknown")],
    [("upper", "left"), ("upper", "right"), ("lower", "left"), ("lower", "unknown")],
    [("none", "unknown"), ("unknown", "left"), ("upper", "right"), ("upper", "unknown")],
]


def test_whenever_a_result_is_settled_the_reported_degree_is_that_result():
    """M3 as a property, checked against engine runs over explicit completions."""
    checked = 0
    for shape in SHAPES:
        for combo in itertools.product([0, 10, 20, 40, 60], repeat=len(shape)):
            decisions = [D(p, g, s) for p, (g, s) in zip(combo, shape)]
            m = assess(decisions)
            if not m.settled:
                continue
            checked += 1
            reported = evaluate_for_report(decisions)[0].final_degree
            assert reported == m.possible[0], (combo, shape)
            # And independently: one explicit completion through the engine.
            first = [(p, g, s) for p, (g, s) in zip(combo, shape)]
            for index, (g, s) in zip(m.unknown, m.by_answers[0][0]):
                first[index] = (first[index][0], g, s)
            # Written out here rather than borrowed from recheck.materiality:
            # sided arm/leg ratings always pair; a single both-sides rating
            # pairs when another compensable rating of the same pair exists
            # (M21-1 V.iv.1.C.4.b). Settled results never hinge on the open case.
            paired = []
            for n, (p, g, s) in enumerate(first):
                if g not in ("upper", "lower"):
                    continue
                if s in ("left", "right"):
                    paired.append(Paired(p, g, s))
                elif s == "both" and any(q >= 10 and h == g for k, (q, h, _) in enumerate(first) if k != n):
                    paired.append(Paired(p, g, s))
            assert evaluate(list(combo), paired=paired).final_degree == reported
    assert checked > 500


# --------------------------------------------------------------------------
# Through the graph
# --------------------------------------------------------------------------

def test_a_single_sideless_knee_completes_without_a_question(tmp_path):
    """M1, end to end: no interrupt, a computed result, and the report says
    which unknown facts were judged immaterial."""
    letter = tabular_letter(tmp_path / "one.txt", [("Post-traumatic stress disorder", 60),
                            ("Limitation of motion of the knee", 20), ("Tinnitus", 10)], stated=70)
    store, result = run_audit(tmp_path / "runs", "one", letter)
    case = store.load("one")
    assert result.interrupts == []
    assert case.status == "complete"
    assert case.recomputed_degree == 70
    assert case.immaterial_unknowns == [1]
    assert "Unknown facts cannot change the result" in actions(store, "one")
    assert "Question for a reviewer" not in actions(store, "one")


def test_a_zero_percent_sideless_knee_completes_without_a_question(tmp_path):
    letter = tabular_letter(tmp_path / "zero.txt", [("Post-traumatic stress disorder", 60),
                            ("Right knee strain", 20), ("Limitation of motion of the knee", 0),
                            ("Tinnitus", 10)], stated=70)
    store, result = run_audit(tmp_path / "runs", "zero", letter)
    assert result.interrupts == []
    assert store.load("zero").status == "complete"


def test_a_material_sideless_pair_interrupts_with_its_stakes(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", [("Post-traumatic stress disorder", 60),
                            ("Right knee strain", 20), ("Limitation of motion of the knee", 10),
                            ("Tinnitus", 10)], stated=70)
    store, result = run_audit(tmp_path / "runs", "pair", letter)
    case = store.load("pair")
    assert case.status == "awaiting_human"
    assert case.possible_degrees == [70, 80]
    (interrupt,) = result.interrupts
    assert interrupt.id == "recheck:pair:assess"
    reason = interrupt.reason
    assert reason["possible_results"] == [70, 80]
    assert reason["stated"] == 70
    (asked,) = reason["conditions"]
    assert (asked["index"], asked["missing"], asked["accepted"]) == (2, ["side"], ["left", "right", "both", "unknown"])
    assert asked["outcomes"] == {"lower-left": [80], "lower-right": [70], "lower-both": [80]}
    assert "compute" not in nodes_run(store, "pair")


def test_a_settled_unknown_side_is_reported_at_the_settled_degree(tmp_path):
    """M3 end to end. The letter states the correct 80%; the old compute said
    70% and raised a (downward) potential discrepancy."""
    letter = tabular_letter(tmp_path / "knees.txt", [("Tinnitus", 10), ("Left knee strain", 10),
                            ("Right knee strain", 10), ("Degenerative arthritis of the knee", 60)], stated=80)
    code, out, _ = main("audit", letter, "--case", "knees", store=tmp_path / "runs")
    assert code == EXIT_OK
    assert "NO DISCREPANCY FOUND" in out
    assert "POTENTIAL DISCREPANCY" not in out
    from recheck.case import CaseStore

    store = CaseStore(tmp_path / "runs")
    case = store.load("knees")
    assert case.recomputed_degree == 80
    assert case.immaterial_unknowns == [3]
    assert "Arithmetic shown with an assumed fact" in actions(store, "knees")


def _both_letter(tmp_path, first: tuple[str, int], stated: int):
    return tabular_letter(tmp_path / "both.txt", [first, ("Bilateral knee strain", 30)], stated=stated)


def test_a_both_sides_evaluation_that_changes_the_result_is_undetermined(tmp_path):
    """M4: never computed, exit 3, and the report shows what it could be."""
    store_root = tmp_path / "runs"
    # M21-1's open case (see test_a_single_both_sides_evaluation_whose_reading_matters).
    letter = tabular_letter(tmp_path / "both.txt", [("Left shoulder strain", 20), ("Right shoulder strain", 10),
                            ("Bilateral knee strain", 30)], stated=50)
    code, out, _ = main("audit", letter, "--case", "both", store=store_root)
    assert code == EXIT_CANNOT_PROCEED
    from recheck.case import CaseStore

    store = CaseStore(store_root)
    case = store.load("both")
    assert case.status == "undetermined"
    assert case.possible_degrees == [50, 60]
    assert case.recomputed_degree is None
    assert "compute" not in nodes_run(store, "both")
    assert "UNDETERMINED - NOT COMPUTED" in out
    assert "50% or 60%" in out
    assert "recomputed final degree" not in out
    assert "NO DISCREPANCY FOUND" not in out


def test_a_both_sides_evaluation_that_cannot_change_the_result_completes(tmp_path):
    store_root = tmp_path / "runs"
    code, out, _ = main("audit", _both_letter(tmp_path, ("Post-traumatic stress disorder", 60), 70),
                        "--case", "both", store=store_root)
    assert code == EXIT_OK, out
    from recheck.case import CaseStore

    case = CaseStore(store_root).load("both")
    assert case.status == "complete"
    assert case.recomputed_degree == 70


def test_an_unknown_group_with_no_partner_does_not_bother_anyone(tmp_path):
    """Unknown is not automatically a question: a lone unlisted condition
    cannot form a 4.26 pair whatever it turns out to be."""
    require_lexicon_abstains(UNLISTED)
    letter = tabular_letter(tmp_path / "lone.txt", [("Post-traumatic stress disorder", 60), (UNLISTED, 10)],
                            stated=60)
    store, result = run_audit(tmp_path / "runs", "lone", letter)
    case = store.load("lone")
    assert result.interrupts == []
    assert case.status == "complete"
    assert case.load_decisions()[1].extremity_group == "unknown", "immaterial, but still honestly unknown"
    assert case.immaterial_unknowns == [1]


@pytest.mark.parametrize("status", ["awaiting_human", "undetermined"])
def test_no_result_is_computed_for_a_case_that_is_not_settled(tmp_path, status):
    rows = {
        "awaiting_human": [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                           ("Limitation of motion of the knee", 10), ("Tinnitus", 10)],
        "undetermined": [("Left shoulder strain", 20), ("Right shoulder strain", 10), ("Bilateral knee strain", 30)],
    }[status]
    letter = tabular_letter(tmp_path / "x.txt", rows, stated=70)
    store, _ = run_audit(tmp_path / "runs", "x", letter)
    case = store.load("x")
    assert case.status == status
    assert (case.recomputed_degree, case.recomputed_combined) == (None, None)
    assert "Final degree of disability" not in actions(store, "x")
