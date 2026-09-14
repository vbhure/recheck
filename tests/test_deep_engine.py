"""Deep review, engine lane: regressions found by differential testing against an
independent 38 CFR 4.25 / 4.26 oracle."""

from __future__ import annotations

import pytest

from _support import case_json, main, tabular_letter
from recheck.cfr.combine import combine, final_degree
from recheck.cfr.rating import Paired, evaluate
from recheck.classify import Decision
from recheck.materiality import assess, evaluate_for_report, paired_disabilities

# 60 PTSD, an 80 left leg, and three arm ratings of 10 (two left, one right).
# Every arm rating in the factor gives 90%; 4.26(d) leaves one left arm rating
# out and keeps the factor on the other two, giving 100%. No factor at all also
# gives 100%.
PARTIAL_426D_ROWS = [
    ("PTSD (DC 9411)", 60),
    ("Neuropathy, left lower extremity (DC 8520)", 80),
    ("Limitation of motion, left elbow (DC 5206)", 10),
    ("Limitation of motion, left wrist (DC 5215)", 10),
    ("Limitation of motion, right shoulder (DC 5201)", 10),
]


def test_partial_426d_exclusion_is_not_reported_as_the_figure_without_the_factor(tmp_path):
    """The ARITHMETIC section labelled the every-member-in-the-factor figure (90%)
    "final degree without the factor" when 4.26(d) had left only some bilateral
    disabilities out. Without the factor these ratings give 100%, not 90%."""
    ratings = [p for _, p in PARTIAL_426D_ROWS]
    ev = evaluate(ratings, paired=[Paired(80, "lower", "left"), Paired(10, "upper", "left"),
                                   Paired(10, "upper", "left"), Paired(10, "upper", "right")])
    # Preconditions: a PARTIAL 4.26(d) exclusion whose default figure differs from no factor at all.
    assert ev.bilateral_applied and len(ev.excluded_under_426d) == 1
    no_factor = final_degree(combine(ratings))
    assert (ev.final_degree, ev.alternative_final_degree, no_factor) == (100, 90, 100)

    letter = tabular_letter(tmp_path / "letter.txt", PARTIAL_426D_ROWS, stated=90)
    code, out, _ = main("audit", letter, "--case", "partial", store=tmp_path / "st")
    assert code == 0
    c = case_json(tmp_path / "st", "partial")
    assert (c["recomputed_degree"], c["alternative_degree"], c["bilateral_applied"]) == (100, 90, True)

    rows = [line.strip() for line in out.splitlines() if line.strip().startswith("final degree ")]
    for row in rows:
        if "without the factor" in row:
            assert row.endswith(f"{no_factor}%"), row
    assert any("4.26(d)" in row and row.endswith("90%") for row in rows), rows


def test_every_bilateral_disability_left_out_still_reads_with_the_factor(tmp_path):
    """The case the label was written for: 4.26(d) leaves the factor out entirely,
    and the alternative really is the figure with the factor."""
    rows = [("Neuropathy, left lower extremity (DC 8520)", 90), ("PTSD (DC 9411)", 30),
            ("Limitation of motion, left elbow (DC 5206)", 10), ("Limitation of motion, right elbow (DC 5206)", 10)]
    ev = evaluate([90, 30, 10, 10], paired=[Paired(90, "lower", "left"), Paired(10, "upper", "left"),
                                            Paired(10, "upper", "right")])
    assert not ev.bilateral_applied and ev.excluded_under_426d
    assert (ev.final_degree, ev.alternative_final_degree) == (100, 90)

    letter = tabular_letter(tmp_path / "letter.txt", rows, stated=90)
    code, out, _ = main("audit", letter, "--case", "all-out", store=tmp_path / "st")
    assert code == 0
    assert any(line.strip().startswith("final degree with the factor") and line.strip().endswith("90%")
               for line in out.splitlines())


def test_subtotal_of_one_pair_left_by_426d_does_not_cite_426b():
    """Both arms and both legs form one 4.26(b) group, but 4.26(d) keeps only the
    legs in the factor. The subtotal step lists two leg ratings and cited 4.26(b)
    ("all four extremities, one factor")."""
    paired = [Paired(10, "upper", "left"), Paired(10, "upper", "right"),
              Paired(20, "lower", "left"), Paired(60, "lower", "right")]
    ev = evaluate([10, 10, 20, 60, 70], paired=paired)
    assert {m.extremity for m in ev.bilateral_members} == {"lower"}
    assert len(ev.excluded_under_426d) == 2 and ev.final_degree == 100
    assert ev.steps[0].rule == "38 CFR 4.26"


def test_a_both_sides_evaluation_kept_out_of_the_factor_is_not_blamed_on_426c(tmp_path):
    """Bilateral pes planus 30 and a knee of unstated side 30: every possibility gives
    80%, and the report shows the established facts, where the pes planus stands
    alone. The note said 4.26(c) needs a disability "on both the left and right
    side ... got 30% both lower" - an evaluation that covers both sides."""
    letter = tabular_letter(tmp_path / "letter.txt", [("PTSD (DC 9411)", 50), ("Bilateral pes planus (DC 5276)", 30),
                                                      ("Limitation of flexion, knee (DC 5260)", 30)], stated=80)
    code, out, _ = main("audit", letter, "--case", "pp", store=tmp_path / "st")
    assert code == 0
    c = case_json(tmp_path / "st", "pp")
    assert c["recomputed_degree"] == 80 and c["decisions"][1]["laterality"] == "both"
    notes = [e["detail"] for e in c["trace"] if e["action"] == "Note"]
    assert notes, "precondition: the established-facts derivation carries a note"
    assert not any(n.startswith("4.26(c)") and "both lower" in n for n in notes), notes


@pytest.mark.parametrize("ratings,paired", [
    ([40, 24.9], None),                                   # was truncated to 24: 50% instead of refusing
    ([15.7], None),                                       # was truncated to 15
    ([True, 10], None),                                   # a bool is not a percentage
    (["40"], None),                                       # text is not a percentage
    ([-10], None),                                        # refused, but as an impossible COMBINED value
    ([10, 10], [Paired(10, "arm", "left"), Paired(10, "arm", "right")]),       # silently no factor
    ([10, 10], [Paired(10, "upper", "north"), Paired(10, "upper", "south")]),  # silently no factor
])
def test_the_engine_refuses_what_is_not_a_whole_percentage_of_an_arm_or_leg(ratings, paired):
    with pytest.raises(ValueError, match="whole percentage|arm or leg"):
        evaluate(ratings, paired=paired)


def test_the_engine_still_takes_every_whole_percentage():
    for value in range(0, 101):
        assert evaluate([value]).final_degree == final_degree(value)


# A left elbow 10, a right elbow 10, a left leg 90 and a shoulder 30 of unstated
# side. Wherever the shoulder is, it joins the elbows in the factor: 100%, and
# 4.26(d) plays no part (the prior rule gives 100% too). The established facts
# leave the shoulder out, so 4.26(d) dropped the elbows' factor there.
SHOULDER_ROWS = [
    ("Limitation of motion, left elbow (DC 5206)", 10),
    ("Limitation of motion, right elbow (DC 5206)", 10),
    ("Neuropathy, left lower extremity (DC 8520)", 90),
    ("Limitation of motion, shoulder (DC 5201)", 30),
]


def _decision(percent, group, side):
    return Decision("c", percent, group, side, None, None, None, None, None)


def _completion_accounts(decisions):
    """(4.26(d) decides, prior-rule figure) for every completion and reading, from the engine directly."""
    m = assess(decisions)
    accounts = set()
    for answers, _ in m.by_answers:
        facts = [(d.percent, d.extremity_group, d.laterality) for d in decisions]
        for index, (group, side) in zip(m.unknown, answers):
            facts[index] = (facts[index][0], group, side)
        for reading in (False, True):
            ev = evaluate([p for p, _, _ in facts], paired=paired_disabilities(facts, both_in_factor=reading),
                          lone_both_in_factor=reading)
            if ev.final_degree == m.possible[0]:
                accounts.add((bool(ev.excluded_under_426d), ev.alternative_final_degree if ev.excluded_under_426d
                              else None))
    return m, accounts


def test_a_426d_caveat_true_for_no_possible_fact_is_not_reported(tmp_path):
    """The report said 'This result depends on the 38 CFR 4.26(d) exception ... the prior
    rule gives 90%' and 'bilateral factor: not applied' - for a letter where every
    possible side of the shoulder applies the factor and 4.26(d) decides nothing."""
    decisions = [_decision(10, "upper", "left"), _decision(10, "upper", "right"),
                 _decision(90, "lower", "left"), _decision(30, "upper", "unknown")]
    m, accounts = _completion_accounts(decisions)
    assert m.settled and m.possible == (100,)
    assert accounts == {(False, None)}, "precondition: no completion is decided by 4.26(d)"
    ev, assumed = evaluate_for_report(decisions)
    assert not ev.excluded_under_426d and ev.bilateral_applied and set(assumed) == {3}

    letter = tabular_letter(tmp_path / "letter.txt", SHOULDER_ROWS, stated=90)
    code, out, _ = main("audit", letter, "--case", "shoulder", store=tmp_path / "st")
    assert code == 0
    assert "4.26(d)" not in out and "prior rule" not in out
    c = case_json(tmp_path / "st", "shoulder")
    assert (c["recomputed_degree"], c["bilateral_applied"], c["alternative_degree"]) == (100, True, 100)


def test_a_426d_caveat_that_depends_on_the_unknown_fact_names_the_assumption(tmp_path):
    """A knee of unstated side instead: on the left, 4.26(d) decides (prior rule 90%); on
    the right it does not. The caveat stays, tied to the fact the arithmetic assumes."""
    rows = SHOULDER_ROWS[:3] + [("Limitation of flexion, knee (DC 5260)", 30)]
    decisions = [_decision(10, "upper", "left"), _decision(10, "upper", "right"),
                 _decision(90, "lower", "left"), _decision(30, "lower", "unknown")]
    m, accounts = _completion_accounts(decisions)
    assert m.settled and len(accounts) > 1 and (True, 90) in accounts
    letter = tabular_letter(tmp_path / "letter.txt", rows, stated=90)
    code, out, _ = main("audit", letter, "--case", "knee", store=tmp_path / "st")
    assert code == 0
    flat = " ".join(out.split())
    assert "the prior rule gives 90% with the fact the arithmetic assumes" in flat
    assert "Arithmetic shown with an assumed fact" in out


def test_what_a_settled_report_says_about_426d_holds_for_every_possible_fact():
    """Property over small letters with one unknown side: the reported 4.26(d) account is one
    some completion has, and it is THE account whenever every completion agrees."""
    import itertools
    known = [("upper", "left"), ("upper", "right"), ("lower", "left"), ("lower", "right"), ("none", "unknown")]
    checked = 0
    for combo in itertools.combinations_with_replacement(list(itertools.product([10, 90], known)), 3):
        for unknown in (("upper", "unknown"), ("lower", "unknown")):
            decisions = [_decision(p, g, s) for p, (g, s) in combo] + [_decision(30, *unknown)]
            m, accounts = _completion_accounts(decisions)
            if not m.settled:
                continue
            ev, _ = evaluate_for_report(decisions)
            reported = (bool(ev.excluded_under_426d), ev.alternative_final_degree if ev.excluded_under_426d else None)
            assert reported in accounts, (combo, unknown, reported, accounts)
            if len(accounts) > 1:
                assert reported[0], (combo, unknown, reported, accounts)
            checked += 1
    assert checked > 100
