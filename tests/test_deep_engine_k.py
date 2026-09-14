"""Deep review, engine lane skeptic: the ARITHMETIC a settled report shows must be true for
some possible fact - not only its account of 4.26(d)."""

from __future__ import annotations

import dataclasses
import itertools

from _support import case_json, main, tabular_letter
from recheck.classify import Decision
from recheck.materiality import assess, evaluate_established, evaluate_for_report, readings_for

# A left elbow 10, bilateral plantar fasciitis 10 and a knee 10 of unstated side. On
# the established facts the lone both-sides rating stays out of the factor (27, 30%).
# Wherever the knee is - left, right or both - it joins the plantar fasciitis in the
# factor (29, 30%). The report printed "factor not applied" and combined value 27.
ROWS = [
    ("Limitation of motion, left elbow (DC 5206)", 10),
    ("Plantar fasciitis, bilateral (DC 5269)", 10),
    ("Limitation of flexion, knee (DC 5260)", 10),
]


def _decision(percent, group, side):
    return Decision("c", percent, group, side, None, None, None, None, None)


def _completion_figures(decisions):
    """(combined value, factor applied, 4.26(d) decides, prior figure) of every completion and reading."""
    m = assess(decisions)
    out = set()
    for answers, _ in m.by_answers:
        completed = list(decisions)
        for index, (group, side) in zip(m.unknown, answers):
            completed[index] = dataclasses.replace(decisions[index], extremity_group=group, laterality=side)
        facts = [(d.percent, d.extremity_group, d.laterality) for d in completed]
        for reading in readings_for(facts):
            out.add(_figures(evaluate_established(completed, both_in_factor=reading)))
    return m, out


def _figures(ev):
    decided = bool(ev.excluded_under_426d)
    return ev.combined_value, ev.bilateral_applied, decided, ev.alternative_final_degree if decided else None


def test_factor_not_applied_is_not_reported_when_every_possible_fact_applies_it(tmp_path):
    decisions = [_decision(10, "upper", "left"), _decision(10, "lower", "both"), _decision(10, "lower", "unknown")]
    m, figures = _completion_figures(decisions)
    assert m.settled and m.possible == (30,)
    assert figures == {(29, True, False, None)}, "precondition: every possible knee joins the factor"
    ev, assumed = evaluate_for_report(decisions)
    assert _figures(ev) in figures and set(assumed) == {2}

    letter = tabular_letter(tmp_path / "letter.txt", ROWS, stated=20)
    code, out, _ = main("audit", letter, "--case", "knee", store=tmp_path / "st")
    assert code == 0
    c = case_json(tmp_path / "st", "knee")
    assert (c["recomputed_degree"], c["recomputed_combined"], c["bilateral_applied"]) == (30, 29, True)
    assert "Arithmetic shown with an assumed fact" in out
    code, shown, _ = main("show", "--case", "knee", store=tmp_path / "st")
    assert code == 0 and "not applied" not in shown.split("ARITHMETIC")[1]


def test_what_a_settled_report_shows_is_true_for_some_possible_fact():
    """Property over small letters with one unknown fact, known both-sides ratings included."""
    known = [("upper", "left"), ("upper", "right"), ("lower", "left"), ("lower", "right"), ("none", "unknown"),
             ("upper", "both"), ("lower", "both")]
    checked = 0
    for combo in itertools.combinations_with_replacement(list(itertools.product([10, 30], known)), 2):
        for unknown in (("upper", "unknown"), ("lower", "unknown"), ("unknown", "unknown")):
            decisions = [_decision(p, g, s) for p, (g, s) in combo] + [_decision(10, *unknown)]
            m, figures = _completion_figures(decisions)
            if not m.settled:
                continue
            ev, _ = evaluate_for_report(decisions)
            assert ev.final_degree == m.possible[0]
            assert _figures(ev) in figures, (combo, unknown, _figures(ev), figures)
            checked += 1
    assert checked > 100
