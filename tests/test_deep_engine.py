"""Deep review, engine lane: regressions found by differential testing against an
independent 38 CFR 4.25 / 4.26 oracle."""

from __future__ import annotations

from _support import case_json, main, tabular_letter
from recheck.cfr.combine import combine, final_degree
from recheck.cfr.rating import Paired, evaluate

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
