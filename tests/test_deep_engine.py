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
