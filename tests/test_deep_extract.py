"""Deep review, lane "extract": misreads found by generating letter variants with known truth.

Each test below failed on 12705f8 unless marked as a control. See the finding
ids in the section comments.
"""

from __future__ import annotations

import pytest

from _support import EXIT_CANNOT_PROCEED, EXIT_OK, main
from recheck.classify import derive_laterality
from recheck.extract.deterministic import _classify_extremity, parse

HEAD = "DEPARTMENT OF VETERANS AFFAIRS\n\nName: J. SYNTHETIC\n\n"


def _prose_letter(path, sentences, stated):
    body = "\n\n".join(sentences)
    path.write_text(f"{HEAD}DECISION\n\n{body}\n\nYour combined evaluation for compensation is {stated} percent.\n",
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# EXTRACT-1: a linked condition or the veteran's handedness written in a form
# the clause patterns did not list became the rated condition's side and
# extremity group, and a false POTENTIAL DISCREPANCY was reported at exit 0.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "condition,group,side",
    [
        ("Major depressive disorder related to right knee injury", "none", "unknown"),
        ("Major depressive disorder worsened by right knee injury", "none", "unknown"),
        ("Left knee strain, caused by right knee strain", "lower", "left"),
        ("Left hip strain, related to service-connected right knee disability", "lower", "left"),
        ("Left ankle sprain, attributable to right knee injury", "lower", "left"),
        ("Left knee strain, because of right ankle instability", "lower", "left"),
        ("Left knee strain (in connection with right knee surgery)", "lower", "left"),
        ("Left hip strain, compensating for right knee disability", "lower", "left"),
        ("Carpal tunnel syndrome, left wrist following right shoulder surgery", "upper", "left"),
        ("Carpal tunnel syndrome, left wrist, right hand-dominant", "upper", "left"),
        ("Carpal tunnel syndrome, left wrist, right-hand-dominant", "upper", "left"),
        ("Carpal tunnel syndrome, left wrist, right handed", "upper", "left"),
        ("Carpal tunnel syndrome, left wrist, right hand is dominant", "upper", "left"),
        # a limb word beside a non-extremity condition, joined in wording no pattern lists
        ("Obstructive sleep apnea, onset after right knee injury", "unrecognised", "right"),
        ("Post-traumatic stress disorder with right hand tremor", "unrecognised", "right"),
        # controls, unchanged: the listed forms, and a side that IS the rated condition's
        ("Left knee strain, secondary to right knee strain", "lower", "left"),
        ("Left wrist strain (right hand dominant)", "upper", "left"),
        ("Carpal tunnel syndrome, dominant right hand", "upper", "right"),
        ("Radiculopathy, right lower extremity, associated with lumbosacral strain", "lower", "right"),
    ],
)
def test_a_linked_condition_or_handedness_is_not_the_rated_side_or_group(condition, group, side):
    assert _classify_extremity(condition) == group
    assert derive_laterality(condition) == side


@pytest.mark.parametrize(
    "first",
    [
        # true reading: two LEFT leg disabilities, no bilateral factor: 20, 30 -> 44 -> 40%
        "Service connection for left hip strain, caused by your service-connected right knee disability, "
        "is granted with an evaluation of 20 percent.",
        # true reading: a mental disorder and a left leg disability: 20, 30 -> 44 -> 40%
        "Service connection for major depressive disorder related to right knee injury is granted with an "
        "evaluation of 20 percent.",
    ],
)
def test_a_linked_right_knee_does_not_create_a_bilateral_factor(tmp_path, first):
    letter = _prose_letter(tmp_path / "linked.txt", [
        first, "Service connection for left ankle strain is granted with an evaluation of 30 percent."], 40)
    code, out, _ = main("audit", letter, "--case", "linked", "--brief", store=tmp_path / "runs")
    assert "POTENTIAL DISCREPANCY" not in out
    assert "side: both" not in out
    assert code == EXIT_OK and "NO DISCREPANCY FOUND" in out


def test_handedness_does_not_put_a_left_wrist_in_the_bilateral_factor(tmp_path):
    # true reading: two LEFT arm disabilities, 30 and 20 -> 44 -> 40%, no factor
    letter = tmp_path / "handed.txt"
    letter.write_text(HEAD + "RATING DECISION\n\n"
                      "  1. Carpal tunnel syndrome, left wrist, right hand-dominant ........ 30%\n"
                      "  2. Lateral epicondylitis, left elbow ........ 20%\n\n"
                      "COMBINED EVALUATION FOR COMPENSATION: 40%\n", encoding="utf-8")
    code, out, _ = main("audit", letter, "--case", "handed", "--brief", store=tmp_path / "runs")
    assert "POTENTIAL DISCREPANCY" not in out
    assert code == EXIT_OK and "NO DISCREPANCY FOUND" in out


def test_a_limb_word_beside_a_mental_disorder_is_not_the_lexicons_call(tmp_path):
    letter = _prose_letter(tmp_path / "osa.txt", [
        "Service connection for obstructive sleep apnea, onset after right knee injury, is granted with an "
        "evaluation of 20 percent.",
        "Service connection for left ankle strain is granted with an evaluation of 30 percent."], 40)
    code, out, _ = main("audit", letter, "--case", "osa", "--brief", store=tmp_path / "runs")
    assert "POTENTIAL DISCREPANCY" not in out
    assert "extremity group: lower [lexicon]   side: right" not in out


# --------------------------------------------------------------------------
# EXTRACT-2: "both" was asserted whenever the words left and right both
# appeared in a name, whatever joined them. Any linked-condition wording or
# stray side word not stripped from the name turned a one-sided disability
# into a bilateral one and put it in the 4.26 factor; an abbreviated side
# beside the other side's word was read as that other side.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "condition,side",
    [
        ("Left hip strain, onset after right knee disability", "unknown"),
        ("Residuals of shell fragment wound, right thigh, with retained fragment left in place", "unknown"),
        ("LEFT shoulder impingement syndrome (subsequent to right shoulder surgery)", "unknown"),
        ("Rt knee strain, onset after left shoulder surgery", "unknown"),
        ("Radiculopathy, Lt. upper extremity (onset after right knee strain)", "unknown"),
        # controls: an explicit statement of both sides is still one evaluation of both
        ("Knee strain, left and right", "both"),
        ("Knee strain, right & left", "both"),
        ("Right knee and left knee strain", "both"),
        ("Bilateral knee strain", "both"),
        ("Strain of both knees", "both"),
        ("Left knee strain", "left"),
        ("L4-L5 radiculopathy, left lower extremity", "left"),
    ],
)
def test_both_sides_are_read_only_from_wording_that_names_both(condition, side):
    assert derive_laterality(condition) == side


def test_a_stray_right_does_not_put_a_left_hip_in_the_bilateral_factor(tmp_path):
    # true reading: two LEFT leg disabilities, 20 and 30 -> 44 -> 40%, no factor
    letter = _prose_letter(tmp_path / "hip.txt", [
        "Service connection for left hip strain, onset after right knee disability, is granted with an "
        "evaluation of 20 percent.",
        "Service connection for left ankle strain is granted with an evaluation of 30 percent."], 40)
    code, out, _ = main("audit", letter, "--case", "hip", "--brief", store=tmp_path / "runs")
    assert "side: both" not in out
    assert "POTENTIAL DISCREPANCY" not in out
    assert code != EXIT_OK or "NO DISCREPANCY FOUND" in out


# --------------------------------------------------------------------------
# EXTRACT-3: a staged combined evaluation after a stop heading was read as
# its first stage. The decision section refuses a second percentage in the
# combined statement; the same sentence after REASONS FOR DECISION reported
# the stage that had ended as the value under review, with a POTENTIAL
# DISCREPANCY at exit 0.
# --------------------------------------------------------------------------

ROWS = ("                RATING DECISION\n\n"
        "  1. Limitation of flexion, right knee (DC 5260) ........ 20%\n"
        "  2. Tinnitus (DC 6260) ................................. 10%\n\n")


@pytest.mark.parametrize("heading", ["REASONS FOR DECISION", "EVIDENCE"])
@pytest.mark.parametrize(
    "statement",
    [
        "Your combined evaluation for compensation is 20 percent until February 28, 2026,\nand 30 percent thereafter.",
        "Your combined evaluation for compensation is 20 percent until\nFebruary 28, 2026, and 30 percent\nthereafter.",
        "Your combined evaluation for compensation is 20 percent from January 1, 2026, and 30 percent from "
        "March 1, 2026.",
        "COMBINED EVALUATION FOR COMPENSATION: 20% (30% from March 1, 2026)",
    ],
)
def test_a_staged_combined_evaluation_after_a_heading_is_refused(heading, statement):
    extraction = parse(HEAD + ROWS + f"{heading}\n\nThe evidence was reviewed.\n\n{statement}\n")
    assert not extraction.ok
    assert extraction.ratings == []


def test_a_single_combined_evaluation_after_a_heading_is_still_read():
    # control
    extraction = parse(HEAD + ROWS + "REASONS FOR DECISION\n\nA 30 percent evaluation requires more.\n\n"
                       "Your combined evaluation for compensation is 30 percent. Your previous combined "
                       "evaluation for compensation is 20 percent.\n")
    assert extraction.ok and extraction.stated_combined == 30


def test_the_audit_does_not_report_a_discrepancy_against_an_ended_stage(tmp_path):
    letter = tmp_path / "staged.txt"
    letter.write_text(HEAD + ROWS + "REASONS FOR DECISION\n\nThe evidence was reviewed.\n\n"
                      "Your combined evaluation for compensation is 20 percent until February 28, 2026,\n"
                      "and 30 percent thereafter.\n", encoding="utf-8")
    code, out, _ = main("audit", letter, "--case", "staged", "--brief", store=tmp_path / "runs")
    assert "POTENTIAL DISCREPANCY" not in out
    assert code == EXIT_CANNOT_PROCEED
