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
        ("Obstructive sleep apnea, onset after right knee injury", "unrecognised", "unknown"),  # side: EXTRACT-7
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


@pytest.mark.parametrize("heading", ["REASONS FOR DECISION", "EVIDENCE"])
@pytest.mark.parametrize(
    "statement",
    [
        # the same staged sentence, with an abbreviated month or a semicolon: the tail's sentence split broke
        # it at "Feb." or ";" and the second stage was in another "sentence" (still read as 20% on 6b317f7)
        "Your combined evaluation for compensation is 20 percent until Feb. 28, 2026, and 30 percent thereafter.",
        "Your combined evaluation for compensation is 20 percent effective Jan. 9, 2026, and 30 percent "
        "effective Mar. 1, 2026.",
        "Your combined evaluation for compensation is 20 percent until February 28, 2026; and 30 percent "
        "thereafter.",
    ],
)
def test_a_staged_combined_evaluation_with_an_abbreviation_or_semicolon_is_refused(heading, statement):
    extraction = parse(HEAD + ROWS + f"{heading}\n\nThe evidence was reviewed.\n\n{statement}\n")
    assert not extraction.ok
    assert extraction.stated_combined is None


def test_the_audit_does_not_report_a_discrepancy_against_an_abbreviated_ended_stage(tmp_path):
    letter = tmp_path / "staged.txt"
    letter.write_text(HEAD + ROWS + "REASONS FOR DECISION\n\nThe evidence was reviewed.\n\n"
                      "Your combined evaluation for compensation is 20 percent until Feb. 28, 2026,\n"
                      "and 30 percent thereafter.\n", encoding="utf-8")
    code, out, _ = main("audit", letter, "--case", "staged", "--brief", store=tmp_path / "runs")
    assert "POTENTIAL DISCREPANCY" not in out
    assert code == EXIT_CANNOT_PROCEED


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


# --------------------------------------------------------------------------
# EXTRACT-4: a hard-wrapped combined evaluation statement ("Your combined
# evaluation for\ncompensation is 30 percent.") was not found, so an ordinary
# prose letter came back "COULD NOT READ THE LETTER" (exit 3). The rating
# statements themselves are matched across line breaks; this one was not.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cut", range(2, 6))  # a break inside the phrase; 1 and 6 were already read
def test_a_wrapped_combined_evaluation_statement_is_read(cut):
    words = "Your combined evaluation for compensation is 30 percent.".split(" ")
    statement = " ".join(words[:cut]) + "\n" + " ".join(words[cut:])
    extraction = parse(HEAD + "DECISION\n\n"
                       "Service connection for right knee strain is granted with an evaluation of\n"
                       "20 percent effective January 9, 2026.\n\n"
                       "Service connection for tinnitus is granted with an evaluation of 10 percent\n"
                       "effective January 9, 2026. " + statement + "\n")
    assert extraction.ok, extraction.unparsed_reason
    assert extraction.stated_combined == 30
    assert [r.percent for r in extraction.ratings] == [20, 10]


def test_a_wrapped_tabular_combined_statement_is_read():
    extraction = parse(HEAD + "RATING DECISION\n\n  1. Tinnitus (DC 6260) ........ 10%\n\n"
                       "COMBINED EVALUATION FOR\nCOMPENSATION: 10%\n")
    assert extraction.ok, extraction.unparsed_reason
    assert extraction.stated_combined == 10


def test_a_wrapped_previous_combined_evaluation_is_still_history():
    extraction = parse(HEAD + "RATING DECISION\n\n  1. Tinnitus (DC 6260) ........ 10%\n\n"
                       "Your previous combined evaluation for\ncompensation is 0 percent. Your combined\n"
                       "evaluation for compensation is 10 percent.\n")
    assert extraction.ok, extraction.unparsed_reason
    assert extraction.stated_combined == 10


def test_a_wrapped_staged_combined_evaluation_after_a_heading_is_refused():
    # control: refused on 12705f8 only because the wrapped statement was not found at all
    extraction = parse(HEAD + ROWS + "REASONS FOR DECISION\n\nThe evidence was reviewed.\n\n"
                       "Your combined evaluation for\ncompensation is 20 percent until\nFebruary 28, 2026, and "
                       "30 percent\nthereafter.\n")
    assert not extraction.ok


def test_a_wrapped_combined_statement_letter_audits(tmp_path):
    letter = tmp_path / "wrapped.txt"
    letter.write_text(HEAD + "DECISION\n\nService connection for right knee strain is granted with an evaluation of\n"
                      "20 percent effective January 9, 2026.\n\nService connection for tinnitus is granted with an "
                      "evaluation of 10 percent\neffective January 9, 2026. Your combined evaluation for\n"
                      "compensation is 30 percent.\n", encoding="utf-8")
    code, out, _ = main("audit", letter, "--case", "wrapped", "--brief", store=tmp_path / "runs")
    assert code == EXIT_OK and "NO DISCREPANCY FOUND" in out


# --------------------------------------------------------------------------
# EXTRACT-5: after a form feed (pdftotext's page break) each rating's stored
# source_line was ANOTHER line of the letter - row 2's case.json evidence
# read "1. Limitation of flexion, right knee ... 20%". Line numbers count
# "\n", but the line text was looked up in str.splitlines(), which also
# breaks at \f, \v, \x1c-\x1e, \x85, U+2028 and U+2029.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("brk", ["\f", "\v", "\x1c", "\x85", " ", " "])
def test_each_rating_keeps_its_own_source_line_after_a_page_break(brk):
    letter = (HEAD + brk + "\n                RATING DECISION\n\n"
              "  1. Limitation of flexion, right knee (DC 5260) ........ 20%\n"
              "  2. Limitation of flexion, left knee (DC 5260) ......... 10%\n\n"
              "COMBINED EVALUATION FOR COMPENSATION: 30%\n")
    extraction = parse(letter)
    assert extraction.ok, extraction.unparsed_reason
    for rating in extraction.ratings:
        assert rating.condition in rating.source_line
        assert rating.condition in letter.split("\n")[rating.source_line_number - 1]


def test_a_prose_rating_keeps_its_own_source_line_after_a_page_break():
    letter = (HEAD + "\f\nDECISION\n\n"
              "Service connection for right knee strain is granted with an evaluation of 20 percent.\n\n"
              "Service connection for tinnitus is granted with an evaluation of 10 percent.\n\n"
              "Your combined evaluation for compensation is 30 percent.\n")
    extraction = parse(letter)
    assert extraction.ok, extraction.unparsed_reason
    for rating in extraction.ratings:
        assert rating.condition in rating.source_line


# --------------------------------------------------------------------------
# EXTRACT-6: "granted with a noncompensable evaluation" has no percent mark,
# so the completeness check never saw it and the evaluation was left out of
# the list the report prints as the evaluations in the letter.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sentence",
    [
        "Service connection for right knee scar is granted with a noncompensable evaluation effective January 9, 2026.",
        "Evaluation of right knee scar is continued as noncompensable.",
    ],
)
def test_a_noncompensable_evaluation_is_read_as_zero_percent(sentence):
    extraction = parse(HEAD + "DECISION\n\n" + sentence + "\n\n"
                       "Service connection for left knee strain is granted with an evaluation of 20 percent.\n\n"
                       "Your combined evaluation for compensation is 20 percent.\n")
    assert extraction.ok, extraction.unparsed_reason
    assert [(r.condition, r.percent) for r in extraction.ratings] == [("right knee scar", 0), ("left knee strain", 20)]


def test_a_noncompensable_stage_and_a_later_increase_are_refused_as_a_repeat():
    # 12705f8 read only the 10 percent. A 0% stage is now a statement too, so the two are refused as any
    # staged prose rating is ("rates ... more than once"), not read as one of them.
    extraction = parse(HEAD + "DECISION\n\n"
                       "Service connection for left knee strain is granted with a noncompensable evaluation.\n\n"
                       "Evaluation of left knee strain is increased to 10 percent effective March 1, 2026.\n\n"
                       "Your combined evaluation for compensation is 10 percent.\n")
    assert not extraction.ok


# --------------------------------------------------------------------------
# EXTRACT-7: a non-extremity condition the lexicon has no hint for, linked
# to a limb in wording no pattern lists ("Scar, abdomen, onset after right
# knee injury"), was read as a right leg disability, entered the 4.26 factor,
# and a false POTENTIAL DISCREPANCY was reported at exit 0. A fact that only
# the text after an opening word (by, after, since, subsequent to, from)
# supplies is no longer taken from the name.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "condition,group,side",
    [
        ("Scar, abdomen, onset after right knee injury", "unrecognised", "unknown"),
        ("Hypertension (onset after right knee surgery)", "unrecognised", "unknown"),
        ("Hemorrhoids, onset since left ankle fracture", "unrecognised", "unknown"),
        # still a right leg disability on 6b317f7
        ("Scar, abdomen, incurred during right knee surgery", "unrecognised", "unknown"),
        ("Hypertension, while treated for right knee injury", "unrecognised", "unknown"),
        ("Left hip strain, incurred during parachute jump", "lower", "left"),  # control
        # the cost, accepted: read correctly on 12705f8, now left to materiality or a question
        ("Scar from shell fragment wound, right thigh", "unrecognised", "unknown"),
        # controls: the same facts before the word, or no such word
        ("Left hip strain, onset after knee injury", "lower", "left"),
        ("Shortening of the right leg by 2 inches", "lower", "right"),
        ("Limitation of flexion, right knee (DC 5260)", "lower", "right"),
    ],
)
def test_a_fact_only_text_after_an_opening_word_supplies_is_not_taken(condition, group, side):
    assert _classify_extremity(condition) == group
    assert derive_laterality(condition) == side


def test_an_abdominal_scar_linked_to_a_right_knee_is_not_in_the_bilateral_factor(tmp_path):
    letter = tmp_path / "scar.txt"
    letter.write_text(HEAD + "RATING DECISION\n\n"
                      "  1. Scar, abdomen, onset after right knee injury ........ 20%\n"
                      "  2. Left ankle strain ........ 30%\n\n"
                      "COMBINED EVALUATION FOR COMPENSATION: 40%\n", encoding="utf-8")
    code, out, _ = main("audit", letter, "--case", "scar", "--brief", store=tmp_path / "runs")
    assert "POTENTIAL DISCREPANCY" not in out
    assert "side: right" not in out


# --------------------------------------------------------------------------
# EXTRACT-8: "Evaluation of X is increased from 10 percent to 30 percent" -
# ordinary increase wording - was refused (COULD NOT READ THE LETTER, exit 3):
# no anchor read it, so both of its percentages were unaccounted for.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Evaluation of right knee strain is increased from 10 percent to 30 percent effective January 9, 2026.",
    "Evaluation of right knee strain is increased from 10 to 30 percent.",
    "Evaluation of right knee strain is increased from\n10 percent to 30\npercent.",
])
def test_an_increase_from_one_value_to_another_is_read(sentence):
    extraction = parse(HEAD + "DECISION\n\n" + sentence + "\n\n"
                       "Service connection for tinnitus is granted with an evaluation of 10 percent.\n\n"
                       "Your combined evaluation for compensation is 40 percent.\n")
    assert extraction.ok, extraction.unparsed_reason
    assert [(r.condition, r.percent) for r in extraction.ratings] == [("right knee strain", 30), ("tinnitus", 10)]


def test_an_increase_with_a_second_stage_is_still_refused():
    # control: a third percentage is not part of the increase
    extraction = parse(HEAD + "DECISION\n\nEvaluation of right knee strain is increased from 10 percent to 20 "
                       "percent effective January 9, 2026, and to 30 percent effective March 1, 2026.\n\n"
                       "Your combined evaluation for compensation is 30 percent.\n")
    assert not extraction.ok
