"""Second-round review of the extraction fixes, each with a regression lock.

The reviewer of commit 68af750 found one new defect and several residuals.
Every test below fails on 68af750 and passes after the fix, except the
controls, which say so.

  P1 (new)     A line with no lower-case letter always ended a sentence, so
               a hard-wrapped "(DC 5260)" line made the condition "is
               granted": a rating with no anatomy, and a wrong figure at exit 0.
  ARITH-F1     Percentages the token scan did not recognise were dropped:
               "10-percent", "ten (10) percent", "10 pct.", the Arabic percent
               sign, and "10 percent each".
  ARITH-F2     "left knee strain" and "the left knee strain" (or a "(DC 5260)"
               suffix) were different conditions, so a staged rating counted
               twice.
  ARITH-F9     After a stop heading, an unnumbered row, a wrapped row or a row
               numbered on from the table was still cut silently; a REASONS
               section naming a rating differently was refused.
  Form feed    A page break before a numbered row refused a valid table, and
               "^\\s*" made blank lines quadratic.
  Lookalikes   The mixed-alphabet check had no test the name check did not
               also satisfy; U+02BC refused "Quervainʼs"; U+2800 passed.
  FILES-P4-05  A Form XObject invoked many times kept pypdf busy for minutes
               on a PDF of a few kilobytes; no cap counted that work.
"""

from __future__ import annotations

import json
import time
import zlib

import pytest

from _support import EXIT_CANNOT_PROCEED, EXIT_OK, LETTERS, case_json, main
from recheck.extract.deterministic import parse
from recheck.graph import DocumentTooLarge, read_document

HEADER = (
    "*** SYNTHETIC TEST LETTER - NOT A REAL VA DECISION ***\n"
    "DEPARTMENT OF VETERANS AFFAIRS\n"
    "Regional Office\n\n"
    "Name: R. SYNTHETIC\n"
    "File Number: 00-000-002\n"
    "Date of Notification: April 2, 2026\n\n"
    "DECISION\n\n"
)


def prose(*sentences: str, stated: int) -> str:
    body = "\n\n".join(sentences)
    return f"{HEADER}{body}\n\nYour combined evaluation for compensation is {stated} percent.\n"


def tabular(*rows: str, stated: int, after: str = "") -> str:
    return ("RATING DECISION\n\n" + "\n".join(rows)
            + f"\n\nCOMBINED EVALUATION FOR COMPENSATION: {stated}%\n" + after)


def read(text: str) -> list[tuple[str, int]]:
    extraction = parse(text)
    assert extraction.ok, extraction.unparsed_reason
    return [(r.condition, r.percent) for r in extraction.ratings]


def refused(text: str) -> str:
    """The refusal reason; fails if the letter was read."""
    extraction = parse(text)
    got = [(r.condition, r.percent) for r in extraction.ratings]
    assert not extraction.ok and got == [], f"expected a refusal, read {got} stated {extraction.stated_combined}"
    return extraction.unparsed_reason or ""


def audit(tmp_path, name: str, text: str, *extra: str) -> tuple[int, str]:
    letter = tmp_path / f"{name}.txt"
    letter.write_text(text, encoding="utf-8")
    code, out, err = main("audit", letter, "--case", name, *extra, store=tmp_path / "runs")
    return code, out + err


# --------------------------------------------------------------------------
# P1: a line with no lower-case letter inside a wrapped sentence
# --------------------------------------------------------------------------

DC_LINE = prose(
    "Service connection for limitation of flexion of the left knee\n(DC 5260)\n"
    "is granted with an evaluation of 30 percent.",
    "Service connection for limitation of flexion of the right knee is granted with an evaluation of 30 percent.",
    stated=60,
)


def test_a_code_line_inside_a_wrapped_rating_sentence_stays_part_of_the_condition():
    """Before: the condition was "is granted", its group unrecognised."""
    extraction = parse(DC_LINE)
    assert [(r.condition, r.percent, r.extremity_group) for r in extraction.ratings] == [
        ("limitation of flexion of the left knee (DC 5260)", 30, "lower"),
        ("limitation of flexion of the right knee", 30, "lower"),
    ]


def test_a_code_line_split_rating_no_longer_prints_a_false_discrepancy(tmp_path):
    """Before: a replayed "none" for "is granted" (a correct answer for that
    text) dropped the bilateral pair: 50%, POTENTIAL DISCREPANCY, exit 0."""
    fixture = tmp_path / "none.json"
    fixture.write_text(json.dumps({"classifications": [
        {"condition": "is granted", "extremity_group": "none", "confidence": 0.9}]}), encoding="utf-8")
    code, output = audit(tmp_path, "dcline", DC_LINE, "--scripted", "--classifications", fixture)
    assert code == EXIT_OK, output
    assert "NO DISCREPANCY FOUND" in output
    assert case_json(tmp_path / "runs", "dcline")["recomputed_degree"] == 60


def test_an_abbreviation_line_inside_a_wrapped_rating_sentence_stays_part_of_the_condition():
    """Before: refused as "names no condition"."""
    letter = prose("Service connection for post-traumatic stress disorder\n(PTSD)\nis granted with an evaluation of "
                   "30 percent.", "Service connection for tinnitus is granted with an evaluation of 10 percent.",
                   stated=40)
    assert read(letter) == [("post-traumatic stress disorder (PTSD)", 30), ("tinnitus", 10)]


def test_an_all_caps_line_inside_a_wrapped_rating_sentence_stays_part_of_the_condition():
    """The next line continues in lower case, so the sentence is visibly one."""
    letter = prose("Service connection for\nLEFT KNEE STRAIN\nis granted with an evaluation of 10 percent.",
                   "Service connection for right knee strain is granted with an evaluation of 10 percent.",
                   stated=20)
    assert read(letter) == [("LEFT KNEE STRAIN", 10), ("right knee strain", 10)]


@pytest.mark.parametrize(
    "split",
    [
        "Service connection for left knee strain\n\nis granted with an evaluation of 10 percent.",
        "Service connection for left knee strain\n\nand instability is granted with an evaluation of 10 percent.",
        "Service connection for left knee strain\n\n(DC 5260) is granted with an evaluation of 10 percent.",
    ],
    ids=["verb", "conjunction", "code-only"],
)
def test_a_condition_name_that_is_only_the_end_of_a_split_sentence_is_refused(split):
    """A blank line inside a sentence (a pdftotext page break) leaves a
    "condition" with no anatomy; a model's "none" for it drops a pair."""
    refused(prose(split, "Service connection for right knee strain is granted with an evaluation of 10 percent.",
                  stated=20))


@pytest.mark.parametrize(
    "body",
    [
        "SERVICE CONNECTION FOR LEFT KNEE\nSTRAIN IS GRANTED WITH AN EVALUATION OF 10 PERCENT.",
        "LEFT KNEE STRAIN IS GRANTED WITH AN\nEVALUATION OF 10 PERCENT.",
        "SERVICE CONNECTION FOR LEFT KNEE STRAIN IS GRANTED\nWITH AN EVALUATION OF 10 PERCENT.",
        "LIMITATION OF FLEXION OF THE LEFT\nKNEE IS CONTINUED AS 10 PERCENT DISABLING.",
    ],
)
def test_wrapped_all_caps_prose_is_refused_not_read_as_a_fragment(body):
    """"STRAIN" alone has no anatomy; it must never reach the arithmetic.
    The first and last shapes were read as fragments before; the middle two
    were already refused and are kept as locks."""
    refused(HEADER + body + "\n\nPTSD IS CONTINUED AS 30 PERCENT DISABLING.\n\n"
            "YOUR COMBINED EVALUATION FOR COMPENSATION IS 40 PERCENT.\n")


def test_unwrapped_all_caps_prose_is_still_read():
    """Control (passes before and after)."""
    letter = (HEADER + "SERVICE CONNECTION FOR LEFT KNEE STRAIN IS GRANTED WITH AN EVALUATION OF 10 PERCENT.\n\n"
              "PTSD IS CONTINUED AS 30 PERCENT DISABLING.\n\nYOUR COMBINED EVALUATION FOR COMPENSATION IS 40 PERCENT.\n")
    assert read(letter) == [("LEFT KNEE STRAIN", 10), ("PTSD", 30)]


def test_an_all_caps_line_directly_above_a_rating_with_no_lead_in_is_refused():
    """A letterhead name or the start of the condition: the text cannot say."""
    letter = ("DEPARTMENT OF VETERANS AFFAIRS\nREGIONAL OFFICE\nJANE Q VETERAN\n"
              "Left cubital tunnel syndrome is continued as 20 percent disabling.\n"
              "Your combined evaluation for compensation is 20 percent.\n")
    assert "no lower-case letters" in refused(letter)


def test_a_heading_line_directly_above_a_rating_still_ends_the_letterhead():
    """Control: "DECISION" is a heading, so the rating after it is whole."""
    letter = ("DEPARTMENT OF VETERANS AFFAIRS\nJANE Q VETERAN\nVA File Number 123456789\nDECISION\n"
              "Left cubital tunnel syndrome is continued as 20 percent disabling.\n"
              "Your combined evaluation for compensation is 20 percent.\n")
    assert read(letter) == [("Left cubital tunnel syndrome", 20)]


def test_a_condition_name_starting_with_a_rating_verb_is_refused_however_it_arose():
    """_NOT_A_NAME_START is the guard, whatever split the sentence."""
    assert "mid-sentence" in refused(prose(
        "Service connection for tinnitus is granted with an evaluation of 10 percent. Has been granted "
        "with an evaluation of 10 percent.", stated=20))


# --------------------------------------------------------------------------
# ARITH-F1: every percent mark, however its number is written
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "third",
    [
        "Service connection for limitation of flexion of the right knee is granted. A 10-percent evaluation is "
        "assigned effective January 9, 2024.",
        "Service connection for limitation of flexion of the right knee is granted with an evaluation of ten (10) "
        "percent effective January 9, 2024.",
        "Service connection for limitation of flexion of the right knee is granted with a ten-percent evaluation.",
        "Service connection for limitation of flexion of the right knee is granted at 10 pct. effective January 9, "
        "2024.",
        "Service connection for limitation of flexion of the right knee is granted; evaluation 10٪.",
    ],
    ids=["10-percent", "ten-(10)-percent", "ten-percent", "pct", "arabic-percent-sign"],
)
def test_a_percentage_in_any_spelling_is_accounted_for_or_refused(third):
    """Before: the third rating was dropped; 40% against a correct 50%, exit 0."""
    refused(prose(
        "Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.",
        "Service connection for limitation of flexion of the left knee is granted with an evaluation of 10 percent.",
        third, stated=50))


def test_the_unread_percentage_is_named_as_written():
    reason = refused(prose("Service connection for tinnitus is granted with an evaluation of 10 percent.",
                           "A 10-percent evaluation is assigned for left knee strain.", stated=20))
    assert "10-percent" in reason


def test_the_unread_percentage_reaches_the_cli_as_could_not_read(tmp_path):
    letter = prose(
        "Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.",
        "Service connection for limitation of flexion of the left knee is granted with an evaluation of 10 percent.",
        "Service connection for limitation of flexion of the right knee is granted. A 10-percent evaluation is "
        "assigned.", stated=50)
    code, output = audit(tmp_path, "hyphen", letter, "--brief")
    assert code == EXIT_CANNOT_PROCEED and "COULD NOT READ THE LETTER" in output
    assert "DISCREPANCY" not in output


@pytest.mark.parametrize(
    "letter",
    [
        prose("Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.",
              "Service connection for limitation of flexion of the left knee and limitation of flexion of the right "
              "knee is granted with an evaluation of 10 percent each.", stated=50),
        prose("Service connection for left knee strain and right knee strain is granted with evaluations of 10 "
              "and 20 percent, respectively.", stated=30),
        tabular("  1. Scars, left and right knee ...... 10% each", stated=20),
    ],
    ids=["prose-each", "prose-respectively", "row-each"],
)
def test_one_statement_giving_percentages_to_several_conditions_is_refused(letter):
    """Before: "10 percent each" was read as ONE rating; 40% against 50%."""
    assert "each" in refused(letter)


def test_percent_like_letters_inside_other_words_are_not_percentages():
    """Control: "upper central" contains "per cent" only across a word boundary."""
    letter = prose("Service connection for tinnitus is granted with an evaluation of 10 percent. Upper central "
                   "incisor noted.", stated=10)
    assert read(letter) == [("tinnitus", 10)]


# --------------------------------------------------------------------------
# ARITH-F2: one condition, however its name is repeated
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "second",
    [
        "Evaluation of the left knee strain is increased to 30 percent effective March 1, 2025.",
        "Evaluation of left knee strain (DC 5260) is increased to 30 percent effective March 1, 2025.",
        "Evaluation of Left Knee Strain, is increased to 30 percent effective March 1, 2025.",
    ],
    ids=["article", "code", "case-and-punctuation"],
)
def test_a_staged_rating_under_a_near_identical_name_is_not_counted_twice(second):
    """Before ("the", "(DC 5260)"): 40, 20 and 30 counted, 70% against a
    correct 60%, exit 0. Case and a trailing comma were already folded."""
    letter = prose(
        "Service connection for post-traumatic stress disorder is granted with an evaluation of 40 percent.",
        "Service connection for left knee strain is granted with an evaluation of 20 percent effective "
        "January 9, 2024.", second, stated=60)
    assert "more than once" in refused(letter)


def test_an_acronym_that_repeats_the_name_does_not_make_it_a_new_condition():
    letter = prose(
        "Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.",
        "Evaluation of post-traumatic stress disorder (PTSD) is increased to 50 percent effective March 1, 2025.",
        stated=50)
    assert "more than once" in refused(letter)


def test_different_sides_stay_different_conditions():
    """Control: normalising must not merge a left and a right knee."""
    letter = prose("Service connection for left knee strain is granted with an evaluation of 10 percent.",
                   "Service connection for the right knee strain (DC 5260) is granted with an evaluation of "
                   "10 percent.", stated=20)
    assert read(letter) == [("left knee strain", 10), ("the right knee strain (DC 5260)", 10)]


# --------------------------------------------------------------------------
# ARITH-F9: what follows a stop heading
# --------------------------------------------------------------------------

TABLE = ("  1. Post-traumatic stress disorder ...... 30%", "  2. Scar, left knee ...... 10%")


@pytest.mark.parametrize(
    "cut",
    [
        "  Scar, right knee ...... 10%",
        "  3. Limitation of flexion, right\n     knee ...... 10%",
        "  3. Scar, left knee ...... 10%",
        "  Scar, left knee ...... 10%",
    ],
    ids=["unnumbered", "wrapped", "identical-row-numbered-on", "identical-row-unnumbered"],
)
def test_a_row_after_a_mid_list_heading_is_refused_unless_it_restates_its_own_row(cut):
    """Before: rows 1-2 read, the third cut; 40% against a correct 50%, exit 0."""
    letter = ("RATING DECISION\n\n" + "\n".join(TABLE) + "\n\nEvidence\n  VA examination dated March 1, 2024.\n"
              + cut + "\n\nCOMBINED EVALUATION FOR COMPENSATION: 50%\n")
    assert "Evidence" in refused(letter)


def test_a_wrapped_row_after_the_heading_reaches_the_cli_as_exit_3(tmp_path):
    letter = ("RATING DECISION\n\n" + "\n".join(TABLE) + "\n\nEvidence\n  VA examination dated March 1, 2024.\n"
              "  3. Limitation of flexion, right\n     knee ...... 10%\n\nCOMBINED EVALUATION FOR COMPENSATION: 50%\n")
    code, output = audit(tmp_path, "wrapped", letter, "--brief")
    assert code == EXIT_CANNOT_PROCEED and "COULD NOT READ THE LETTER" in output


def test_a_row_restated_with_its_own_number_under_reasons_is_accepted():
    """Control (passes before and after)."""
    letter = tabular(*TABLE, stated=40, after="\nREASONS FOR DECISION\n  2. Scar, left knee ...... 10%\n")
    assert read(letter) == [("Post-traumatic stress disorder", 30), ("Scar, left knee", 10)]


@pytest.mark.parametrize(
    "restatement",
    [
        "The evaluation of limitation of flexion of the right knee is increased to 20 percent because flexion is "
        "limited to 30 degrees.",
        # The row reads "Tinnitus (DC 6260)".
        "Evaluation of tinnitus is continued as 10 percent disabling because it is recurrent.",
    ],
    ids=["reworded-name", "without-the-code"],
)
def test_reasons_restating_a_row_under_a_differently_written_name_is_accepted(restatement):
    """Before: fixture 01 with this REASONS section was refused as a cut list."""
    letter = (LETTERS / "01_tabular.txt").read_text(encoding="utf-8") + "\nREASONS FOR DECISION\n\n" + restatement + "\n"
    assert [percent for _, percent in read(letter)] == [60, 20, 10, 10]


def test_reasons_restating_with_an_acronym_is_accepted():
    letter = tabular("  1. Post-traumatic stress disorder ...... 50%", "  2. Tinnitus ...... 10%", stated=60,
                     after="\nREASONS FOR DECISION\nEvaluation of post-traumatic stress disorder (PTSD) is increased "
                           "to 50 percent because of reduced reliability.\n")
    assert read(letter) == [("Post-traumatic stress disorder", 50), ("Tinnitus", 10)]


def test_reasons_naming_a_different_percentage_is_still_refused():
    """Control (refused before and after): same name, different value."""
    letter = (LETTERS / "01_tabular.txt").read_text(encoding="utf-8") + (
        "\nREASONS FOR DECISION\n\nThe evaluation of limitation of flexion of the right knee is increased to "
        "30 percent.\n")
    refused(letter)


# --------------------------------------------------------------------------
# Form feeds and the cost of whitespace
# --------------------------------------------------------------------------

def test_a_form_feed_before_a_numbered_row_is_still_a_row():
    """Before: "1 numbered lines but 2 rating rows"; a pdftotext page break."""
    letter = ("RATING DECISION\n\n  1. Post-traumatic stress disorder ...... 30%\n"
              "\f  2. Tinnitus ...... 10%\n\nCOMBINED EVALUATION FOR COMPENSATION: 40%\n")
    assert read(letter) == [("Post-traumatic stress disorder", 30), ("Tinnitus", 10)]


@pytest.mark.parametrize(
    "text",
    [
        "\n" * 20_000 + "x",
        "\f\n" * 20_000,
        "DECISION\n\n  1. a" + "." * 20_000 + "\n",
        "DECISION\n\n  1. a" + " " * 20_000 + "x\n",
    ],
    ids=["blank-lines", "form-feed-lines", "leader-dots", "spaces"],
)
def test_whitespace_and_leader_runs_parse_in_linear_time(text):
    """Before: 10 s, 17 s, 61 s and 24 s. "^\\s*" crossed line breaks, and the
    leader was retried from every dot or space of a run."""
    started = time.monotonic()
    parse(text)
    assert time.monotonic() - started < 2


# --------------------------------------------------------------------------
# Look-alike and invisible characters
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "letter",
    [
        prose("Service connection for tinnitus is granted with an evaluation of 10 percent.",
              "Service connection for left knee strain is granted at 10 pеrcent.", stated=10),
        prose("Service connection for tinnitus is granted with an evaluation of 10 percent.",
              "Service connection for left knee strain is granted with an evaluation of 10 pеrcent.",
              stated=10),
    ],
    ids=["unread-wording", "anchor-wording"],
)
def test_a_cyrillic_letter_hiding_a_percentage_is_refused(letter):
    """A lock, not a regression: this passed before, but nothing failed when
    _lookalike_problem was switched off. The condition names here are clean
    Latin, so only the mixed-alphabet check can see the Cyrillic "е"; without
    it the knee rating vanished from the list."""
    assert "alphabets" in refused(letter)


def test_a_modifier_letter_apostrophe_reads_as_an_apostrophe():
    """Before: refused as a word mixing alphabets."""
    letter = prose("Service connection for right De Quervainʼs tenosynovitis is granted with an evaluation of "
                   "10 percent.", stated=10)
    assert read(letter) == [("right De Quervain's tenosynovitis", 10)]


def test_a_braille_blank_inside_a_word_is_refused():
    """Before: "kn<U+2800>ee" was read, and recognised as nothing."""
    letter = tabular("  1. Left wrist strain ...... 30%", "  2. Tenosynovitis, right kn⠀ee ...... 30%", stated=60)
    assert "braille" in refused(letter)


def test_an_enclosing_mark_is_folded_like_other_marks():
    letter = tabular("  1. Left wrist strain ...... 30%", "  2. Tenosynovitis, right kn⃝ee ...... 30%", stated=60)
    assert [r.extremity_group for r in parse(letter).ratings] == ["upper", "lower"]
