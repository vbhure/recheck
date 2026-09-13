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
    """"STRAIN" alone has no anatomy; it must never reach the arithmetic."""
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
