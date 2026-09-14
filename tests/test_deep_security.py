"""Security and privacy lane: findings reproduced on 12705f8, each with a regression lock.

  SEC-1  Parser time grows with the square of a whitespace run. Five patterns
         let two adjacent whitespace quantifiers (or a whitespace quantifier and
         a lazy name capture) share out one run of spaces or blank lines in
         every possible way. A 30 KB letter whose "COMBINED EVALUATION FOR
         COMPENSATION" is followed by blank lines, or a row "1." followed by
         80,000 spaces, held one audit - and so a whole sweep - for minutes;
         the 500,000-character cap allows hours.
"""

from __future__ import annotations

import time

import pytest

from recheck.extract.deterministic import parse

TABLE = "RATING DECISION\n  1. Tinnitus ........ 10%\nCOMBINED EVALUATION FOR COMPENSATION: 10%\n"
RUN = 20_000


# --------------------------------------------------------------------------
# SEC-1: bounded parse time on long whitespace runs
# --------------------------------------------------------------------------

CASES = dict([
    ("row number then spaces", "RATING DECISION\n  1. " + " " * RUN + "x 10%\n"
                               "COMBINED EVALUATION FOR COMPENSATION: 10%\n"),
    ("row number then tabs", "RATING DECISION\n  1." + "\t" * RUN + "x 10%\n"
                             "COMBINED EVALUATION FOR COMPENSATION: 10%\n"),
    ("row after a stop heading", TABLE + "REASONS FOR DECISION\n1. " + " " * RUN + "x 10% ...\n"),
    ("parenthetical line", "RATING DECISION\n(X)" + " " * RUN + "Y\n" + TABLE),
    ("combined statement then spaces", TABLE + "COMBINED EVALUATION FOR COMPENSATION" + " " * RUN + "x\n"),
    ("combined statement then blank lines", TABLE + "COMBINED EVALUATION FOR COMPENSATION" + "\n" * RUN + "x\n"),
    ("stop heading then spaces", TABLE + "REASONS FOR DECISION" + " " * RUN + "x\n"),
])


@pytest.mark.parametrize("name", list(CASES))
def test_a_long_whitespace_run_is_parsed_in_linear_time(name):
    text = CASES[name]
    started = time.perf_counter()
    parse(text)
    elapsed = time.perf_counter() - started
    # 20,000 characters of whitespace took 6 to 30 s on 12705f8 and takes
    # hundredths of a second now; the bound leaves room for a loaded machine.
    assert elapsed < 3.0, f"{name}: {elapsed:.1f} s to parse {len(text):,} characters"


def test_the_rewritten_patterns_read_ordinary_letters_as_before():
    """The fix narrows backtracking only; what the patterns match is unchanged."""
    table = parse("RATING DECISION\n\n  1.\tLeft knee strain .......... 10%\n  2. Tinnitus ... 10 %\n\n"
                  "COMBINED EVALUATION FOR COMPENSATION :  20%\n")
    assert table.ok, table.unparsed_reason
    assert [(r.condition, r.percent) for r in table.ratings] == [("Left knee strain", 10), ("Tinnitus", 10)]
    assert table.stated_combined == 20

    colon_heading = parse(TABLE + "\nREASONS FOR DECISION :  \n  1. Tinnitus ........ 10%\n")
    assert colon_heading.ok, colon_heading.unparsed_reason

    cut = parse(TABLE + "\nREASONS FOR DECISION:\n  2. Left knee strain ........ 20%\n")
    assert not cut.ok and "REASONS FOR DECISION" in cut.unparsed_reason
