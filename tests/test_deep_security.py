"""Security and privacy lane: findings reproduced on 12705f8, each with a regression lock.

  SEC-1  Parser time grows with the square of a whitespace run. Five patterns
         let two adjacent whitespace quantifiers (or a whitespace quantifier and
         a lazy name capture) share out one run of spaces or blank lines in
         every possible way. A 30 KB letter whose "COMBINED EVALUATION FOR
         COMPENSATION" is followed by blank lines, or a row "1." followed by
         80,000 spaces, held one audit - and so a whole sweep - for minutes;
         the 500,000-character cap allows hours.
  SEC-2  A case.json JSON cannot decode (nesting past the recursion limit, bytes
         that are not UTF-8, a number over the int conversion limit) escaped
         CaseStore.load as RecursionError or ValueError: a traceback from show
         and resume, and a case sweep --fresh could never re-audit.
  SEC-3  Terminal control characters from a letter or a hand-edited case.json
         were printed as they stood, so a condition name could erase report
         lines, conceal everything after it, or write the clipboard (OSC 52).
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


# --------------------------------------------------------------------------
# SEC-2: a case.json that JSON cannot decode is corrupt, not a crash
# --------------------------------------------------------------------------

UNDECODABLE = {
    "nested arrays": lambda good: good.replace(b'"ratings": [', b'"ratings": [' + b"[" * 100_000 + b"]" * 100_000
                                                + b",", 1),
    "not UTF-8": lambda good: good.replace(b'"c1"', b'"c1\xff"', 1),
    "a 5,000-digit number": lambda good: good.replace(b'"stated_combined": 70', b'"stated_combined": '
                                                       + b"7" * 5000, 1),
}


@pytest.mark.parametrize("kind", list(UNDECODABLE))
@pytest.mark.parametrize("command", ["show", "resume"])
def test_an_undecodable_case_file_is_refused_with_exit_3(tmp_path, kind, command):
    from _support import EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED, main, tabular_letter

    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "pair.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                                                    ("Limitation of motion of the knee", 10), ("Tinnitus", 10)],
                            stated=70)
    code, out, err = main("audit", letter, "--case", "c1", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    path = store / "c1" / "case.json"
    path.write_bytes(UNDECODABLE[kind](path.read_bytes()))

    args = ["--case", "c1"] + (["--answer", "2=left"] if command == "resume" else [])
    code, out, err = main(command, *args, store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    # The state lane fixed the same defect (STATE-4) in the same except clause;
    # the release keeps that one implementation and its wording.
    assert "not valid JSON" in err


@pytest.mark.parametrize("kind", list(UNDECODABLE))
def test_sweep_fresh_re_audits_an_undecodable_case_file(tmp_path, kind):
    from _support import EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED, main, tabular_letter

    store, letters = tmp_path / "runs", tmp_path / "letters"
    tabular_letter(letters / "c1.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                                        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)], stated=70)
    code, out, err = main("sweep", letters, store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    path = store / "c1" / "case.json"
    path.write_bytes(UNDECODABLE[kind](path.read_bytes()))

    code, out, _ = main("sweep", letters, store=store)
    assert code == EXIT_CANNOT_PROCEED and "Nothing was deleted; re-audit with --fresh" in " ".join(out.split())
    # --fresh discards only a case it calls corrupt. A RecursionError or a
    # UnicodeDecodeError was not called that, so it could never be re-audited.
    code, out, err = main("sweep", letters, "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert "NEEDS YOUR ANSWER" in out


# --------------------------------------------------------------------------
# SEC-3: text Recheck did not write cannot drive the terminal
# --------------------------------------------------------------------------

ESC, BEL = "\x1b", "\x07"
CONTROLS = (ESC, BEL, "\x08", "\r", "\x9b", "\u202e")


def _no_controls(text: str) -> None:
    found = sorted({repr(ch) for ch in text if ch in CONTROLS})
    assert not found, f"terminal control characters reached the output: {found}"


def test_control_characters_in_a_letter_are_shown_escaped_not_sent_to_the_terminal(tmp_path):
    from _support import EXIT_AWAITING_HUMAN, EXIT_OK, main, tabular_letter

    store = tmp_path / "runs"
    # Erase the line above; conceal everything after; write the clipboard (OSC 52).
    letter = tabular_letter(tmp_path / "esc.txt", [
        ("Post-traumatic stress disorder", 70),
        (f"Tinnitus{ESC}[2K{ESC}[1A{ESC}[2K", 10),
        (f"Tinnitus{ESC}[8m", 10),
        (f"Tinnitus{ESC}]52;c;ZWNobyBQV05FRA=={BEL}", 10),
    ], stated=80)
    code, out, err = main("audit", letter, "--case", "esc", "--brief", store=store)
    assert code == EXIT_OK, out + err
    _no_controls(out + err)
    assert r"Tinnitus\x1b[2K" in out and r"\x1b]52;c;" in out and r"\x07" in out

    letters = tmp_path / "letters"
    tabular_letter(letters / "pair.txt", [("Post-traumatic stress disorder", 60),
                                          ("Right knee strain", 20),
                                          (f"Limitation of motion of the knee{ESC}[8m", 10), ("Tinnitus", 10)],
                            stated=70)
    code, out, err = main("sweep", letters, store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    _no_controls(out + err)
    assert r"\x1b[8m" in out


def test_control_characters_in_a_hand_edited_case_file_are_shown_escaped(tmp_path):
    import json

    from _support import EXIT_AWAITING_HUMAN, main, tabular_letter

    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "pair.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                                                    ("Limitation of motion of the knee", 10), ("Tinnitus", 10)],
                            stated=70)
    code, out, err = main("audit", letter, "--case", "c1", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    path = store / "c1" / "case.json"
    original = path.read_text(encoding="utf-8")
    name = "Limitation of motion of the knee"
    forged = f"{name}{ESC}[1A{ESC}[2K\u202e"
    raw = json.loads(original)
    raw["decisions"][2]["condition"] = forged
    raw["decisions"][2]["note"] = f"not stated{BEL}\r{ESC}[8m"
    path.write_text(json.dumps(raw), encoding="utf-8")

    # Edited in the facts alone, the name no longer matches the ratings and
    # trace it was read into, and the case is refused (STATE-1) - with nothing
    # raw on the terminal either way.
    code, out, err = main("show", "--case", "c1", store=store)
    _no_controls(out + err)
    assert code != EXIT_AWAITING_HUMAN and "differ" in err, out + err

    # Edited consistently everywhere the name is recorded, the case loads and
    # the report shows the controls escaped.
    raw = json.loads(original)
    raw["decisions"][2]["condition"] = forged
    raw["decisions"][2]["note"] = f"not stated{BEL}\r{ESC}[8m"
    raw["ratings"][2]["condition"] = forged
    extracted = [e for e in raw["trace"] if e.get("action") == "Extracted rating"]
    assert extracted[2]["detail"] == name
    extracted[2]["detail"] = forged
    path.write_text(json.dumps(raw), encoding="utf-8")

    code, out, err = main("show", "--case", "c1", store=store)
    _no_controls(out + err)
    assert r"\x1b[1A" in out and r"\u202e" in out and r"\x07\x0d\x1b[8m" in out, out + err
