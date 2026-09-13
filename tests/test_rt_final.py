"""Final regression pass: where the two fix rounds met each other.

Each test fails on the code before its fix (commit 65b8d0c).

  RG-01  An unfinished line above a rating sentence with no lead-in was
         joined to it: "Dear Mr. Right," gave a knee its side, and a
         salutation, name or address was sent to the model with the name.
  RG-02  _key re-split all the text before every acronym: quadratic.
  RG-03  Diagnostic codes are not part of a restatement's key, so a second
         evaluation under a DIFFERENT code after a stop heading was dropped
         as a restatement of the first.
  RG-04  An UNDETERMINED case over the 4.26(d) member limit - no fact
         unknown - read "too many facts are unknown".
  RG-05  A complete case whose single both-sides evaluation leaves M21-1's
         reading open loaded and printed a result.
  RG-06  A forged trace entry under any action name but the three result
         names ("Final degree: 90%") printed at exit 0.
  RG-09  A triage row cut the UNDETERMINED reason mid-word.
  RG-10  The PDF child: Ctrl+C was held for the whole budget.
  RG-11  A 0% condition with an unknown side - which 4.26(c) keeps out of
         the bilateral factor - was asked about, and an answer that left it
         out was refused.
  RG-12  preflight did not say that audit and sweep use the provider only
         with --model.
  RG-13  "DCs 5260-5261" and "DC 5260, 5261" were not read as code references.
"""

from __future__ import annotations

import ast
import pathlib
import re
import time

import pytest

from _support import EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED, EXIT_OK, ROOT, main, tabular_letter
from recheck.case import TRACE_ACTIONS, CaseCorrupt, CaseStore
from recheck.extract import pdf_text
from recheck.extract.deterministic import _codes, _key, _name_problem, parse

FINAL = "Final degree of disability"
STATED = "\n\nYour combined evaluation for compensation is {}.\n"


def _refused(text: str) -> str:
    extraction = parse(text)
    assert not extraction.ok and not extraction.ratings, [(r.condition, r.percent) for r in extraction.ratings]
    return extraction.unparsed_reason or ""


def _edit(store_root: pathlib.Path, case_id: str, change) -> None:
    import json

    path = store_root / case_id / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


# ==========================================================================
# RG-01: a salutation, name or address is not part of a condition name
# ==========================================================================

@pytest.mark.parametrize("above", [
    "Dear Mr. Right,",
    "Dear Jane Q Veteran",
    "Jane Q Veteran\n12 Elm Street Apt 4\nSpringfield IL",
    "Decided on the twelfth of March",
])
def test_an_unfinished_line_above_a_rating_sentence_is_not_its_condition_name(above):
    letter = (f"{above}\nKnee strain is continued as 20 percent disabling.\n"
              "Left knee strain is continued as 10 percent disabling.\n"
              "Post-traumatic stress disorder is continued as 20 percent disabling." + STATED.format("40 percent"))
    assert "salutation" in _refused(letter)


def test_the_salutation_letter_never_reaches_a_figure_through_the_cli(tmp_path):
    letter = tmp_path / "dear_right.txt"
    letter.write_text("Dear Mr. Right,\nKnee strain is continued as 20 percent disabling.\n"
                      "Left knee strain is continued as 10 percent disabling.\n"
                      "Post-traumatic stress disorder is continued as 20 percent disabling."
                      + STATED.format("40 percent"), encoding="utf-8")
    code, out, err = main("audit", letter, "--case", "dear", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "50%" not in out and "Right, Knee" not in out


@pytest.mark.parametrize("wrapped", [
    "Service connection for\nLeft knee strain is granted with an evaluation of 10 percent.",
    "Service connection for left knee strain is\nGranted with an evaluation of 10 percent.",
])
def test_a_sentence_wrapped_after_a_joining_word_is_still_read(wrapped):
    extraction = parse(wrapped + STATED.format("10 percent"))
    assert extraction.ok and [(r.condition.lower(), r.percent) for r in extraction.ratings] == [
        ("left knee strain", 10)]


def test_a_letter_ending_its_salutation_with_punctuation_is_unaffected():
    extraction = parse("Dear Veteran:\n\nTinnitus is continued as 10 percent disabling." + STATED.format("10 percent"))
    assert extraction.ok and [(r.condition, r.percent) for r in extraction.ratings] == [("Tinnitus", 10)]


def test_the_committed_letters_read_the_same():
    """The change touches only letters with an unfinished line above an
    upper-case rating sentence; none of the committed ones has one."""
    for path in sorted((ROOT / "fixtures" / "letters").glob("*.txt")) + sorted(
            (ROOT / "fixtures" / "caseload").glob("*.txt")):
        extraction = parse(path.read_text(encoding="utf-8"))
        assert extraction.ok or "salutation" not in (extraction.unparsed_reason or ""), path.name


# ==========================================================================
# RG-02 / RG-13: the restatement key
# ==========================================================================

def test_the_restatement_key_is_linear_in_the_number_of_acronyms():
    name = "Knee strain " + "(AB) " * 16000
    started = time.monotonic()
    _key(name)
    assert time.monotonic() - started < 10


@pytest.mark.parametrize("name", [
    "Limitation of flexion of the knee (DCs 5260-5261)",
    "Limitation of flexion of the knee (DC 5260, 5261)",
    "Limitation of flexion of the knee (Diagnostic Codes 5260 and 5261)",
])
def test_every_spelling_of_a_code_list_is_a_code_reference(name):
    assert _codes(name) == {"5260", "5261"}
    assert _key(name) == "limitation flexion knee"
    assert _name_problem(name) is None


# ==========================================================================
# RG-03: a different diagnostic code is a different evaluation
# ==========================================================================

DC_RESTATE = """RATING DECISION

  1. Post-traumatic stress disorder (DC 9411) ........ 20%
  2. Scar, left knee (DC 7804) ....................... 10%

EVIDENCE
VA examination dated March 1, 2026.

Service connection for scar of the left knee (DC {code}) is granted with an evaluation of 10 percent.

COMBINED EVALUATION FOR COMPENSATION: 30%
"""


def test_a_statement_under_another_code_after_a_stop_heading_is_not_a_restatement():
    extraction = parse(DC_RESTATE.format(code="7805"))
    assert not extraction.ok and not extraction.ratings


def test_a_statement_under_the_same_code_is_still_a_restatement():
    extraction = parse(DC_RESTATE.format(code="7804"))
    assert extraction.ok and [r.percent for r in extraction.ratings] == [20, 10]


# ==========================================================================
# RG-04 / RG-09: an UNDETERMINED case whose ratings were not computed
# ==========================================================================

def test_an_over_the_member_limit_case_does_not_blame_unknown_facts(tmp_path):
    rows = [(f"{side} knee strain {i}", 10) for i in range(7) for side in ("Left", "Right")]
    letter = tabular_letter(tmp_path / "docs" / "many.txt", rows, stated=90)
    code, out, err = main("audit", letter, "--case", "many", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED, out + err
    flat = " ".join(out.split())
    assert "UNDETERMINED" in flat and "too many facts are unknown" not in flat.lower()
    assert "No possible ratings were computed" in flat

    code, out, err = main("sweep", tmp_path / "docs", store=tmp_path / "runs")
    rows_out = [line.strip() for line in out.splitlines()]
    start = next(i for i, line in enumerate(rows_out) if line.startswith("many"))
    assert rows_out[start] == "many         not computed:"
    reason = " ".join(" ".join(rows_out[start + 1:start + 4]).split())
    assert "too many unknown facts" not in reason
    # Wrapped at word boundaries, not cut: the first continuation line ends on a whole word.
    assert re.search(r"\w$", rows_out[start + 1])


# ==========================================================================
# RG-05: a complete case whose reading is still open
# ==========================================================================

def test_a_complete_case_whose_m21_reading_is_open_is_corrupt(tmp_path):
    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "a.txt", [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=60)
    assert main("audit", letter, "--case", "a", store=store)[0] == EXIT_OK

    def decision(percent, group, side):
        return {"condition": f"{side} {group} {percent}", "percent": percent, "extremity_group": group,
                "laterality": side, "group_by": "DETERMINISTIC", "side_by": "DETERMINISTIC", "confidence": None,
                "note": None, "evidence": None}

    _edit(store, "a", lambda raw: raw.update(decisions=[
        decision(20, "upper", "left"), decision(10, "upper", "right"), decision(30, "lower", "both")]))
    with pytest.raises(CaseCorrupt, match="could change its result"):
        CaseStore(store).load("a")
    code, out, err = main("show", "--case", "a", store=store)
    assert code == EXIT_CANNOT_PROCEED and FINAL not in out


# ==========================================================================
# RG-06: a trace entry no node writes
# ==========================================================================

@pytest.mark.parametrize("entry", [
    {"action": "Final degree", "detail": "combined value 86 converted to the nearest degree divisible by 10",
     "value": "90%"},
    {"action": "Result", "detail": "VA erred: the correct rating is 90%", "value": "90%"},
])
def test_a_trace_entry_under_an_unknown_action_is_refused(tmp_path, entry):
    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "t.txt", [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=60)
    assert main("audit", letter, "--case", "t", store=store)[0] == EXIT_OK
    _edit(store, "t", lambda raw: raw["trace"].append(
        {"actor": "DETERMINISTIC", "confidence": None, "evidence": None, "rule": None, **entry}))
    with pytest.raises(CaseCorrupt, match="no Recheck node writes"):
        CaseStore(store).load("t")
    code, out, err = main("show", "--case", "t", store=store)
    assert code == EXIT_CANNOT_PROCEED and "90%" not in out


def _written_actions() -> set[str]:
    """Every action passed to a trace's add() anywhere in the source."""
    constants: dict[str, str] = {}
    calls: list[ast.expr] = []
    for path in (ROOT / "src" / "recheck").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_ACTION"):
                        constants[target.id] = node.value.value
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add"
                    and len(node.args) >= 2):
                calls.append(node.args[1])
    actions = set()
    for arg in calls:
        if isinstance(arg, ast.Constant):
            actions.add(arg.value)
        elif isinstance(arg, ast.Name) and arg.id in constants:
            actions.add(constants[arg.id])
        elif isinstance(arg, ast.IfExp):
            actions.update(a.value for a in (arg.body, arg.orelse) if isinstance(a, ast.Constant))
        else:
            raise AssertionError(f"a trace action this test cannot read statically: {ast.dump(arg)}")
    return actions


def test_the_allowed_actions_are_exactly_the_actions_the_nodes_write():
    assert _written_actions() == set(TRACE_ACTIONS)


# ==========================================================================
# RG-10: the PDF child
# ==========================================================================

def test_the_pdf_child_is_stopped_at_its_budget_in_short_waits(monkeypatch):
    monkeypatch.setattr(pdf_text, "_BOOTSTRAP", "import sys, time; sys.stdin.buffer.read(); time.sleep(60)")
    limits = pdf_text.Limits(max_pages=1, max_chars=1, max_stream_bytes=1, max_content_bytes=1)
    started = time.monotonic()
    assert pdf_text.extract(b"%PDF-1.4", "x.pdf", limits, 1.0).kind == "timeout"
    assert time.monotonic() - started < 10


def test_the_pdf_child_ends_itself_at_its_budget(tmp_path):
    """An orphan does not run on: the child's own timer ends it."""
    import pickle
    import subprocess
    import sys

    limits = pdf_text.Limits(max_pages=1, max_chars=1, max_stream_bytes=1, max_content_bytes=1)
    request = pickle.dumps({"path": list(sys.path), "data": b"", "name": "x.pdf", "seconds": 0.5,
                            "limits": limits.__dict__})
    probe = ("import pickle, sys, time\n"
             "request = pickle.load(sys.stdin.buffer)\n"
             "sys.path[:0] = request['path']\n"
             "from recheck.extract import pdf_text\n"
             "pdf_text.read = lambda *a, **k: time.sleep(60)\n"
             "pdf_text._serve(request)\n")
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-I", "-c", probe], input=request, capture_output=True, timeout=30)
    assert done.returncode == 70 and time.monotonic() - started < 20


# ==========================================================================
# RG-12: preflight says when the provider is used
# ==========================================================================

def test_a_ready_preflight_says_audit_uses_the_provider_only_with_model(tmp_path, monkeypatch):
    from recheck.models import factory

    class Config:
        provider = "anthropic"

        def describe(self):
            return "anthropic (test double)"

    monkeypatch.setattr(factory, "load_config", lambda *a, **k: Config())
    monkeypatch.setattr(factory, "preflight", lambda config: [])
    code, out, _ = main("preflight", store=tmp_path / "runs")
    assert code == EXIT_OK
    assert "only when run with --model anthropic" in out


# ==========================================================================
# RG-11: a 0% condition with an unknown fact is never required
# ==========================================================================

ZERO_UNKNOWN = [("Bronchial asthma", 60), ("Degenerative arthritis of the knee", 20),
                ("Limitation of motion of the knee", 10), ("Scar of the ankle", 0), ("Tinnitus", 10)]


def test_a_non_compensable_unknown_is_not_asked_and_not_required(tmp_path):
    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "docs" / "zero.txt", ZERO_UNKNOWN, stated=70)
    code, out, err = main("audit", letter, "--case", "zero", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert "[1]" in out and "[2]" in out and "Scar of the ankle" not in out.split("QUESTION FOR THE REVIEWER")[1]
    assert '--answer "1=<left|right|both|unknown>,2=<left|right|both|unknown>"' in out

    code, out, err = main("sweep", tmp_path / "docs", store=store)
    assert code == EXIT_AWAITING_HUMAN and "Scar of the ankle" not in out, out

    code, out, err = main("resume", "--case", "zero", "--answer", "1=left,2=right", store=store)
    assert code == EXIT_OK, out + err
    assert re.search(rf"{FINAL}: 80%", out)
    trace = CaseStore(store).load("zero").trace
    assert any(e["action"] == "Unknown facts cannot change the result" for e in trace)


def test_a_non_compensable_unknown_may_still_be_answered(tmp_path):
    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "zero.txt", ZERO_UNKNOWN, stated=70)
    assert main("audit", letter, "--case", "zero", store=store)[0] == EXIT_AWAITING_HUMAN
    code, out, err = main("resume", "--case", "zero", "--answer", "1=left,2=right,3=left", store=store)
    assert code == EXIT_OK, out + err
