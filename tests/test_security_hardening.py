"""Findings from the pre-release security review, each with a regression lock.

Three defects were found by attacking the running product rather than reading
the code. None of them produced a WRONG rating - the fail-closed gate and the
answer validation held throughout - but two of them let a run that had
computed nothing report success, which for an audit tool is its own kind of
wrong: the operator believes the letter was checked.

  D1  A corrupted graph_state.json deserialised into a state with no pending
      work. The graph reported COMPLETED, executed no nodes, wrote no result,
      and the CLI exited 0. Fixed by defining success as a PRODUCT outcome -
      the case must be complete and carry a recomputed degree - rather than
      trusting the framework's status.

  D2  read_document had no size limit. A 40 MB text file was read in 0.09s;
      a multi-gigabyte one would exhaust memory, and PDF decompression bombs
      are a known vector. Fixed with byte and page-count caps.

  D4  The persisted graph state was handed to the framework unvalidated. A
      JSON array produced an unhandled AttributeError from inside Strands'
      deserialize_state - a stack trace instead of an explanation. Found by
      parametrising D1's test over several corrupt payloads rather than one.
      Fixed by validating the shape and treating any deserialisation failure
      as a corrupt case.

  D3  Strands logs node and graph failures at ERROR, so a deliberate refusal
      printed a stack of "node failed / graph execution failed" lines beneath
      Recheck's own one-line explanation, making correct behaviour look like a
      crash. Fixed by suppressing framework logging unless --debug.

What held up under attack, and is pinned here so it stays that way: hand-editing
case.json to inject a fabricated laterality and mark it human-resolved does NOT
produce arithmetic.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from recheck.graph import (
    MAX_DOCUMENT_BYTES,
    MAX_PDF_PAGES,
    DocumentTooLarge,
    read_document,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
LETTER_NO_SIDE = ROOT / "fixtures" / "letters" / "08_nerve_no_side.txt"
CLEAN = ROOT / "fixtures" / "letters" / "06_agrees.txt"
CLASSIFICATIONS = ROOT / "fixtures" / "classifications" / "08_nerve_no_side.json"

EXIT_OK, EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED = 0, 2, 3


def cli(*args: str, store: pathlib.Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(store), *[str(a) for a in args]],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=180,
    )


@pytest.fixture
def store(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "runs"


def _interrupted_case(store: pathlib.Path, case: str) -> subprocess.CompletedProcess:
    result = cli("audit", LETTER_NO_SIDE, "--case", case,
                 "--scripted", "--classifications", CLASSIFICATIONS, store=store)
    assert result.returncode == EXIT_AWAITING_HUMAN, result.stdout + result.stderr
    return result


# --------------------------------------------------------------------------
# D1: a run that computes nothing must not report success
# --------------------------------------------------------------------------

def test_corrupted_graph_state_fails_loudly(store):
    _interrupted_case(store, "c1")
    (store / "c1" / "graph_state.json").write_text('{"evil": true}', encoding="utf-8")

    result = cli("resume", "--case", "c1", "--answer", "2=left,3=right",
                 "--scripted", "--classifications", CLASSIFICATIONS, store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED
    combined = result.stdout + result.stderr
    assert "reported success" in combined
    assert "may be corrupt" in combined

    case = json.loads((store / "c1" / "case.json").read_text(encoding="utf-8"))
    assert case["recomputed_degree"] is None, "a no-op run must not leave a result"


@pytest.mark.parametrize(
    "payload",
    [
        '{"evil": true}',      # valid object, no pending work
        "{}",                  # empty object
        "[]",                  # D4: array, not an object
        '{"nodes": "not-a-dict"}',
        "not json at all",     # unparseable
        "null",
        '"a string"',
    ],
)
def test_various_corrupt_graph_states_all_fail_closed(store, payload):
    case_id = "gs" + str(abs(hash(payload)) % 10000)
    _interrupted_case(store, case_id)
    (store / case_id / "graph_state.json").write_text(payload, encoding="utf-8")
    result = cli("resume", "--case", case_id, "--answer", "2=left,3=right",
                 "--scripted", "--classifications", CLASSIFICATIONS, store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED, result.stdout + result.stderr


def test_hand_edited_case_cannot_manufacture_a_rating(store):
    """Held up under attack. Pinned so it keeps holding.

    The tamper marks both nerve conditions as human-resolved on opposite
    sides - which, if believed, would apply 4.26 and change the band.
    """
    _interrupted_case(store, "c2")
    path = store / "c2" / "case.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    for decision in case["decisions"]:
        if "nerve" in decision["condition"]:
            decision["needs_human"] = False
            decision["laterality"] = "left" if "median" in decision["condition"] else "right"
            decision["extremity_group"] = "upper"
            decision["decided_by"] = "DETERMINISTIC"
    case["status"] = "resumed"
    path.write_text(json.dumps(case), encoding="utf-8")

    result = cli("resume", "--case", "c2", "--answer", "2=left,3=right",
                 "--scripted", "--classifications", CLASSIFICATIONS, store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "did not require human input" in (result.stdout + result.stderr)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["recomputed_degree"] is None


# --------------------------------------------------------------------------
# D2: bounded input
# --------------------------------------------------------------------------

def test_the_limits_are_sane_but_generous():
    """A rating decision is a few pages. The caps should be far above that
    and far below anything that threatens memory."""
    assert 1024 * 1024 <= MAX_DOCUMENT_BYTES <= 64 * 1024 * 1024
    assert 20 <= MAX_PDF_PAGES <= 500


def test_oversized_text_document_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("recheck.graph.MAX_DOCUMENT_BYTES", 2048)
    path = tmp_path / "big.txt"
    path.write_text("RATING DECISION\n" + "x" * 4096, encoding="utf-8")
    with pytest.raises(DocumentTooLarge, match="over the"):
        read_document(path)


def test_a_document_just_under_the_limit_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr("recheck.graph.MAX_DOCUMENT_BYTES", 4096)
    path = tmp_path / "ok.txt"
    path.write_text("RATING DECISION\n" + "x" * 100, encoding="utf-8")
    assert "RATING DECISION" in read_document(path)


def test_pdf_with_too_many_pages_is_refused(tmp_path, monkeypatch):
    from fpdf import FPDF

    monkeypatch.setattr("recheck.graph.MAX_PDF_PAGES", 3)
    pdf = FPDF()
    for index in range(5):
        pdf.add_page()
        pdf.set_font("Courier", size=10)
        pdf.cell(0, 5, f"page {index}", new_x="LMARGIN", new_y="NEXT")
    path = tmp_path / "many.pdf"
    pdf.output(str(path))
    with pytest.raises(DocumentTooLarge, match="pages"):
        read_document(path)


def test_oversized_document_refusal_reaches_the_cli_as_exit_3(store, tmp_path):
    """End to end: the refusal must escape the graph node and become an exit
    code, not a stack trace."""
    big = tmp_path / "huge.txt"
    big.write_text("RATING DECISION\n" + "x" * (MAX_DOCUMENT_BYTES + 1024), encoding="utf-8")
    result = cli("audit", big, "--case", "big", "--scripted", store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "over the" in (result.stdout + result.stderr)


# --------------------------------------------------------------------------
# D3: a refusal must read as a refusal, not a crash
# --------------------------------------------------------------------------

def test_refusal_output_contains_no_framework_trace(store, tmp_path):
    big = tmp_path / "huge.txt"
    big.write_text("RATING DECISION\n" + "x" * (MAX_DOCUMENT_BYTES + 1024), encoding="utf-8")
    result = cli("audit", big, "--case", "big2", "--scripted", store=store)
    combined = result.stdout + result.stderr
    for noise in ("Traceback", "node_id=", "graph execution failed", "node failed"):
        assert noise not in combined, f"framework noise leaked into product output: {noise!r}"
    assert "[recheck]" in combined, "the refusal must still explain itself"


def test_debug_flag_restores_framework_logging(store, tmp_path):
    """The noise is suppressed, not discarded - it must be recoverable."""
    big = tmp_path / "huge.txt"
    big.write_text("RATING DECISION\n" + "x" * (MAX_DOCUMENT_BYTES + 1024), encoding="utf-8")
    quiet = cli("audit", big, "--case", "q", "--scripted", store=store)
    loud = cli("--debug", "audit", big, "--case", "l", "--scripted", store=store)
    assert len(loud.stdout + loud.stderr) > len(quiet.stdout + quiet.stderr)


# --------------------------------------------------------------------------
# the happy paths must be untouched by all of the above
# --------------------------------------------------------------------------

def test_clean_letter_still_completes(store):
    result = cli("audit", CLEAN, "--case", "clean", "--scripted", store=store)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert "NO DISCREPANCY FOUND" in result.stdout


def test_full_interrupt_resume_still_completes(store):
    _interrupted_case(store, "happy")
    result = cli("resume", "--case", "happy", "--answer", "2=left,3=right",
                 "--scripted", "--classifications", CLASSIFICATIONS, store=store)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    case = json.loads((store / "happy" / "case.json").read_text(encoding="utf-8"))
    assert case["recomputed_degree"] == 80
    assert case["status"] == "complete"
