"""Findings from attacking the running product, each with a regression lock.

None of these produced a wrong rating on their own, but several let a run
that established nothing look like a finished audit - which, for an audit
tool, is its own kind of wrong - or turned a deliberate refusal into a stack
trace.

  H1  Unbounded input. read_document had no size limit: a multi-gigabyte text
      file would exhaust memory, and PDF decompression bombs are a known
      vector. Byte and page-count caps, refused by name and exit 3.
  H2  Untrusted persisted state. A case file or a Strands session is input:
      malformed JSON, the wrong shape, unknown fields, an unknown status or a
      foreign schema version is refused as corrupt rather than guessed at,
      and a refused resume leaves case.json byte-identical.
  H3  Case ids arrive from the command line and from filenames. Anything that
      could escape the store, or name a Windows device, is refused.
  H4  Framework log noise. Strands logs node failures at ERROR, so a correct
      refusal printed "node failed / graph execution failed" beneath
      Recheck's one-line explanation. Suppressed unless --debug.
  H5  Tampering with case.json cannot manufacture a rating: a side marked
      as read from the letter for a name that states none is refused when the
      case is read, so no answer is accepted and nothing is computed.
"""

from __future__ import annotations

import json

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    case_json,
    cli,
    main,
    tabular_letter,
)
from recheck.case import SCHEMA_VERSION, Case, CaseCorrupt, CaseStore
from recheck.graph import MAX_DOCUMENT_BYTES, MAX_PDF_PAGES, DocumentTooLarge, read_document

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]


@pytest.fixture
def store(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def interrupted(store, tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    code, out, err = main("audit", letter, "--case", "c1", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    return store / "c1"


# --------------------------------------------------------------------------
# H1: bounded input
# --------------------------------------------------------------------------

def test_the_limits_are_sane_but_generous():
    assert 1024 * 1024 <= MAX_DOCUMENT_BYTES <= 64 * 1024 * 1024
    assert 20 <= MAX_PDF_PAGES <= 500


def test_oversized_text_document_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("recheck.graph.MAX_DOCUMENT_BYTES", 2048)
    path = tmp_path / "big.txt"
    path.write_text("RATING DECISION\n" + "x" * 4096, encoding="utf-8")
    with pytest.raises(DocumentTooLarge, match="over the"):
        read_document(path)


def test_a_document_under_the_limit_is_accepted(tmp_path, monkeypatch):
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
    big = tmp_path / "huge.txt"
    big.write_text("RATING DECISION\n" + "x" * (MAX_DOCUMENT_BYTES + 1024), encoding="utf-8")
    code, out, err = main("audit", big, "--case", "big", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "over the" in err
    assert "NO DISCREPANCY FOUND" not in out


def test_an_image_only_pdf_is_refused_through_the_cli(store, tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.rect(20, 20, 100, 60, style="F")
    scanned = tmp_path / "scanned.pdf"
    pdf.output(str(scanned))
    code, out, err = main("audit", scanned, "--case", "scan", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "requires OCR" in err


def test_a_corrupt_pdf_is_refused_not_crashed_on(store, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\n1 0 obj << /Type /Catalog /Pages 2 0 R >>\n%%EOF garbage")
    code, out, err = main("audit", broken, "--case", "broken", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "Traceback" not in out + err


# --------------------------------------------------------------------------
# H2: persisted state is untrusted input
# --------------------------------------------------------------------------

def _valid_case_dict() -> dict:
    from dataclasses import asdict

    return asdict(Case("c1", "x.txt", status="awaiting_human"))


@pytest.mark.parametrize(
    "content,fragment",
    [
        ("{not json", "not valid JSON"),
        ("[]", "not a JSON object"),
        ("null", "not a JSON object"),
        ('"a string"', "not a JSON object"),
        (json.dumps({**_valid_case_dict(), "schema_version": SCHEMA_VERSION - 1}), "schema_version"),
        (json.dumps({k: v for k, v in _valid_case_dict().items() if k != "schema_version"}), "schema_version"),
        (json.dumps({**_valid_case_dict(), "needs_human": True}), "unexpected fields"),
        (json.dumps({**_valid_case_dict(), "status": "resumed"}), "unknown status"),
        (json.dumps({**_valid_case_dict(), "status": "invalid_answer"}), "unknown status"),
        (json.dumps({**_valid_case_dict(), "decisions": [{"condition": "x"}]}), "malformed content"),
        (json.dumps({**_valid_case_dict(), "trace": [{"actor": "ROBOT", "action": "a", "detail": "d"}]}),
         "malformed content"),
    ],
)
def test_a_corrupt_or_foreign_case_file_is_refused(tmp_path, content, fragment):
    store = CaseStore(tmp_path / "runs")
    path = store.path_for("c1")
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CaseCorrupt, match=fragment):
        store.load("c1")


@pytest.mark.parametrize("command", ["resume", "show"])
def test_the_cli_refuses_a_corrupt_case_with_exit_3(store, interrupted, command):
    path = interrupted / "case.json"
    path.write_text("[]", encoding="utf-8")
    args = ["--case", "c1"] + (["--answer", "2=left"] if command == "resume" else [])
    code, _, err = main(command, *args, store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "not a JSON object" in err
    assert path.read_text(encoding="utf-8") == "[]"


@pytest.mark.parametrize(
    "payload",
    ['{"evil": true}', "{}", "[]", "not json at all", "null", '"a string"', '{"nodes": "not-a-dict"}'],
)
def test_a_corrupt_strands_session_is_refused_and_the_case_is_untouched(store, interrupted, payload):
    (session_file,) = (interrupted / "session").rglob("multi_agent.json")
    session_file.write_text(payload, encoding="utf-8")
    before = (interrupted / "case.json").read_bytes()
    code, out, err = main("resume", "--case", "c1", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "re-audit it with --fresh" in err
    assert "Traceback" not in out + err
    assert (interrupted / "case.json").read_bytes() == before


def test_hand_marked_facts_cannot_manufacture_a_rating(store, interrupted):
    """H5. The tamper claims the letter established the knee's side. The name
    states no side, so the case is refused as corrupt when it is read: every
    answer is refused, nothing is written and nothing computes. (It used to
    load, and the answer was rejected as "not asked about"; see
    tests/test_deep_state.py STATE-2 for the tamper that then finished at exit 0.)"""
    path = interrupted / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["decisions"][2].update(laterality="left", side_by="DETERMINISTIC")
    path.write_text(json.dumps(raw), encoding="utf-8")

    for attempt in ("2=left", "2=right", "1=left"):
        code, _, err = main("resume", "--case", "c1", "--answer", attempt, store=store)
        assert code == EXIT_CANNOT_PROCEED
        after = case_json(store, "c1")
        assert after["recomputed_degree"] is None and after["status"] == "awaiting_human"
        assert "which does not state one" in err and after == raw

    raw = case_json(store, "c1")
    raw["status"] = "ready"
    path.write_text(json.dumps(raw), encoding="utf-8")
    code, _, err = main("resume", "--case", "c1", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "which does not state one" in err
    assert case_json(store, "c1")["recomputed_degree"] is None


# --------------------------------------------------------------------------
# H3: case ids
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hostile",
    ["../escape", "..", "a/b", "a\\b", "with space", "C:x", "/abs", "", "x" * 65, "café",
     "con", "CON", "nul", "Aux", "prn", "com1", "COM9", "lpt1", "LPT9"],
)
def test_case_ids_that_escape_the_store_or_name_a_device_are_refused(tmp_path, hostile):
    with pytest.raises(ValueError):
        CaseStore(tmp_path / "runs").path_for(hostile)


@pytest.mark.parametrize("hostile", ["../escape", "a/b", "nul", "COM1"])
def test_the_cli_refuses_a_hostile_case_id_with_exit_3(store, tmp_path, hostile):
    code, _, err = main("audit", LETTERS / "06_agrees.txt", "--case", hostile, store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "case id" in err
    assert not (tmp_path / "escape").exists()


# --------------------------------------------------------------------------
# H4: a refusal reads as a refusal, in a real process
# --------------------------------------------------------------------------

def test_refusal_output_contains_no_framework_trace_unless_debug(store, tmp_path):
    big = tmp_path / "huge.txt"
    big.write_text("RATING DECISION\n" + "x" * (MAX_DOCUMENT_BYTES + 1024), encoding="utf-8")
    quiet = cli("audit", big, "--case", "q", store=store)
    combined = quiet.stdout + quiet.stderr
    assert quiet.returncode == EXIT_CANNOT_PROCEED
    for noise in ("Traceback", "node_id=", "graph execution failed", "node failed", "Graph without execution"):
        assert noise not in combined, f"framework noise leaked into product output: {noise!r}"
    assert "[recheck]" in combined, "the refusal must still explain itself"

    loud = cli("--debug", "audit", big, "--case", "l", store=store)
    assert loud.returncode == EXIT_CANNOT_PROCEED
    assert len(loud.stdout + loud.stderr) > len(combined), "the noise is suppressed, not discarded"


def test_the_happy_path_is_untouched_by_the_hardening(store):
    code, out, _ = main("audit", LETTERS / "06_agrees.txt", "--case", "clean", store=store)
    assert code == EXIT_OK
    assert "NO DISCREPANCY FOUND" in out
