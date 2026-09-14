"""Deep review, lane "state": a store directory someone else can write to.

Each test reproduces one finding and fails on 12705f8.

  STATE-1  a case.json whose stated value alone was edited turned a POTENTIAL
           DISCREPANCY into NO DISCREPANCY FOUND at exit 0, under a trace that
           still printed the letter's figure; an unreadable letter marked
           "ready" was finished as "Recheck computes 0%"
  STATE-2  a side the letter does not state, marked as read from the letter,
           let `resume` finish an open question without the reviewer: a
           figure at exit 0 on "the evaluations as printed"
  STATE-3  a link inside a case's Strands session was followed: every resume
           read the linked tree into memory, and a failed resume copied files
           from outside the store into it
  STATE-4  a case.json of nested brackets tracebacked show and resume; one
           that is not UTF-8 was not treated as corrupt, so sweep --fresh
           could not replace it
  STATE-5  possible degrees and "nobody was asked" indices edited on file were
           printed as Recheck's findings
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import stat

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    ROOT,
    UNLISTED,
    case_json,
    cli,
    main,
    require_lexicon_abstains,
    tabular_letter,
)
from recheck.case import CaseCorrupt, CaseStore

# 60, 20 and 10 combine to 71 -> 70% without the factor. With the knee of
# unstated side on the left, the pair of knees takes the factor: 80%.
PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]


@pytest.fixture
def store(tmp_path):
    return tmp_path / "runs"


def _edit(store_root: pathlib.Path, case_id: str, change) -> None:
    path = store_root / case_id / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _awaiting(store: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    letter = tabular_letter(tmp_path / "docs" / "p.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "p", store=store)[0] == EXIT_AWAITING_HUMAN
    return letter


# ==========================================================================
# STATE-1: the stated value and the evaluations are the ones the letter gave
# ==========================================================================

# 60, then the knees 20 and 10 with the factor (31), then 10: 75 -> 80%.
DISCREPANT = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20), ("Left knee strain", 10),
              ("Tinnitus", 10)]


def test_a_stated_value_edited_on_file_does_not_hide_a_discrepancy(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "d.txt", DISCREPANT, stated=70)
    code, out, _ = main("sweep", work, store=store)
    assert code == EXIT_OK and "stated 70% recomputed 80%" in _flat(out)

    _edit(store, "d", lambda raw: raw.update(stated_combined=80))
    with pytest.raises(CaseCorrupt, match="stated combined evaluation"):
        CaseStore(store).load("d")
    code, out, err = main("show", "--case", "d", store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "NO DISCREPANCY FOUND" not in out
    code, out, err = main("sweep", work, store=store)
    assert "NO DISCREPANCY FOUND 0" in _flat(out) and code != EXIT_OK, out + err


def test_an_evaluation_edited_on_file_is_not_computed_as_the_letters(store, tmp_path):
    """The knee of unstated side decides 70% or 80%. Edited to 0% and marked
    ready, resume computed 70% as "the evaluations as printed", exit 0."""
    _awaiting(store, tmp_path)

    def tamper(raw):
        raw["decisions"][2]["percent"] = 0
        raw["status"] = "ready"

    _edit(store, "p", tamper)
    with pytest.raises(CaseCorrupt, match="differ from"):
        CaseStore(store).load("p")
    code, out, err = main("resume", "--case", "p", store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert case_json(store, "p")["recomputed_degree"] is None


def test_a_letter_that_could_not_be_read_cannot_be_finished_as_a_result(store, tmp_path):
    letter = tmp_path / "u.txt"
    letter.write_text("nothing a rating decision says\n", encoding="utf-8")
    assert main("audit", letter, "--case", "u", store=store)[0] == EXIT_CANNOT_PROCEED
    _edit(store, "u", lambda raw: raw.update(status="ready"))
    with pytest.raises(CaseCorrupt, match="could not be read"):
        CaseStore(store).load("u")
    code, out, err = main("resume", "--case", "u", store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "Recheck computes" not in out and case_json(store, "u")["status"] == "ready"


def test_every_genuine_case_still_loads(store, tmp_path):
    work = tmp_path / "docs"
    work.mkdir()
    for name in ("01_tabular.txt", "02_prose.txt", "05_missing_side.txt", "06_agrees.txt",
                 "07_clinical_terms.txt", "08_clinical_terms_no_side.txt"):
        shutil.copy(LETTERS / name, work / name)
    fixture = ROOT / "fixtures" / "classifications" / "caseload.json"
    main("sweep", work, "--scripted", "--classifications", fixture, store=store)
    main("resume", "--case", "05_missing_side", "--answer", "2=left", store=store)
    loaded = 0
    for case_dir in sorted(store.iterdir()):
        if case_dir.is_dir() and not case_dir.name.startswith("."):
            CaseStore(store).load(case_dir.name)
            loaded += 1
    assert loaded == 6


# ==========================================================================
# STATE-2: a fact is one its recorded owner could have established
# ==========================================================================

def test_a_side_the_letter_does_not_state_cannot_finish_an_open_question(store, tmp_path):
    """resume without an answer computed 80% from the forged side, exit 0."""
    _awaiting(store, tmp_path)

    def tamper(raw):
        raw["decisions"][2].update(laterality="left", side_by="DETERMINISTIC")
        raw["status"] = "ready"

    _edit(store, "p", tamper)
    with pytest.raises(CaseCorrupt, match="does not state"):
        CaseStore(store).load("p")
    code, out, err = main("resume", "--case", "p", store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "POTENTIAL DISCREPANCY" not in out and "[letter]" not in out
    assert case_json(store, "p")["recomputed_degree"] is None


TWO_UNLISTED = [("Post-traumatic stress disorder", 60), (f"Left {UNLISTED}", 20), (f"Right {UNLISTED}", 10)]


PROVENANCE_TAMPERS = {
    "side changed from the name's": lambda raw: raw["decisions"][1].update(laterality="left"),
    "lexicon group changed": lambda raw: raw["decisions"][1].update(extremity_group="upper"),
    "side established by nobody": lambda raw: raw["decisions"][2].update(laterality="left", side_by=None),
    "side attributed to the reviewer": lambda raw: raw["decisions"][2].update(laterality="left", side_by="HUMAN"),
    "answer on file that is not the fact": lambda raw: raw.update(human_answers={"2": "lower-right"}),
    "answer for a condition it does not have": lambda raw: raw.update(human_answers={"9": "lower-left"}),
}


@pytest.mark.parametrize("name", sorted(PROVENANCE_TAMPERS))
def test_a_fact_its_owner_could_not_have_established_is_refused(store, tmp_path, name):
    _awaiting(store, tmp_path)
    CaseStore(store).load("p")
    _edit(store, "p", PROVENANCE_TAMPERS[name])
    with pytest.raises(CaseCorrupt, match=r"decision \[\d\]|does not have"):
        CaseStore(store).load("p")


def test_an_ai_group_cannot_be_passed_off_as_the_lexicon(store, tmp_path):
    """Relabelled DETERMINISTIC, the verdict and the triage stopped naming the
    AI classification the result rests on - at exit 0, figure unchanged."""
    require_lexicon_abstains(f"Left {UNLISTED}", f"Right {UNLISTED}")
    work = tmp_path / "docs"
    tabular_letter(work / "ai.txt", TWO_UNLISTED, stated=60)
    fixture = tmp_path / "map.json"
    fixture.write_text(json.dumps({"anatomy": {"zorblatt": "lower"}, "confidence": 0.9}), encoding="utf-8")
    code, out, err = main("sweep", work, "--scripted", "--classifications", fixture, store=store)
    assert "AI groups [1] [2]" in out, out + err
    CaseStore(store).load("ai")

    def launder(raw):
        for d in raw["decisions"][1:]:
            d.update(group_by="DETERMINISTIC", confidence=None)

    _edit(store, "ai", launder)
    with pytest.raises(CaseCorrupt, match="lexicon"):
        CaseStore(store).load("ai")
    code, out, err = main("sweep", work, store=store)
    assert "COULD NOT PROCEED 1" in _flat(out), out + err


def test_answers_accepted_over_two_rounds_still_load(store, tmp_path):
    """A group answered first and a side answered in a follow-up are both on file."""
    letter = tabular_letter(tmp_path / "u.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                                                  (UNLISTED, 10), ("Tinnitus", 10)], stated=70)
    assert main("audit", letter, "--case", "u", store=store)[0] == EXIT_AWAITING_HUMAN
    code, out, err = main("resume", "--case", "u", "--answer", "2=lower", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    code, out, err = main("resume", "--case", "u", "--answer", "2=left", store=store)
    assert code == EXIT_OK, out + err
    case = CaseStore(store).load("u")
    assert case.human_answers == {"2": "lower-left"} and case.recomputed_degree == 80


# ==========================================================================
# STATE-3: a link in a case's session is refused, never followed
# ==========================================================================

def _link(link: pathlib.Path, target: pathlib.Path) -> None:
    """A directory link anyone who can write the store can make: a junction on Windows."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _awaiting_case_with_a_link(store: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    _awaiting(store, tmp_path)
    outside = tmp_path / "outside" / "private"
    outside.mkdir(parents=True)
    (outside / "tax-return.txt").write_text("not the store's to copy", encoding="utf-8")
    sessions = [p for p in (store / "p" / "session").iterdir() if p.is_dir()]
    _link(sessions[0] / "agents" / "linked", outside.parent)
    return outside


def test_a_snapshot_does_not_read_through_a_link(store, tmp_path):
    _awaiting_case_with_a_link(store, tmp_path)
    with pytest.raises(OSError, match="link"):
        CaseStore(store).snapshot("p")


def test_a_failed_resume_does_not_copy_linked_files_into_the_store(store, tmp_path):
    """A read-only case.json makes the resume fail; restore() then wrote the
    linked directory back into the session as real files."""
    outside = _awaiting_case_with_a_link(store, tmp_path)
    case_file = store / "p" / "case.json"
    os.chmod(case_file, 0o444)
    try:
        code, out, err = main("resume", "--case", "p", "--answer", "2=left", store=store)
    finally:
        os.chmod(case_file, 0o666)
    assert code == EXIT_CANNOT_PROCEED, out + err
    (linked,) = (store / "p" / "session").glob("*/agents/linked")
    info = os.lstat(linked)
    still_a_link = stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    assert still_a_link, "the link was replaced by a copy of the directory it points to"
    assert (outside / "tax-return.txt").read_text(encoding="utf-8") == "not the store's to copy"


def test_a_session_larger_than_any_real_one_is_not_read_into_memory(store, tmp_path, monkeypatch):
    from recheck import case as case_module

    _awaiting(store, tmp_path)
    (store / "p" / "session" / "padding.bin").write_bytes(b"\0" * 4096)
    monkeypatch.setattr(case_module, "_SESSION_READ_LIMIT", 2048, raising=False)
    with pytest.raises(OSError, match="MB"):
        CaseStore(store).snapshot("p")
    code, out, err = main("resume", "--case", "p", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED and "Nothing was changed" in err, out + err


# ==========================================================================
# STATE-4: every case file that cannot be decoded is corrupt, not a crash
# ==========================================================================

UNDECODABLE = {
    "nested brackets": lambda path: path.write_text("[" * 5000 + "]" * 5000, encoding="utf-8"),
    "not UTF-8": lambda path: path.write_bytes(b'{"case_id": "p", "status": \xff}'),
    "a 5000-digit number": lambda path: path.write_text('{"schema_version": ' + "9" * 5000 + "}", encoding="utf-8"),
}


@pytest.mark.parametrize("name", sorted(UNDECODABLE))
def test_a_case_file_that_cannot_be_decoded_is_refused_as_corrupt(store, tmp_path, name):
    letter = _awaiting(store, tmp_path)
    UNDECODABLE[name](store / "p" / "case.json")

    with pytest.raises(CaseCorrupt, match="not valid JSON"):
        CaseStore(store).load("p")
    for args in (["show", "--case", "p"], ["resume", "--case", "p", "--answer", "2=left"]):
        code, out, err = main(*args, store=store)
        assert code == EXIT_CANNOT_PROCEED and "not valid JSON" in err, out + err

    # Corrupt, so a sweep with --fresh replaces it; without --fresh it deletes nothing.
    code, out, _ = main("sweep", letter.parent, store=store)
    assert "Nothing was deleted" in _flat(out)
    code, out, err = main("sweep", letter.parent, "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert CaseStore(store).load("p").status == "awaiting_human"


def test_nested_brackets_do_not_traceback_in_a_new_process(store, tmp_path):
    _awaiting(store, tmp_path)
    UNDECODABLE["nested brackets"](store / "p" / "case.json")
    done = cli("show", "--case", "p", store=store)
    assert done.returncode == EXIT_CANNOT_PROCEED and "Traceback" not in done.stderr, done.stderr


# ==========================================================================
# STATE-5: possible degrees and "nobody was asked" are Recheck's, re-derived
# ==========================================================================

def test_possible_degrees_edited_on_file_are_not_printed_as_recheck_s(store, tmp_path):
    letter = _awaiting(store, tmp_path)
    code, out, _ = main("show", "--case", "p", store=store)
    assert "70% or 80%" in out
    _edit(store, "p", lambda raw: raw.update(possible_degrees=[90]))
    with pytest.raises(CaseCorrupt, match="possible"):
        CaseStore(store).load("p")
    code, out, err = main("sweep", letter.parent, store=store)
    assert "could be 90%" not in out and "COULD NOT PROCEED 1" in _flat(out), out + err


def test_possible_degrees_edited_onto_a_run_that_did_not_finish_are_refused(store, tmp_path):
    """A run stopped after extract: the report printed "possible final degrees 90%"."""
    import asyncio

    from recheck.graph import ExtractNode, open_case

    letter = tabular_letter(tmp_path / "docs" / "p.txt", PAIR, stated=70)
    case_store = CaseStore(store)
    open_case(case_store, "p", str(letter))
    asyncio.run(ExtractNode(case_store, "p", str(letter)).invoke_async("extract"))
    assert case_store.load("p").status == "extracted"
    _edit(store, "p", lambda raw: raw.update(possible_degrees=[90]))
    code, out, err = main("show", "--case", "p", store=store)
    assert code == EXIT_CANNOT_PROCEED and "90%" not in out, out + err


def test_a_condition_never_unknown_is_not_reported_as_unasked(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "d.txt", DISCREPANT, stated=70)
    assert main("sweep", work, store=store)[0] == EXIT_OK
    _edit(store, "d", lambda raw: raw.update(immaterial_unknowns=[1]))
    with pytest.raises(CaseCorrupt, match="unknown fact"):
        CaseStore(store).load("d")
    code, out, err = main("show", "--case", "d", store=store)
    assert code == EXIT_CANNOT_PROCEED and "nobody was asked" not in out, out + err
