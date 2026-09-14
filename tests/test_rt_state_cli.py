"""Red-team findings against durable state, the command line and the report.

Each test reproduces one reproduced finding and fails on the code before its
fix. The finding id is in the section heading.

  FILES-P4-02  a case.json edited to "complete" with a made-up figure was
               printed by show and sweep as the 38 CFR engine's result
  FILES-P4-03  a type-confused field crashed show, resume and the whole sweep
  FILES-P4-04  a case.json naming another case id wrote into that case
  HUMAN-F7     (the same, reached through resume)
  HUMAN-F1     concurrent resumes corrupted case.json: no lock, shared temp file
  HUMAN-F3     a resume killed before compute left answers on file and no way
               to finish without discarding them
  HUMAN-F4     a case "awaiting" a question its session did not hold printed a
               resume command that was always refused
  HUMAN-F5     --fresh on a case that could not be fully removed tracebacked,
               half-deleted it, and stopped the sweep
  HUMAN-F8     usage errors exited 2, the code for "waiting on an answer"
  HUMAN-F9     case ids starting with "-" printed commands that fail when pasted
  HUMAN-F10    a letter's file name was pasted into printed commands unescaped
  HUMAN-F14    resume and show never noticed the letter had changed
  HUMAN-F15    printed commands used backslash paths that bash mangles
  SECRETS-F1   --debug printed the AWS session token and access key id
  SECRETS-F4   (report wording only) a run that stopped was reported as "could
               not establish enough facts"
  SECRETS-F7   the 4.26(d) effective-date caveat was missing from --brief and
               the triage
  SECRETS-F11  the verdict named reviewer facts but not AI classifications
  SECRETS-F12  "questions asked" counted only open questions
  SECRETS-F13  question header wording; the unparsed verdict capitalised a file name
  MODEL-RT-P2P6-05  the MAP fixture classified a condition from its linked clause
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    ROOT,
    UNLISTED,
    UNLISTED_NERVE,
    case_json,
    cli,
    main,
    require_lexicon_abstains,
    tabular_letter,
)
from recheck.case import Case, CaseCorrupt, CaseStore
from recheck.graph import build_graph

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]
D426 = [("Bronchial asthma", 90), ("Cervical strain", 30), ("Right knee strain", 10), ("Left knee strain", 10)]
CASELOAD_MAP = ROOT / "fixtures" / "classifications" / "caseload.json"


@pytest.fixture
def store(tmp_path):
    return tmp_path / "runs"


def _edit(store_root: pathlib.Path, case_id: str, change) -> None:
    path = store_root / case_id / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


def _fingerprint(store_root: pathlib.Path, case_id: str) -> None:
    """The audit recorded the letter's SHA-256, as the extract node does.

    This helper used to supply the digest itself when the audit had left it
    unset (written before the extract node recorded one). It then hid the
    loss of that recording: with the extract node storing no digest, every
    letter-changed test here still passed on the digest the helper wrote.
    """
    case = CaseStore(store_root).load(case_id)
    assert case.document_sha256 == hashlib.sha256(pathlib.Path(case.source_path).read_bytes()).hexdigest(), \
        "the audit must record the letter's fingerprint"
    assert case_json(store_root, case_id).get("document_sha256"), "the fingerprint must persist in case.json"


def _commands(text: str, verb: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("recheck") and f" {verb} " in line]


# ==========================================================================
# FILES-P4-02: a result on file is re-derived before anything reports it
# ==========================================================================

def _forge_complete(raw: dict) -> None:
    raw.update(status="complete", recomputed_degree=90, recomputed_combined=90, bilateral_applied=True)


def test_a_forged_complete_result_is_not_reported_by_show_or_sweep(store, tmp_path):
    work = tmp_path / "docs"
    work.mkdir()
    shutil.copy(LETTERS / "05_missing_side.txt", work / "05_missing_side.txt")
    shutil.copy(LETTERS / "06_agrees.txt", work / "06_agrees.txt")
    code, out, _ = main("sweep", work, store=store)
    assert code == EXIT_AWAITING_HUMAN and "70% or 80%" in out

    _edit(store, "05_missing_side", _forge_complete)
    code, out, err = main("show", "--case", "05_missing_side", "--brief", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "90%" not in out and "POTENTIAL DISCREPANCY" not in out
    assert "do not give" in err or "could change its result" in err

    code, out, err = main("sweep", work, store=store)
    assert "recomputed 90%" not in out
    assert "Traceback" not in out + err
    assert code != EXIT_OK


def test_a_complete_case_whose_figure_is_edited_is_refused(store, tmp_path):
    letter = tabular_letter(tmp_path / "agrees.txt", [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=60)
    assert main("audit", letter, "--case", "c", store=store)[0] == EXIT_OK
    CaseStore(store).load("c")  # the genuine result loads

    _edit(store, "c", lambda raw: raw.update(recomputed_degree=70))
    with pytest.raises(CaseCorrupt, match="do not give"):
        CaseStore(store).load("c")


def test_a_figure_on_a_case_that_is_not_complete_is_refused(store, tmp_path):
    """The verdict printed a recomputed figure for any status that was not
    awaiting, undetermined or unparsed - "ready" with a made-up 90% included."""
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "c", store=store)[0] == EXIT_AWAITING_HUMAN
    _edit(store, "c", lambda raw: raw.update(status="ready", recomputed_degree=90, recomputed_combined=90))
    with pytest.raises(CaseCorrupt, match="carries a recomputed result"):
        CaseStore(store).load("c")


def test_the_4_26_d_caveat_cannot_be_edited_out_of_a_result(store, tmp_path):
    letter = tabular_letter(tmp_path / "d.txt", D426, stated=90)
    assert main("audit", letter, "--case", "d", store=store)[0] == EXIT_OK
    _edit(store, "d", lambda raw: raw.update(
        trace=[e for e in raw["trace"] if e["action"] != "38 CFR 4.26(d) decides this result"]))
    with pytest.raises(CaseCorrupt, match="4.26"):
        CaseStore(store).load("d")


# ==========================================================================
# FILES-P4-03: field types are validated; one bad case never costs the triage
# ==========================================================================

TYPE_TAMPERS = {
    "possible_degrees int": lambda raw: raw.update(possible_degrees=5),
    "possible_degrees of text": lambda raw: raw.update(possible_degrees=["70"]),
    "condition int": lambda raw: raw["decisions"][0].update(condition=5),
    "condition null": lambda raw: raw["decisions"][0].update(condition=None),
    "source_path int": lambda raw: raw.update(source_path=5),
    "source_path null": lambda raw: raw.update(source_path=None),
    "classifier int": lambda raw: raw.update(classifier=5),
    "timeline entry without pid": lambda raw: raw.update(timeline=[{"node": "extract"}]),
    "timeline text": lambda raw: raw.update(timeline="abc"),
    "timeline pid bool": lambda raw: raw.update(timeline=[{"node": "extract", "pid": True}]),
    "trace detail int": lambda raw: raw["trace"][0].update(detail=5),
    "immaterial_unknowns int": lambda raw: raw.update(immaterial_unknowns=5),
    "immaterial_unknowns out of range": lambda raw: raw.update(immaterial_unknowns=[99]),
    "human_answers list": lambda raw: raw.update(human_answers=["2=left"]),
    "rejected_answer int": lambda raw: raw.update(rejected_answer=5),
    "undetermined_reason list": lambda raw: raw.update(undetermined_reason=["x"]),
    "bilateral_applied text": lambda raw: raw.update(bilateral_applied="yes"),
    "document_sha256 not hex": lambda raw: raw.update(document_sha256="not-a-digest"),
}


@pytest.mark.parametrize("name", sorted(TYPE_TAMPERS))
def test_a_type_confused_case_file_is_refused_as_corrupt(store, tmp_path, name):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "c", store=store)[0] == EXIT_AWAITING_HUMAN
    _edit(store, "c", TYPE_TAMPERS[name])
    with pytest.raises(CaseCorrupt, match="malformed content|impossible|does not have"):
        CaseStore(store).load("c")


@pytest.mark.parametrize("command", ["show", "resume"])
@pytest.mark.parametrize("name", ["condition int", "source_path null", "timeline text", "immaterial_unknowns int",
                                  "possible_degrees int"])
def test_the_cli_refuses_a_type_confused_case_without_a_traceback(store, tmp_path, command, name):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "c", store=store)[0] == EXIT_AWAITING_HUMAN
    _edit(store, "c", TYPE_TAMPERS[name])
    args = ["--case", "c"] + (["--answer", "2=left"] if command == "resume" else [])
    code, out, err = main(command, *args, store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "malformed content" in err
    assert case_json(store, "c")["recomputed_degree"] is None


def test_one_case_that_cannot_be_rendered_does_not_cost_the_whole_triage(store, tmp_path, monkeypatch):
    from recheck.sweep import Outcome, render_triage

    work = tmp_path / "docs"
    work.mkdir()
    shutil.copy(LETTERS / "01_tabular.txt", work / "01_tabular.txt")
    shutil.copy(LETTERS / "05_missing_side.txt", work / "05_missing_side.txt")
    main("sweep", work, store=store)

    real_load = CaseStore.load

    def load(self, case_id):
        if case_id == "05_missing_side":
            raise TypeError("'int' object is not iterable")  # a defect no validation anticipated
        return real_load(self, case_id)

    monkeypatch.setattr(CaseStore, "load", load)
    text = render_triage(CaseStore(store), [Outcome("01_tabular", Outcome.COMPLETE),
                                            Outcome("05_missing_side", Outcome.AWAITING_HUMAN)])
    assert "CASELOAD TRIAGE" in text
    assert re.search(r"05_missing_side\s+case unreadable: 'int' object is not iterable", text)
    assert "2 documents:" in text and "1 could not proceed" in text


# ==========================================================================
# FILES-P4-04 / HUMAN-F7: a case file cannot steer writes into another case
# ==========================================================================

def test_a_case_file_naming_another_case_cannot_overwrite_it(store, tmp_path):
    assert main("audit", LETTERS / "06_agrees.txt", "--case", "A", store=store)[0] == EXIT_OK
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "B", store=store)[0] == EXIT_AWAITING_HUMAN
    victim = (store / "A" / "case.json").read_bytes()

    _edit(store, "B", lambda raw: raw.update(case_id="A"))
    code, _, err = main("resume", "--case", "B", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "names a different case id 'A'" in err
    assert (store / "A" / "case.json").read_bytes() == victim
    code, out, _ = main("show", "--case", "A", "--brief", store=store)
    assert code == EXIT_OK and "NO DISCREPANCY FOUND" in out


# ==========================================================================
# HUMAN-F1: one process per case; saves never share a temporary file
# ==========================================================================

def test_a_resume_on_a_case_another_process_holds_is_refused_and_changes_nothing(store, tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "c", store=store)[0] == EXIT_AWAITING_HUMAN
    before = (store / "c" / "case.json").read_bytes()

    with CaseStore(store).lock("c"):
        result = cli("resume", "--case", "c", "--answer", "2=left", store=store)  # a separate process
        assert result.returncode == EXIT_CANNOT_PROCEED, result.stdout + result.stderr
        assert "busy in another process" in result.stderr
        code, _, err = main("audit", letter, "--case", "c", "--fresh", store=store)
        assert code == EXIT_CANNOT_PROCEED and "busy" in err
        code, out, _ = main("sweep", letter.parent, store=tmp_path / "unused")  # a different store is not blocked
        assert code == EXIT_AWAITING_HUMAN
    assert (store / "c" / "case.json").read_bytes() == before

    assert main("resume", "--case", "c", "--answer", "2=left", store=store)[0] == EXIT_OK
    assert sorted(p.name for p in store.iterdir()) == ["c"], "the lock file is removed on release"


def test_a_sweep_leaves_a_busy_case_alone(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "pair.txt", PAIR, stated=70)
    shutil.copy(LETTERS / "06_agrees.txt", work / "agrees.txt")
    main("sweep", work, store=store)
    before = (store / "pair" / "case.json").read_bytes()
    with CaseStore(store).lock("pair"):
        code, out, _ = main("sweep", work, "--fresh", store=store)
    assert re.search(r"pair\s+case pair is busy in another process", out)
    assert (store / "pair" / "case.json").read_bytes() == before
    assert CaseStore(store).load("agrees").status == "complete"


def test_each_save_writes_through_its_own_temporary_file(tmp_path, monkeypatch):
    """Two writers of 'case.json.tmp' interleaved into invalid JSON."""
    store = CaseStore(tmp_path / "runs")
    sources = []
    real_replace = os.replace

    def recording_replace(src, dst):
        sources.append(pathlib.Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    store.save(Case("c", "x.txt"))
    store.save(Case("c", "x.txt"))
    assert len(sources) == 2 and sources[0] != sources[1]
    assert "case.json.tmp" not in sources


def test_concurrent_writers_and_readers_never_see_a_broken_case(tmp_path):
    store = CaseStore(tmp_path / "runs")
    store.save(Case("c", "x.txt", status="open"))
    errors: list[BaseException] = []

    def writer(tag: str):
        try:
            for i in range(40):
                store.save(Case("c", f"{tag}-{i}.txt", status="open"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(80):
                store.load("c")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in "ab"] + [threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.load("c").source_path.endswith("-39.txt")
    assert sorted(p.name for p in (tmp_path / "runs" / "c").iterdir()) == ["case.json"]


def test_a_case_that_cannot_be_read_back_after_a_run_exits_3_without_a_traceback(store, tmp_path):
    from types import SimpleNamespace

    from recheck.cli import _finish
    from recheck.sweep import Outcome

    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    main("audit", letter, "--case", "c", store=store)
    (store / "c" / "case.json").write_text("{half a case", encoding="utf-8")
    args = SimpleNamespace(store=str(store), brief=True)
    for state in (Outcome.COMPLETE, Outcome.AWAITING_HUMAN, Outcome.FAILED):
        assert _finish(args, CaseStore(store), Outcome("c", state, "x")) == EXIT_CANNOT_PROCEED


# ==========================================================================
# HUMAN-F3: a case stopped after its answers were accepted can be finished
# ==========================================================================

def test_a_resume_killed_before_compute_is_finished_from_the_answers_on_file(store, tmp_path, monkeypatch):
    from recheck.graph import ComputeNode

    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "k", store=store)[0] == EXIT_AWAITING_HUMAN

    async def killed(self, *args, **kwargs):
        raise RuntimeError("process killed before compute")

    with monkeypatch.context() as patch:
        patch.setattr(ComputeNode, "invoke_async", killed)
        code, _, err = main("resume", "--case", "k", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED
    stranded = case_json(store, "k")
    assert stranded["status"] == "ready" and stranded["human_answers"] == {"2": "lower-left"}
    assert stranded["recomputed_degree"] is None

    # Answering again is not silently ignored: the answers on file are what the result rests on.
    before = (store / "k" / "case.json").read_bytes()
    code, _, err = main("resume", "--case", "k", "--answer", "2=right", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "without --answer" in err and "--fresh" in err
    assert (store / "k" / "case.json").read_bytes() == before

    code, out, err = main("resume", "--case", "k", store=store)
    assert code == EXIT_OK, out + err
    case = case_json(store, "k")
    assert case["status"] == "complete" and case["recomputed_degree"] == 80
    assert case["human_answers"] == {"2": "lower-left"}
    assert case["timeline"][-1] == {"node": "compute", "pid": os.getpid()}
    assert "POTENTIAL DISCREPANCY" in out and "the facts you supplied for [2]" in out


def test_a_ready_case_whose_facts_do_not_settle_the_result_is_not_computed(store, tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "c", store=store)[0] == EXIT_AWAITING_HUMAN
    _edit(store, "c", lambda raw: raw.update(status="ready"))
    code, out, err = main("resume", "--case", "c", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "could change its result" in err
    assert case_json(store, "c")["recomputed_degree"] is None


def test_the_triage_and_report_say_how_to_finish_a_ready_case(store):
    from recheck.report import verdict
    from recheck.sweep import outcome_from_case

    case = Case("r", "x.txt", status="ready", human_answers={"2": "lower-left"})
    headline, explanation = verdict(case)
    assert headline.startswith("RUN DID NOT FINISH")
    assert "resume" in explanation and "without an answer" in explanation
    CaseStore(store).save(case)
    assert "run resume to finish" in outcome_from_case(CaseStore(store), "r", reused=True).detail


# ==========================================================================
# HUMAN-F4: a question the session does not hold is not offered
# ==========================================================================

@pytest.mark.parametrize("damage", ["session deleted", "session without the question"])
def test_a_question_missing_from_the_session_is_not_offered(store, tmp_path, damage):
    work = tmp_path / "docs"
    letter = tabular_letter(work / "pair.txt", PAIR, stated=70)
    assert main("sweep", work, store=store)[0] == EXIT_AWAITING_HUMAN
    shutil.rmtree(store / "pair" / "session")
    if damage == "session without the question":
        build_graph(CaseStore(store), "pair", str(letter), None)  # writes an empty session
        assert (store / "pair" / "session").is_dir()

    code, out, _ = main("sweep", work, store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert not _commands(out, "resume"), "no resume command for a question that cannot be answered"
    assert re.search(r"pair\s+question not in its Strands session; re-audit with --fresh", out)

    code, out, err = main("show", "--case", "pair", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert not _commands(out, "resume")
    assert "QUESTION FOR THE REVIEWER" not in out
    assert "--fresh" in err


# ==========================================================================
# HUMAN-F5: discarding a case never half-deletes it or stops a sweep
# ==========================================================================

def _two_letter_sweep(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "pair.txt", PAIR, stated=70)
    shutil.copy(LETTERS / "06_agrees.txt", work / "unlisted.txt")
    assert main("sweep", work, store=store)[0] == EXIT_AWAITING_HUMAN
    return work


def test_a_file_that_cannot_be_deleted_does_not_stop_a_fresh_audit_or_sweep(store, tmp_path, monkeypatch):
    work = _two_letter_sweep(store, tmp_path)
    real_unlink = os.unlink

    def locked_case_file(path, *args, **kwargs):
        if os.path.basename(path) == "case.json":
            raise PermissionError(13, "Access is denied", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", locked_case_file)
    code, out, err = main("sweep", work, "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert "2 documents:" in out and "0 could not proceed" in out

    code, out, err = main("audit", work / "pair.txt", "--case", "pair", "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    case = CaseStore(store).load("pair")
    assert case.status == "awaiting_human" and (store / "pair" / "session").is_dir()


def test_a_read_only_case_file_is_discarded_cleanly(store, tmp_path):
    """The reproduction on Windows: attrib +R case.json, then --fresh."""
    work = _two_letter_sweep(store, tmp_path)
    os.chmod(store / "pair" / "case.json", stat.S_IREAD)
    code, out, err = main("audit", work / "pair.txt", "--case", "pair", "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert sorted(p.name for p in store.iterdir()) == ["pair", "unlisted"]


def test_a_case_that_cannot_be_moved_aside_is_left_whole(store, tmp_path, monkeypatch):
    work = _two_letter_sweep(store, tmp_path)
    before = (store / "pair" / "case.json").read_bytes()
    real_replace = os.replace

    def held_directory(src, dst, *args, **kwargs):
        if pathlib.Path(src).is_dir():
            raise PermissionError(32, "The process cannot access the file", str(src))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", held_directory)
    code, out, err = main("audit", work / "pair.txt", "--case", "pair", "--fresh", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "could not discard case pair" in err and "Traceback" not in err
    assert (store / "pair" / "case.json").read_bytes() == before
    assert (store / "pair" / "session").is_dir()

    code, out, _ = main("sweep", work, "--fresh", store=store)
    assert re.search(r"pair\s+could not discard the case on file", out)
    assert "2 documents:" in out


# ==========================================================================
# HUMAN-F8: usage errors do not exit with the "waiting on an answer" code
# ==========================================================================

@pytest.mark.parametrize(
    "argv",
    [[], ["resume"], ["resume", "--case", "c1", "--answer", "-2=left"], ["resume", "--case", "-x", "--answer", "2=left"],
     ["audit", "--case", "x"], ["show"], ["sweep"], ["frobnicate"]],
)
def test_a_usage_error_exits_3(argv, capsys):
    from recheck.cli import main as cli_main

    with pytest.raises(SystemExit) as info:
        cli_main(argv)
    assert info.value.code == EXIT_CANNOT_PROCEED
    assert "error:" in capsys.readouterr().err


def test_a_usage_error_exits_3_in_a_real_process(store):
    result = cli("resume", store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED


# ==========================================================================
# HUMAN-F9: no case id starts with "-"
# ==========================================================================

def test_a_case_id_may_not_start_with_a_dash(store):
    with pytest.raises(ValueError, match="may not start with '-'"):
        CaseStore(store).path_for("-x")
    code, _, err = main("audit", LETTERS / "06_agrees.txt", "--case=-x", store=store)
    assert code == EXIT_CANNOT_PROCEED and "may not start with '-'" in err


def test_commands_printed_for_dash_named_letters_parse_when_pasted(store, tmp_path):
    from recheck.cli import build_parser
    from recheck.sweep import case_id_for

    assert case_id_for(pathlib.Path("--help.txt")) == "case_--help"
    work = tmp_path / "dash"
    tabular_letter(work / "--help.txt", PAIR, stated=70)
    tabular_letter(work / "-v.txt", PAIR, stated=70)
    code, out, _ = main("sweep", work, store=store)
    assert code == EXIT_AWAITING_HUMAN
    commands = _commands(out, "resume")
    assert len(commands) == 2
    for command in commands:
        argv = shlex.split(command)[1:]
        argv[argv.index("--answer") + 1] = "2=left"
        args = build_parser().parse_args(argv)
        assert CaseStore(args.store).exists(args.case)


# ==========================================================================
# HUMAN-F10 / HUMAN-F15: printed commands are safe to paste, in any shell
# ==========================================================================

@pytest.mark.parametrize(
    "value,token",
    [("runs", "runs"), ("fixtures/letters/06_agrees.txt", "fixtures/letters/06_agrees.txt"),
     ("C:/Users/Jane Doe/runs", '"C:/Users/Jane Doe/runs"'), ("-v.txt", "./-v.txt"),
     ("rt/inj/a$(touch PWNED).txt", "LETTER"), ("a`id`.txt", "LETTER"), ("rt/R&D", "LETTER"),
     ("it's.txt", "LETTER"), ('say "hi".txt', "LETTER"), ("100%.txt", "LETTER"), ("a;b.txt", "LETTER")],
)
def test_a_path_is_printed_only_when_no_shell_would_interpret_it(value, token):
    from recheck.cli import _shell_path

    printed, note = _shell_path(value, "LETTER")
    assert printed == token
    assert bool(note) == (token == "LETTER")


@pytest.mark.skipif(os.name != "nt", reason="backslash is a path separator only on Windows")
def test_a_windows_path_is_printed_with_forward_slashes():
    from recheck.cli import _shell_path

    assert _shell_path("fixtures\\letters\\06_agrees.txt", "LETTER") == ("fixtures/letters/06_agrees.txt", "")


def test_a_letter_named_with_shell_syntax_never_reaches_a_printed_command(store, tmp_path):
    work = tmp_path / "inj"
    work.mkdir()
    hostile = work / "a$(touch PWNED).txt"
    shutil.copy(LETTERS / "06_agrees.txt", hostile)
    assert main("sweep", work, store=store)[0] == EXIT_OK
    (case_id,) = [p.name for p in store.iterdir()]

    code, _, err = main("resume", "--case", case_id, "--answer", "0=left", store=store)
    assert code == EXIT_CANNOT_PROCEED
    command = re.search(r"`(recheck [^`]*)`", err).group(1)
    assert "$(" not in command and "(" not in command  # the case id is sanitised; the path is left out
    assert re.search(r"audit LETTER --case \S+ --fresh", command)
    assert "LETTER stands for" in err and "touch PWNED" in err  # named, outside the command


def test_a_hostile_store_path_is_replaced_by_a_placeholder(tmp_path):
    store = tmp_path / "R&D $(id)"
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    code, out, _ = main("audit", letter, "--case", "c", store=store)
    assert code == EXIT_AWAITING_HUMAN
    (command,) = _commands(out, "resume")
    assert command.startswith("recheck --store STORE resume --case c ")
    assert "STORE stands for" in out


def test_a_printed_audit_command_names_the_letter_so_a_posix_shell_finds_it(store):
    """HUMAN-F15: fixtures\\letters\\06_agrees.txt pasted into bash became fixturesletters06_agrees.txt."""
    letter = LETTERS / "06_agrees.txt"
    assert main("audit", letter, "--case", "s", store=store)[0] == EXIT_OK
    code, _, err = main("resume", "--case", "s", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED
    command = re.search(r"`(recheck [^`]*)`", err).group(1)
    argv = shlex.split(command)  # POSIX shell word splitting and quote removal
    assert pathlib.Path(argv[argv.index("audit") + 1]).is_file()
    assert "\\" not in command


# ==========================================================================
# HUMAN-F14: a changed letter is noticed
# ==========================================================================

def test_resume_refuses_when_the_letter_has_changed_since_the_audit(store, tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "m", store=store)[0] == EXIT_AWAITING_HUMAN
    _fingerprint(store, "m")
    before = (store / "m" / "case.json").read_bytes()
    tabular_letter(letter, [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                            ("Left knee limitation of motion", 10), ("Tinnitus", 10)], stated=80)

    code, out, err = main("resume", "--case", "m", "--answer", "2=right", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "has changed since it was audited" in err and "--fresh" in err
    assert "NO DISCREPANCY FOUND" not in out
    assert (store / "m" / "case.json").read_bytes() == before

    code, out, err = main("show", "--case", "m", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "THE LETTER HAS CHANGED" in out
    assert not _commands(out, "resume")


def test_a_sweep_does_not_report_a_stale_case_for_a_changed_letter(store, tmp_path):
    work = tmp_path / "docs"
    letter = tabular_letter(work / "agrees.txt", [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=60)
    assert main("sweep", work, store=store)[0] == EXIT_OK
    _fingerprint(store, "agrees")
    tabular_letter(letter, [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=70)
    code, out, _ = main("sweep", work, store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert re.search(r"agrees\s+letter changed since it was audited; re-audit with --fresh", out)
    # --fresh re-audits the letter as it is now: 60% recomputed against the 70% it now states.
    assert main("sweep", work, "--fresh", store=store)[0] == EXIT_OK
    assert CaseStore(store).load("agrees").stated_combined == 70


def test_an_unchanged_or_deleted_letter_does_not_block_resume(store, tmp_path):
    letter = tabular_letter(tmp_path / "gone" / "pair.txt", PAIR, stated=70)
    for case_id in ("same", "gone"):
        assert main("audit", letter, "--case", case_id, store=store)[0] == EXIT_AWAITING_HUMAN
        _fingerprint(store, case_id)
    assert main("resume", "--case", "same", "--answer", "2=left", store=store)[0] == EXIT_OK
    shutil.rmtree(tmp_path / "gone")
    assert main("resume", "--case", "gone", "--answer", "2=left", store=store)[0] == EXIT_OK


# ==========================================================================
# SECRETS-F1: --debug never prints credential material
# ==========================================================================

_GUARD = '''
import socket
_LOOPBACK = ("127.0.0.1", "::1", "localhost")
_connect, _connect_ex, _getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
def _host(address):
    return address[0] if isinstance(address, tuple) else address
def connect(self, address):
    if _host(address) not in _LOOPBACK:
        raise OSError("network blocked by the test guard")
    return _connect(self, address)
def connect_ex(self, address):
    if _host(address) not in _LOOPBACK:
        raise OSError("network blocked by the test guard")
    return _connect_ex(self, address)
def getaddrinfo(host, *args, **kwargs):
    if host is not None and host not in _LOOPBACK:
        raise OSError("network blocked by the test guard")
    return _getaddrinfo(host, *args, **kwargs)
socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo
'''


def test_debug_logging_never_prints_aws_credentials(tmp_path):
    """Planted credentials, a letter that needs the classifier, --model bedrock,
    and every connection off the machine refused by a socket guard. The
    request is still signed and logged before the guard refuses it."""
    pytest.importorskip("boto3")
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(_GUARD, encoding="utf-8")
    require_lexicon_abstains(UNLISTED)
    letter = tabular_letter(tmp_path / "letter.txt", [("Post-traumatic stress disorder", 60),
                            ("Right knee strain", 20), (UNLISTED, 10)], stated=70)
    token, key_id = "PLANTEDSESSIONTOKENzzzzqqqq", "AKIAPLANTEDTESTKEY01"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "RECHECK_"))}
    env.update(PYTHONPATH=os.pathsep.join([str(guard), str(ROOT / "src")]), PYTHONIOENCODING="utf-8",
               AWS_ACCESS_KEY_ID=key_id, AWS_SECRET_ACCESS_KEY="plantedSECRETvalue", AWS_SESSION_TOKEN=token,
               AWS_REGION="us-west-2", AWS_EC2_METADATA_DISABLED="true",
               AWS_CONFIG_FILE=str(tmp_path / "none"), AWS_SHARED_CREDENTIALS_FILE=str(tmp_path / "none"))
    result = subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--debug", "--store", str(tmp_path / "runs"), "audit", str(letter),
         "--case", "f1", "--model", "bedrock"],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ROOT), timeout=300,
    )
    combined = result.stdout + result.stderr
    assert "DEBUG strands" in combined, "precondition: --debug output is on"
    assert "bedrock-runtime" in combined, "precondition: the model call was attempted"
    assert token not in combined
    assert key_id not in combined


def test_the_redaction_filter_masks_credential_shaped_values():
    from recheck.cli import RedactCredentials

    record = logging.LogRecord(
        "some.logger", logging.DEBUG, __file__, 1,
        "headers=%s", ({"X-Amz-Security-Token": b"TOKENVALUE123", "Authorization":
                        b"AWS4-HMAC-SHA256 Credential=AKIAABCDEFGHIJKLMNOP/20260913/us-west-2", "x-api-key":
                        "sk-ant-abcdef"},), None,
    )
    RedactCredentials().filter(record)
    text = record.getMessage()
    for secret in ("TOKENVALUE123", "AKIAABCDEFGHIJKLMNOP", "sk-ant-abcdef"):
        assert secret not in text


# ==========================================================================
# SECRETS-F4 (report wording): a run that stopped is not blamed on the facts
# ==========================================================================

@pytest.mark.parametrize("status", ["open", "extracted", "classified"])
def test_a_run_that_stopped_is_reported_as_unfinished(status):
    from recheck.report import render, verdict

    headline, explanation = verdict(Case("s", "x.txt", status=status))
    assert headline == "RUN DID NOT FINISH - NO RECOMPUTATION"
    assert "could not establish enough facts" not in explanation
    # Only a complete case reports a figure, whatever else the file says.
    text = render(Case("s", "x.txt", status=status, stated_combined=70, recomputed_degree=90))
    assert "90%" not in text


# ==========================================================================
# SECRETS-F7: the 4.26(d) caveat reaches --brief and the triage
# ==========================================================================

def test_the_4_26_d_caveat_is_in_the_brief_report_and_the_triage(store, tmp_path):
    work = tmp_path / "docs"
    letter = tabular_letter(work / "d426.txt", D426, stated=90)
    code, out, _ = main("audit", letter, "--case", "d426", "--brief", store=store)
    assert code == EXIT_OK
    assert "DECISION TRACE" not in out
    verdict_text = " ".join(out.split())
    assert "38 CFR 4.26(d) exception, in force from April 16, 2023" in verdict_text
    assert "the prior rule gives 90%" in verdict_text

    code, out, _ = main("sweep", work, store=tmp_path / "swept")
    assert re.search(r"d426\s+stated 90%\s+recomputed 100%\s+\(\+10, 4.26\(d\) exception - check decision period\)",
                     out)
    assert "4.26 not applied" not in out


# ==========================================================================
# SECRETS-F11: the verdict names the AI classifications it rests on
# ==========================================================================

def test_a_verdict_that_rests_on_ai_groups_says_so(store, tmp_path):
    listed = f"{UNLISTED_NERVE}, left"
    require_lexicon_abstains(listed)
    fixture = tmp_path / "map.json"
    fixture.write_text(json.dumps({"anatomy": {"zorblatt nerve": "lower"}, "confidence": 0.9}), encoding="utf-8")
    work = tmp_path / "docs"
    letter = tabular_letter(work / "ai.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                                              (listed, 10), ("Tinnitus", 10)], stated=70)
    code, out, _ = main("audit", letter, "--case", "ai", "--brief", "--scripted", "--classifications", fixture,
                        store=store)
    assert code == EXIT_OK
    assert "the AI classification of [2] (replayed from a fixture)" in " ".join(out.split())

    code, out, _ = main("sweep", work, "--scripted", "--classifications", fixture, store=tmp_path / "swept")
    assert re.search(r"ai\s+stated 70%\s+recomputed 80%\s+\(\+10, 4.26 applied; AI groups \[2\]\)", out)


# ==========================================================================
# SECRETS-F12: questions asked, not questions still open
# ==========================================================================

def test_the_sweep_counts_questions_asked_including_answered_ones(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "pair.txt", PAIR, stated=70)
    code, out, _ = main("sweep", work, store=store)
    assert "questions asked: 1." in out
    assert main("resume", "--case", "pair", "--answer", "2=left", store=store)[0] == EXIT_OK
    code, out, _ = main("sweep", work, store=store)
    assert "questions asked: 1." in out and "0 waiting on an answer" in out


# ==========================================================================
# SECRETS-F13: wording
# ==========================================================================

def test_the_question_says_the_letter_leaves_out_facts_that_could_change_the_rating(store, tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    code, out, _ = main("audit", letter, "--case", "q", store=store)
    assert "The letter leaves out facts that could change this rating." in out
    assert "does not establish facts that change this rating" not in out


def test_an_unparsed_verdict_keeps_the_file_name_exactly_as_on_disk():
    from recheck.report import verdict

    case = Case("s", "letters/scanned.pdf", status="unparsed", trace=[{
        "actor": "DETERMINISTIC", "action": "Extraction FAILED", "detail": "scanned.pdf has no extractable text layer.",
        "value": "cannot proceed", "confidence": None, "evidence": None, "rule": None}])
    assert verdict(case)[1].startswith("scanned.pdf has no extractable text layer")
    case.trace[0]["detail"] = "no ratings found"
    assert verdict(case)[1] == "No ratings found."


# ==========================================================================
# MODEL-RT-P2P6-05: the MAP fixture matches the rated condition only
# ==========================================================================

def test_the_map_fixture_does_not_classify_a_condition_from_its_linked_clause(store, tmp_path):
    name = "Left Lisfranc injury, secondary to de Quervain tenosynovitis"
    require_lexicon_abstains(name)
    letter = tabular_letter(tmp_path / "map_linked.txt", [("Right wrist strain", 30), (name, 30)], stated=50)
    code, out, err = main("audit", letter, "--case", "linked", "--scripted", "--classifications", CASELOAD_MAP,
                          store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    decision = case_json(store, "linked")["decisions"][1]
    assert decision["extremity_group"] == "unknown" and decision["group_by"] is None
    assert "POTENTIAL DISCREPANCY" not in out
