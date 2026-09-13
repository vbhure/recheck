"""Interrupt in one process, answer in another - proven with real OS processes.

Calling the graph twice inside one interpreter proves nothing about
durability: module state, caches and the graph object all survive. So the
tests that make a durability claim shell out to a SEPARATE python and
communicate only through the filesystem.

Persistence has exactly one owner for graph state - the Strands
FileSessionManager under <store>/<case>/session - and case.json holds the
domain facts. The decisive tests delete one or the other:

  - delete the session and resume must REFUSE (no outstanding interrupt),
    leaving case.json byte-identical. A resume that still worked would be
    inheriting state from somewhere other than the session.
  - delete everything except the session and case.json - the letter
    included - and resume must still WORK.

Defects regression-locked here:

  P1  Two copies of graph state. 0d018c9 kept a hand-written
      graph_state.json beside the Strands session; the two could disagree,
      and a corrupted copy resumed into a run that computed nothing and
      exited 0. There is now no graph_state.json at all.
  P2  Unknown treated as "no". At 0d018c9 a letter with a term outside the
      lexicon and no classifier asked only for a side; answering "unknown"
      reported NO DISCREPANCY FOUND. The extremity group is now asked for,
      and "unknown" leaves the case UNDETERMINED with both possible degrees.
  P3  Re-running audit on an interrupted case restored its Strands session
      and died with "TypeError: ... must resume from interrupt with list of
      interruptResponse's". audit now refuses an existing case unless
      --fresh, which discards the case and its session first.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    UNLISTED,
    case_json,
    cli,
    main,
    require_lexicon_abstains,
    tabular_letter,
)

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]


@pytest.fixture
def store(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def letter(tmp_path):
    return tabular_letter(tmp_path / "letters" / "pair.txt", PAIR, stated=70)


def _resume_command(stdout: str) -> list[str]:
    """The runnable command the question prints, as argv after `recheck`."""
    (line,) = [l.strip() for l in stdout.splitlines() if l.strip().startswith("recheck") and " resume " in l]
    match = re.fullmatch(r'recheck --store (\S+) resume --case (\S+) --answer "([^"]+)"', line)
    assert match, line
    return ["resume", "--case", match.group(2), "--answer", match.group(3)]


# --------------------------------------------------------------------------
# Process A: ask, persist, exit
# --------------------------------------------------------------------------

def test_process_a_asks_only_what_matters_persists_and_exits(store, letter):
    result = cli("audit", letter, "--case", "c1", store=store)
    assert result.returncode == EXIT_AWAITING_HUMAN, result.stdout + result.stderr
    out = result.stdout

    assert "QUESTION FOR THE REVIEWER - case c1" in out
    assert "could be 70% or 80%" in out and "The letter states 70%" in out
    assert "[2] Limitation of motion of the knee" in out
    assert "not established: side" in out
    assert "if lower/left: 80%" in out and "if lower/right: 70%" in out
    assert "answer with one of: left, right, both, unknown" in out
    assert "[1] Right knee strain" not in out, "a condition whose side the letter states is not asked"
    assert "never asks for a percentage" in out
    assert _resume_command(out) == ["resume", "--case", "c1", "--answer", "2=<left|right|both|unknown>"]
    for noise in ("Tool #", "Traceback", "node_id="):
        assert noise not in out + result.stderr

    case = case_json(store, "c1")
    assert case["status"] == "awaiting_human"
    assert case["recomputed_degree"] is None
    assert (store / "c1" / "session").is_dir()
    assert not list((store / "c1").rglob("graph_state.json")), "P1: one owner for graph state"


# --------------------------------------------------------------------------
# Process B: answer
# --------------------------------------------------------------------------

def test_resume_completes_in_a_separate_process_and_the_timeline_shows_both(store, letter):
    first = cli("audit", letter, "--case", "c2", store=store)
    assert first.returncode == EXIT_AWAITING_HUMAN
    pid_a = int(re.search(r"Process (\d+) is exiting", first.stdout).group(1))

    argv = _resume_command(first.stdout)
    argv[-1] = "2=left"
    second = cli(*argv, store=store)
    assert second.returncode == EXIT_OK, second.stdout + second.stderr

    case = case_json(store, "c2")
    assert case["status"] == "complete"
    assert case["recomputed_degree"] == 80 and case["stated_combined"] == 70
    assert case["bilateral_applied"] is True
    assert case["human_answers"] == {"2": "lower-left"}
    assert "POTENTIAL DISCREPANCY" in second.stdout

    timeline = case["timeline"]
    pids = {step["pid"] for step in timeline}
    assert len(pids) == 2, timeline
    assert [s["node"] for s in timeline if s["pid"] == pid_a] == ["extract", "classify", "assess"]
    (pid_b,) = pids - {pid_a}
    assert [s["node"] for s in timeline if s["pid"] == pid_b] == ["assess", "compute"]
    assert f"[process {pid_a}]" in second.stdout and f"[process {pid_b}]" in second.stdout

    # A finished case is not resumable, and trying changes nothing.
    before = (store / "c2" / "case.json").read_bytes()
    third = cli("resume", "--case", "c2", "--answer", "2=right", store=store)
    assert third.returncode == EXIT_CANNOT_PROCEED
    assert "not waiting on an answer" in third.stderr
    assert (store / "c2" / "case.json").read_bytes() == before


def test_the_printed_command_survives_a_store_path_with_spaces(tmp_path, letter):
    """The question is meant to be pasted into a terminal. An unquoted store
    under a profile like "C:/Users/Jane Doe" split into two arguments."""
    store = tmp_path / "case store with spaces"
    code, out, _ = main("audit", letter, "--case", "c9", store=store)
    assert code == EXIT_AWAITING_HUMAN
    (line,) = [l.strip() for l in out.splitlines() if l.strip().startswith("recheck") and " resume " in l]
    match = re.fullmatch(r'recheck --store "([^"]+)" resume --case (\S+) --answer "([^"]+)"', line)
    assert match, line
    assert match.group(1) == store.as_posix()  # forward slashes paste into any shell

    result = cli("resume", "--case", match.group(2), "--answer", "2=left", store=match.group(1))
    assert result.returncode == EXIT_OK, result.stdout + result.stderr


def test_resume_refuses_when_the_strands_session_is_gone(store, letter):
    assert cli("audit", letter, "--case", "c3", store=store).returncode == EXIT_AWAITING_HUMAN
    shutil.rmtree(store / "c3" / "session")
    before = (store / "c3" / "case.json").read_bytes()

    result = cli("resume", "--case", "c3", "--answer", "2=left", store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "holds no open question" in result.stderr
    assert (store / "c3" / "case.json").read_bytes() == before


def test_the_session_and_case_file_are_all_a_resume_needs(store, tmp_path):
    """Even the letter is gone: nothing but the persisted state is read."""
    source = tabular_letter(tmp_path / "gone" / "pair.txt", PAIR, stated=70)
    assert cli("audit", source, "--case", "c4", store=store).returncode == EXIT_AWAITING_HUMAN
    shutil.rmtree(tmp_path / "gone")
    (store / "c4" / "stray.tmp").write_text("x", encoding="utf-8")
    (store / "unrelated").mkdir()
    for path in list((store / "c4").iterdir()) + [store / "unrelated"]:
        if path.name not in ("session", "case.json"):
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    result = cli("resume", "--case", "c4", "--answer", "2=left", store=store)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert case_json(store, "c4")["recomputed_degree"] == 80


def test_a_rejected_answer_keeps_the_question_open_across_processes(store, letter):
    assert cli("audit", letter, "--case", "c5", store=store).returncode == EXIT_AWAITING_HUMAN

    rejected = cli("resume", "--case", "c5", "--answer", "1=left,2=left", store=store)
    assert rejected.returncode == EXIT_CANNOT_PROCEED
    assert "Your previous answer was not accepted" in rejected.stdout
    assert "condition 1 was not asked about" in rejected.stdout
    case = case_json(store, "c5")
    assert case["status"] == "awaiting_human" and case["recomputed_degree"] is None

    accepted = cli("resume", "--case", "c5", "--answer", "2=left", store=store)
    assert accepted.returncode == EXIT_OK, accepted.stdout + accepted.stderr
    assert case_json(store, "c5")["recomputed_degree"] == 80
    assert len({step["pid"] for step in case_json(store, "c5")["timeline"]}) == 3


def test_an_unknown_term_without_a_classifier_is_never_an_all_clear(store, tmp_path):
    """P2. Stated 70%; the unlisted condition would make it 80% if it were a
    left leg. The old answer to "unknown" was NO DISCREPANCY FOUND."""
    require_lexicon_abstains(UNLISTED)
    source = tabular_letter(tmp_path / "unknown.txt", [("Post-traumatic stress disorder", 60),
                            ("Right knee strain", 20), (UNLISTED, 10), ("Tinnitus", 10)], stated=70)
    first = cli("audit", source, "--case", "c6", store=store)
    assert first.returncode == EXIT_AWAITING_HUMAN, first.stdout + first.stderr
    assert "no classifier configured" in first.stdout
    assert f"[2] {UNLISTED}" in first.stdout
    assert "not established: extremity group and side" in first.stdout
    assert "NO DISCREPANCY FOUND" not in first.stdout

    second = cli("resume", "--case", "c6", "--answer", "2=unknown", store=store)
    assert second.returncode == EXIT_CANNOT_PROCEED
    assert "UNDETERMINED - NOT COMPUTED" in second.stdout
    assert "70% or 80%" in second.stdout
    assert "NO DISCREPANCY FOUND" not in second.stdout
    case = case_json(store, "c6")
    assert case["status"] == "undetermined"
    assert case["possible_degrees"] == [70, 80]
    assert case["recomputed_degree"] is None
    assert "compute" not in [step["node"] for step in case["timeline"]]


# --------------------------------------------------------------------------
# Refusals that need no second process (run in-process for speed)
# --------------------------------------------------------------------------

def test_audit_refuses_an_existing_case_unless_fresh(store, letter):
    """P3."""
    assert main("audit", letter, "--case", "c7", store=store)[0] == EXIT_AWAITING_HUMAN
    before = (store / "c7" / "case.json").read_bytes()

    code, _, err = main("audit", letter, "--case", "c7", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "already exists" in err and "--fresh" in err
    assert (store / "c7" / "case.json").read_bytes() == before

    # The exact shape that raised the TypeError: the old session is interrupted.
    code, out, err = main("audit", letter, "--case", "c7", "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert "TypeError" not in out + err
    case = case_json(store, "c7")
    assert case["status"] == "awaiting_human"
    assert [step["node"] for step in case["timeline"]] == ["extract", "classify", "assess"], \
        "a fresh audit starts a fresh timeline and session"

    code, out, _ = main("audit", LETTERS / "06_agrees.txt", "--case", "c7", "--fresh", store=store)
    assert code == EXIT_OK, out
    case = case_json(store, "c7")
    assert case["status"] == "complete" and case["source_path"].endswith("06_agrees.txt")


def test_resume_of_an_unknown_case_fails_cleanly(store):
    code, _, err = main("resume", "--case", "never-existed", "--answer", "0=left", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "no such case" in err


@pytest.mark.parametrize(
    "tamper,fragment",
    [
        (lambda raw: "{not json", "not valid JSON"),
        (lambda raw: json.dumps({**raw, "schema_version": 1}), "schema_version"),
        (lambda raw: json.dumps({**raw, "status": "resumed"}), "unknown status"),
    ],
)
def test_a_corrupt_case_file_is_refused_not_guessed_at(store, letter, tamper, fragment):
    assert main("audit", letter, "--case", "c8", store=store)[0] == EXIT_AWAITING_HUMAN
    path = store / "c8" / "case.json"
    path.write_text(tamper(json.loads(path.read_text(encoding="utf-8"))), encoding="utf-8")
    code, _, err = main("resume", "--case", "c8", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert fragment in err


def test_a_clean_letter_completes_without_any_question(store):
    code, out, _ = main("audit", LETTERS / "06_agrees.txt", "--case", "clean", store=store)
    assert code == EXIT_OK, out
    assert "QUESTION FOR THE REVIEWER" not in out
    assert "NO DISCREPANCY FOUND" in out
    case = case_json(store, "clean")
    assert case["recomputed_degree"] == case["stated_combined"] == 70
