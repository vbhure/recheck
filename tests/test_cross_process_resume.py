"""Cross-process interrupt and resume, proven with real OS processes.

A test that calls the graph twice inside one interpreter proves nothing about
durability: module state, caches and the graph object all survive. So every
test here shells out to a SEPARATE python executable and communicates only
through the filesystem.

The decisive test is test_resume_requires_the_persisted_state_on_disk. It
deletes the persisted graph state and shows the resume then FAILS. That is
the evidence that the successful resume was reading disk rather than
inheriting anything - a passing resume alone could not distinguish the two.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LETTER = ROOT / "fixtures" / "letters" / "08_nerve_no_side.txt"
CLEAN = ROOT / "fixtures" / "letters" / "06_agrees.txt"
CLASSIFICATIONS = ROOT / "fixtures" / "classifications" / "08_nerve_no_side.json"

EXIT_OK = 0
EXIT_AWAITING_HUMAN = 2
EXIT_CANNOT_PROCEED = 3


def run_cli(*args: str, store: pathlib.Path) -> subprocess.CompletedProcess:
    """Invoke the CLI in a brand-new interpreter."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(store), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=180,
    )


def audit(store: pathlib.Path, case: str, letter: pathlib.Path = LETTER):
    return run_cli(
        "audit", str(letter), "--case", case,
        "--scripted", "--classifications", str(CLASSIFICATIONS),
        store=store,
    )


def resume(store: pathlib.Path, case: str, answer: str):
    return run_cli(
        "resume", "--case", case, "--answer", answer,
        "--scripted", "--classifications", str(CLASSIFICATIONS),
        store=store,
    )


@pytest.fixture
def store(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "runs"


# --------------------------------------------------------------------------
# Process A
# --------------------------------------------------------------------------

def test_process_a_interrupts_and_persists_then_exits(store):
    result = audit(store, "c1")
    assert result.returncode == EXIT_AWAITING_HUMAN, result.stdout + result.stderr
    assert "HUMAN INPUT REQUIRED" in result.stdout
    assert (store / "c1" / "case.json").exists()
    assert (store / "c1" / "graph_state.json").exists()

    case = json.loads((store / "c1" / "case.json").read_text(encoding="utf-8"))
    assert case["status"] == "awaiting_human"
    assert case["recomputed_degree"] is None, "must not compute before the human answers"


def test_process_a_asks_only_for_laterality(store):
    result = audit(store, "c2")
    assert "never asks a human for a percentage" in result.stdout
    assert "accepted values: left, right, unknown" in result.stdout


def test_no_framework_chatter_in_product_output(store):
    """Strands' default callback handler prints tool banners to stdout."""
    result = audit(store, "c3")
    assert "Tool #" not in result.stdout


# --------------------------------------------------------------------------
# Process B
# --------------------------------------------------------------------------

def test_resume_completes_in_a_genuinely_separate_process(store):
    first = audit(store, "c4")
    assert first.returncode == EXIT_AWAITING_HUMAN

    second = resume(store, "c4", "2=left,3=right")
    assert second.returncode == EXIT_OK, second.stdout + second.stderr

    # The two runs really were different OS processes.
    pid_a = _pid_of(first.stdout)
    pid_b = _pid_of(second.stdout)
    assert pid_b is not None
    if pid_a is not None:
        assert pid_a != pid_b

    case = json.loads((store / "c4" / "case.json").read_text(encoding="utf-8"))
    assert case["status"] == "complete"
    assert case["recomputed_degree"] == 80
    assert case["stated_combined"] == 70
    assert case["bilateral_applied"] is True
    assert case["human_answers"] == {"2": "left", "3": "right"}
    assert "POTENTIAL DISCREPANCY" in second.stdout


def test_resume_requires_the_persisted_state_on_disk(store):
    """The decisive evidence that resume is not inheriting memory.

    Delete the persisted graph state and the resume must fail. If the second
    process were relying on anything carried over from the first, this would
    still succeed.
    """
    assert audit(store, "c5").returncode == EXIT_AWAITING_HUMAN
    (store / "c5" / "graph_state.json").unlink()

    result = resume(store, "c5", "2=left,3=right")
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "nothing to resume" in (result.stdout + result.stderr)


def test_resume_of_an_unknown_case_fails_cleanly(store):
    result = resume(store, "never-existed", "0=left")
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "no such case" in (result.stdout + result.stderr)


def test_corrupt_case_file_is_refused_not_guessed_at(store):
    assert audit(store, "c6").returncode == EXIT_AWAITING_HUMAN
    (store / "c6" / "case.json").write_text("{not json", encoding="utf-8")
    result = resume(store, "c6", "2=left,3=right")
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "not valid JSON" in (result.stdout + result.stderr)


def test_case_from_an_incompatible_schema_is_refused(store):
    assert audit(store, "c7").returncode == EXIT_AWAITING_HUMAN
    path = store / "c7" / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")
    result = resume(store, "c7", "2=left,3=right")
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "schema_version" in (result.stdout + result.stderr)


# --------------------------------------------------------------------------
# The human answer is validated, not trusted
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "answer,expected_fragment",
    [
        ("2=sideways,3=right", "is not one of"),
        ("2=left", "no answer supplied"),
        ("99=left,2=left,3=right", "no condition at index 99"),
        ("0=left,2=left,3=right", "did not require human input"),
        ("garbage", "cannot parse"),
        ("=left", "is not a condition index"),
    ],
)
def test_invalid_human_answers_are_rejected(store, answer, expected_fragment):
    case_id = "bad" + str(abs(hash(answer)) % 10000)
    assert audit(store, case_id).returncode == EXIT_AWAITING_HUMAN
    result = resume(store, case_id, answer)
    assert result.returncode == EXIT_CANNOT_PROCEED, result.stdout
    assert expected_fragment in (result.stdout + result.stderr)

    case = json.loads((store / case_id / "case.json").read_text(encoding="utf-8"))
    assert case["recomputed_degree"] is None, "a rejected answer must not produce arithmetic"


def test_human_answer_cannot_supply_a_percentage(store):
    """The answer grammar has no place for a number, so a human cannot set
    the outcome directly even by trying."""
    assert audit(store, "c8").returncode == EXIT_AWAITING_HUMAN
    result = resume(store, "c8", "2=100,3=100")
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "is not one of" in (result.stdout + result.stderr)


def test_human_may_answer_unknown_and_the_factor_is_not_applied(store):
    """Declining to determine a side is a legitimate answer, and it must
    prevent 4.26 rather than provoke a guess."""
    assert audit(store, "c9").returncode == EXIT_AWAITING_HUMAN
    result = resume(store, "c9", "2=unknown,3=unknown")
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    case = json.loads((store / "c9" / "case.json").read_text(encoding="utf-8"))
    assert case["bilateral_applied"] is False
    assert case["recomputed_degree"] == 70
    assert "NO DISCREPANCY FOUND" in result.stdout


# --------------------------------------------------------------------------
# The no-discrepancy control
# --------------------------------------------------------------------------

def test_clean_letter_completes_without_any_interrupt(store):
    """Golden case C: Recheck is not built to manufacture errors."""
    result = audit(store, "clean", letter=CLEAN)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert "HUMAN INPUT REQUIRED" not in result.stdout
    assert "NO DISCREPANCY FOUND" in result.stdout
    case = json.loads((store / "clean" / "case.json").read_text(encoding="utf-8"))
    assert case["recomputed_degree"] == case["stated_combined"] == 70


def test_case_id_cannot_escape_the_store_directory(store):
    for hostile in ("../escape", "..", "a/b", "with space"):
        result = run_cli("show", "--case", hostile, store=store)
        assert result.returncode == EXIT_CANNOT_PROCEED
        assert "case id may contain only" in (result.stdout + result.stderr)


def _pid_of(text: str) -> int | None:
    import re

    match = re.search(r"\(pid (\d+)\)", text)
    return int(match.group(1)) if match else None
