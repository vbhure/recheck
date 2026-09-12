"""The caseload sweep - the product's actual shape.

A single interactive audit demonstrates the machinery. Working through a stack
unattended, completing everything resolvable without a human, and surfacing
only the cases where a person must supply a fact the document does not contain
is the product. These tests pin that behaviour, including the property the
whole argument rests on: MOST documents must be resolved without a human.

They also pin the two things a sweep must never do - stop because one document
was bad, and let a document's filename escape the case store.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from recheck.sweep import Outcome, case_id_for, documents_in, triage_exit_code

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASELOAD = ROOT / "fixtures" / "caseload"
CASELOAD_MAP = ROOT / "fixtures" / "classifications" / "caseload.json"
LETTERS = ROOT / "fixtures" / "letters"

EXIT_OK, EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED = 0, 2, 3


def cli(*args: str, store: pathlib.Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(store), *[str(a) for a in args]],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=600,
    )


def sweep_cli(directory, store: pathlib.Path) -> subprocess.CompletedProcess:
    return cli("sweep", directory, "--scripted", "--classifications", CASELOAD_MAP, store=store)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "runs"


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("case_001", "case_001"),
        ("letter-07", "letter-07"),
        ("with space", "with_space"),
        ("weird!@#$%", "weird_____"),
        # Path.stem already discards directory components, so a traversal
        # attempt in a filename never reaches the sanitiser at all.
        ("../../escape", "escape"),
        ("a/b/c", "c"),
    ],
)
def test_case_ids_are_sanitised_from_filenames(name, expected):
    assert case_id_for(pathlib.Path(f"{name}.txt")) == expected


@pytest.mark.parametrize(
    "filename",
    ["../../etc/passwd.txt", "a/b/c.txt", ".txt", "..txt", "C:/windows/system32/x.txt",
     "with space.txt", "weird!@#$%.txt", "-.txt"],
)
def test_a_case_id_can_never_escape_the_store(filename):
    """The property that matters, asserted rather than guessed at.

    The store itself also rejects anything outside [A-Za-z0-9_-], so this is
    the outer of two independent guards.
    """
    case_id = case_id_for(pathlib.Path(filename))
    assert case_id, "a case id must never be empty"
    assert all(c.isalnum() or c in "-_" for c in case_id), case_id
    for forbidden in ("/", "\\", "..", ":"):
        assert forbidden not in case_id


def test_only_documents_are_swept(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    found = {p.name for p in documents_in(tmp_path)}
    assert found == {"a.txt", "b.pdf"}


def test_exit_code_reflects_whether_a_human_is_needed():
    assert triage_exit_code([Outcome("a", Outcome.COMPLETE)]) == EXIT_OK
    assert triage_exit_code([Outcome("a", Outcome.FAILED, "x")]) == EXIT_OK
    assert triage_exit_code(
        [Outcome("a", Outcome.COMPLETE), Outcome("b", Outcome.AWAITING_HUMAN)]
    ) == EXIT_AWAITING_HUMAN


# --------------------------------------------------------------------------
# The sweep end to end
# --------------------------------------------------------------------------

def test_the_caseload_sweep_resolves_most_documents_without_a_human(store):
    """The property the product's whole argument rests on.

    If a majority of a caseload needed a human, the tool would be a queue
    generator rather than an assistant. A regression here is a product
    regression, not a test regression - which is exactly what happened once:
    laterality came from the model, the model correctly abstained, and 15 of
    24 documents were escalated instead of 4.
    """
    result = sweep_cli(CASELOAD, store=store)
    assert result.returncode == EXIT_AWAITING_HUMAN, result.stdout + result.stderr
    out = result.stdout

    assert "CASELOAD TRIAGE" in out
    assert "24 document(s)" in out

    swept = list(store.iterdir())
    assert len(swept) == 24, "every document must get its own durable case"

    awaiting = _count_awaiting(store)
    assert awaiting > 0, "the caseload deliberately contains ambiguous letters"
    assert awaiting <= 8, f"{awaiting} of 24 need a human; the product claim fails above ~1/3"
    assert f"The other {24 - awaiting} were resolved without one." in out


def test_the_sweep_reports_discrepancies_cautiously(store):
    result = sweep_cli(CASELOAD, store=store)
    out = result.stdout
    assert "POTENTIAL DISCREPANCY" in out
    assert "not findings of error" in out
    # Never a legal conclusion, anywhere in the triage.
    for forbidden in ("the VA is wrong", "VA erred", "incorrect decision", "you are owed"):
        assert forbidden.lower() not in out.lower()


def test_every_swept_case_is_independently_resumable(store):
    """Each document carries its own state, so the ones needing a decision can
    be answered later, in any order, by a different process."""
    assert sweep_cli(CASELOAD, store=store).returncode == EXIT_AWAITING_HUMAN
    awaiting = [
        path.name for path in store.iterdir()
        if json.loads((path / "case.json").read_text(encoding="utf-8"))["status"]
        == "awaiting_human"
    ]
    assert awaiting, "expected at least one case awaiting a human"

    case_id = sorted(awaiting)[0]
    case = json.loads((store / case_id / "case.json").read_text(encoding="utf-8"))
    indices = [i for i, d in enumerate(case["decisions"]) if d["needs_human"]]
    answer = ",".join(f"{index}={'left' if n % 2 == 0 else 'right'}"
                      for n, index in enumerate(indices))

    result = cli("resume", "--case", case_id, "--answer", answer,
                 "--scripted", "--classifications", CASELOAD_MAP, store=store)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    after = json.loads((store / case_id / "case.json").read_text(encoding="utf-8"))
    assert after["status"] == "complete"
    assert after["recomputed_degree"] is not None


def test_one_unreadable_document_does_not_stop_the_sweep(store, tmp_path):
    """A stack of real correspondence will contain something unusable."""
    work = tmp_path / "mixed"
    work.mkdir()
    shutil.copy(LETTERS / "06_agrees.txt", work / "good_one.txt")
    shutil.copy(LETTERS / "01_tabular.txt", work / "good_two.txt")
    (work / "unparseable.txt").write_text("Dear veteran,\n\nThank you.\n", encoding="utf-8")
    (work / "empty.txt").write_text("", encoding="utf-8")

    result = sweep_cli(work, store=store)
    out = result.stdout
    assert "4 document(s)" in out
    assert "COULD NOT PROCEED" in out
    # The good ones still got audited.
    assert (store / "good_one" / "case.json").exists()
    assert (store / "good_two" / "case.json").exists()


def test_sweep_of_an_empty_directory_fails_cleanly(store, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    result = sweep_cli(empty, store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "no .txt or .pdf documents" in (result.stdout + result.stderr)


def test_sweep_of_a_missing_directory_fails_cleanly(store, tmp_path):
    result = sweep_cli(tmp_path / "does-not-exist", store=store)
    assert result.returncode == EXIT_CANNOT_PROCEED
    assert "not a directory" in (result.stdout + result.stderr)


def test_the_sweep_reports_ownership_across_the_whole_caseload(store):
    """The division of labour must be visible in aggregate, not just per case."""
    out = sweep_cli(CASELOAD, store=store).stdout
    assert "ownership across the sweep:" in out
    assert "deterministic" in out and "AI" in out and "human" in out


def _count_awaiting(store: pathlib.Path) -> int:
    return sum(
        1 for path in store.iterdir()
        if (path / "case.json").exists()
        and json.loads((path / "case.json").read_text(encoding="utf-8"))["status"]
        == "awaiting_human"
    )
