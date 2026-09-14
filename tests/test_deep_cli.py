"""Deep review, CLI lane: regression tests for defects reproduced on 12705f8.

Each test fails on 12705f8 and passes with the fix committed alongside it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from _support import EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED, EXIT_OK, ROOT, main, tabular_letter


def _cli_cp1252(*args, store: pathlib.Path) -> subprocess.CompletedProcess:
    """The command line as it runs on Windows with its output redirected or piped.

    A console gets UTF-8 from Python, but `recheck ... > report.txt` or `| more`
    encodes stdout in the ANSI code page, cp1252, with errors="strict".
    PYTHONIOENCODING=cp1252 gives that stream on any platform.
    """
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="cp1252")
    proc = subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(store), *[str(a) for a in args]],
        capture_output=True, env=env, cwd=str(ROOT), timeout=300,
    )
    proc.stdout = proc.stdout.decode("cp1252", "replace")
    proc.stderr = proc.stderr.decode("cp1252", "replace")
    return proc


# A non-breaking hyphen, as word processors write "Post-traumatic". Not in cp1252.
NB_HYPHEN_NAME = "Post‑traumatic stress disorder"


def test_a_report_is_printed_to_a_cp1252_stream_when_a_condition_name_is_outside_it(tmp_path):
    """A finished audit exited 3 with "'charmap' codec can't encode character" and no report."""
    letter = tabular_letter(tmp_path / "nb.txt", [(NB_HYPHEN_NAME, 50), ("Cervical strain", 30)], 70)
    store = tmp_path / "runs"

    audit = _cli_cp1252("audit", letter, "--case", "nb", "--brief", store=store)
    assert "codec" not in audit.stderr, audit.stderr
    assert audit.returncode == EXIT_OK, audit.stderr
    assert "traumatic stress disorder" in audit.stdout

    show = _cli_cp1252("show", "--case", "nb", "--brief", store=store)
    assert "codec" not in show.stderr, show.stderr
    assert show.returncode == EXIT_OK, show.stderr
    assert "traumatic stress disorder" in show.stdout


def test_a_sweep_prints_its_triage_to_a_cp1252_stream_when_a_file_name_is_outside_it(tmp_path):
    """One file name outside cp1252 in a collision row cost the operator the whole triage (exit 3)."""
    folder = tmp_path / "letters"
    tabular_letter(folder / "__letter.txt", [("Cervical strain", 30)], 30)
    tabular_letter(folder / "腰_letter.txt", [("Cervical strain", 30)], 30)

    sweep = _cli_cp1252("sweep", folder, store=tmp_path / "runs")
    assert "codec" not in sweep.stderr, sweep.stderr
    assert "CASELOAD TRIAGE" in sweep.stdout
    assert "COULD NOT PROCEED" in sweep.stdout
    assert sweep.returncode == EXIT_CANNOT_PROCEED


# ---------------------------------------------------------------------------
# audit --fresh must not discard the case before the audit can run
# ---------------------------------------------------------------------------

LETTERS = ROOT / "fixtures" / "letters"


def test_audit_fresh_with_an_unusable_provider_keeps_the_answered_case_on_file(tmp_path):
    """--fresh deleted the case, reviewer's answer included, and then refused the provider."""
    store = tmp_path / "runs"
    letter = LETTERS / "07_clinical_terms.txt"
    assert main("audit", letter, "--case", "keep", store=store)[0] == EXIT_AWAITING_HUMAN
    assert main("resume", "--case", "keep", "--answer", "1=upper,2=upper", store=store)[0] == EXIT_OK

    code, _, err = main("audit", letter, "--case", "keep", "--fresh", "--model", "no-such-provider", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "no-such-provider" in err
    code, out, _ = main("show", "--case", "keep", "--brief", store=store)
    assert code == EXIT_OK, "the answered case was deleted by a command that then refused to run"
    assert "reviewer" in out.lower()


# ---------------------------------------------------------------------------
# --classifications that cannot be used
# ---------------------------------------------------------------------------

def test_a_missing_classification_fixture_is_refused_not_replayed_as_empty(tmp_path):
    """A mistyped path ran with no classifications, and reports named the missing file as the fixture."""
    store = tmp_path / "runs"
    missing = tmp_path / "07_clinical_term.json"
    code, out, err = main("audit", LETTERS / "07_clinical_terms.txt", "--case", "typo", "--scripted",
                          "--classifications", missing, store=store)
    assert code == EXIT_CANNOT_PROCEED, out
    # The model lane (MODEL-7) fixed the same defect; the release keeps its wording.
    assert "no such --classifications fixture" in err
    assert not (store / "typo").exists()


def test_a_classification_fixture_that_is_not_an_object_is_refused_without_a_traceback(tmp_path):
    """A JSON list escaped from main as AttributeError (exit 1, traceback)."""
    fixture = tmp_path / "list.json"
    fixture.write_text("[1, 2]", encoding="utf-8")
    code, _, err = main("audit", LETTERS / "07_clinical_terms.txt", "--case", "lst", "--scripted",
                        "--classifications", fixture, store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert "must be a JSON object" in err


# ---------------------------------------------------------------------------
# sweep --fresh and the reviewer's answers on file
# ---------------------------------------------------------------------------

def test_sweep_fresh_says_which_reviewer_answers_it_discards(tmp_path):
    """The triage offered --fresh, and --fresh silently turned an answered case back into an open question."""
    import shutil

    store = tmp_path / "runs"
    folder = tmp_path / "letters"
    folder.mkdir()
    shutil.copy(LETTERS / "07_clinical_terms.txt", folder / "answered.txt")
    assert main("sweep", folder, store=store)[0] == EXIT_AWAITING_HUMAN
    assert main("resume", "--case", "answered", "--answer", "1=upper,2=upper", store=store)[0] == EXIT_OK

    code, out, _ = main("sweep", folder, store=store)
    assert code == EXIT_OK
    assert "discards the reviewer answers on file for 1 case(s)" in " ".join(out.split())

    code, out, _ = main("sweep", folder, "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN
    assert "--fresh discarded the reviewer answers on file for 1 case(s): answered (1=upper-left,2=upper-right)" in out


def test_sweep_warns_before_fresh_discards_the_answers_of_a_case_whose_letter_changed(tmp_path):
    """The row said "letter changed ...; re-audit with --fresh" and the footer did not say that discards answers."""
    import shutil

    store = tmp_path / "runs"
    folder = tmp_path / "letters"
    folder.mkdir()
    letter = folder / "answered.txt"
    shutil.copy(LETTERS / "07_clinical_terms.txt", letter)
    assert main("sweep", folder, store=store)[0] == EXIT_AWAITING_HUMAN
    assert main("resume", "--case", "answered", "--answer", "1=upper,2=upper", store=store)[0] == EXIT_OK
    with letter.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    code, out, _ = main("sweep", folder, store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "letter changed" in out
    assert "discards the reviewer answers on file for 1 case(s)" in " ".join(out.split())
