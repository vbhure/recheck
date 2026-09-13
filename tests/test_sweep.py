"""The caseload sweep - the product's actual shape.

Working through a stack unattended, finishing what can be established, and
surfacing only the cases where an answer would change a rating is the
product. fixtures/caseload is regenerated independently, so these tests
assert STRUCTURE and INVARIANTS - that the triage agrees with the case files,
that nothing is asked or computed that should not be - rather than counts or
case ids.

Defects regression-locked here:

  S1  Re-running a sweep over its own store re-entered each interrupted
      Strands session with a plain prompt and failed with "TypeError: ...
      must resume from interrupt with list of interruptResponse's"; the
      case moved to COULD NOT PROCEED and the exit code dropped from 2 to 0.
      A second sweep now reports existing cases as they stand, identically.
  S2  At 0d018c9 a sweep in which every document failed exited 0, the code
      for "finished". It now exits 3 and the summary claims nothing.
  S3  Two documents whose names sanitise to the same case id were both
      audited into the same case directory, the second on top of the first.
      The second is now reported, not merged.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    ROOT,
    cli,
    main,
    tabular_letter,
)
from recheck.case import CaseStore
from recheck.classify import derive_laterality
from recheck.graph import build_graph, interrupt_id, outstanding_interrupt
from recheck.materiality import assess
from recheck.provenance import Actor
from recheck.schema import CONFIDENCE_FLOOR
from recheck.sweep import Outcome, case_id_for, documents_in, triage_exit_code

CASELOAD = ROOT / "fixtures" / "caseload"
CASELOAD_MAP = ROOT / "fixtures" / "classifications" / "caseload.json"
FORBIDDEN = ("the va is wrong", "va erred", "incorrect decision", "you are owed", "error by the va")
SUMMARY = re.compile(r"(\d+) documents: (\d+) no discrepancy, (\d+) to review, (\d+) waiting on an answer, "
                     r"(\d+) undetermined, (\d+) could not proceed")

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]


def _triage(stdout: str) -> str:
    """The triage block, without the line that only says cases were reused."""
    start = stdout.index("CASELOAD TRIAGE")
    return "\n".join(l for l in stdout[start:].splitlines() if "already on file" not in l)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [("case_001", "case_001"), ("letter-07", "letter-07"), ("with space", "with_space"),
     ("weird!@#$%", "weird_____"), ("../../escape", "escape"), ("a/b/c", "c"),
     ("con", "case_con"), ("LPT1", "case_LPT1")],
)
def test_case_ids_are_sanitised_from_filenames(name, expected):
    assert case_id_for(pathlib.Path(f"{name}.txt")) == expected


@pytest.mark.parametrize(
    "filename",
    ["../../etc/passwd.txt", "a/b/c.txt", ".txt", "..txt", "C:/windows/system32/x.txt",
     "with space.txt", "weird!@#$%.txt", "-.txt", "nul.txt", "COM3.pdf"],
)
def test_a_derived_case_id_is_always_accepted_by_the_store(filename, tmp_path):
    """The two guards agree: whatever the sweep derives, the store accepts,
    and it never names a path outside the store."""
    case_id = case_id_for(pathlib.Path(filename))
    store = CaseStore(tmp_path / "runs")
    path = store.path_for(case_id)  # raises if the store would refuse it
    assert path.resolve().parent.parent == (tmp_path / "runs").resolve()


def test_only_documents_are_swept(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.PDF").write_bytes(b"%PDF-1.4")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub.txt").mkdir()
    assert {p.name for p in documents_in(tmp_path)} == {"a.txt", "b.PDF"}


def test_the_exit_code_says_whether_anyone_is_needed_or_anything_went_unresolved():
    done, ask = Outcome("a", Outcome.COMPLETE), Outcome("b", Outcome.AWAITING_HUMAN)
    undetermined, failed = Outcome("c", Outcome.UNDETERMINED), Outcome("d", Outcome.FAILED, "x")
    assert triage_exit_code([done]) == EXIT_OK
    assert triage_exit_code([done, failed]) == EXIT_CANNOT_PROCEED
    assert triage_exit_code([done, undetermined]) == EXIT_CANNOT_PROCEED
    assert triage_exit_code([failed, undetermined, ask]) == EXIT_AWAITING_HUMAN


# --------------------------------------------------------------------------
# The caseload: invariants, not counts
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def caseload(tmp_path_factory):
    store_root = tmp_path_factory.mktemp("caseload") / "runs"
    result = cli("sweep", CASELOAD, "--scripted", "--classifications", CASELOAD_MAP, store=store_root, timeout=900)
    assert "Traceback" not in result.stdout + result.stderr, result.stderr
    snapshot = {p.name: (p / "case.json").read_bytes() for p in store_root.iterdir()}
    return store_root, result, snapshot


def _cases(store_root):
    store = CaseStore(store_root)
    return store, {p.name: store.load(p.name) for p in sorted(store_root.iterdir())}


def test_every_document_becomes_its_own_case_and_the_triage_agrees_with_the_files(caseload):
    store_root, result, _ = caseload
    documents = documents_in(CASELOAD)
    _, cases = _cases(store_root)
    assert sorted(cases) == sorted(case_id_for(d) for d in documents)

    total, agree, review, waiting, undetermined, failed = map(int, SUMMARY.search(result.stdout).groups())
    statuses = [c.status for c in cases.values()]
    assert total == len(documents)
    assert waiting == statuses.count("awaiting_human")
    assert undetermined == statuses.count("undetermined")
    complete = [c for c in cases.values() if c.status == "complete"]
    assert agree == sum(c.recomputed_degree == c.stated_combined for c in complete)
    assert review == sum(c.recomputed_degree != c.stated_combined for c in complete)
    assert failed == total - agree - review - waiting - undetermined

    expected = EXIT_AWAITING_HUMAN if waiting else EXIT_CANNOT_PROCEED if (undetermined or failed) else EXIT_OK
    assert result.returncode == expected


def test_nothing_is_computed_that_is_not_settled_and_nothing_settled_is_asked(caseload):
    store_root, result, _ = caseload
    store, cases = _cases(store_root)
    for case_id, case in cases.items():
        decisions = case.load_decisions()
        m = assess(decisions)
        ran_compute = "compute" in [step["node"] for step in case.timeline]
        if case.status == "complete":
            assert ran_compute and case.recomputed_degree is not None, case_id
            assert m.settled, f"{case_id} was computed although its unknown facts matter"
            assert case.recomputed_degree == m.possible[0], f"{case_id}: reported figure is not the settled one"
            assert sorted(case.immaterial_unknowns) == list(m.unknown), case_id
        else:
            assert not ran_compute and case.recomputed_degree is None, case_id
        if case.status == "awaiting_human":
            assert m.answers_matter and len(set(case.possible_degrees)) > 1, case_id
            pending = outstanding_interrupt(build_graph(store, case_id, case.source_path, None))
            assert pending is not None and pending.id == interrupt_id(case_id), case_id
            asked = [c["index"] for c in pending.reason["conditions"]]
            assert asked == list(m.unknown), case_id
            assert f"resume --case {case_id} --answer" in result.stdout
        if case.status == "undetermined":
            assert len(case.possible_degrees) > 1, case_id


def test_ownership_holds_across_the_whole_caseload(caseload):
    store_root, result, _ = caseload
    _, cases = _cases(store_root)
    counts = {"DETERMINISTIC": 0, "AI": 0, "HUMAN": 0, "unknown": 0}
    sides = 0
    for case in cases.values():
        for d in case.load_decisions():
            counts[d.group_by.value if d.group_by else "unknown"] += 1
            if d.group_by is Actor.AI:
                assert d.confidence is not None and d.confidence >= CONFIDENCE_FLOOR
            if d.side_by is Actor.DETERMINISTIC:
                sides += 1
                assert d.laterality == derive_laterality(d.condition), d.condition
            assert d.side_by is not Actor.AI, "the model never establishes a side"
            if d.extremity_group == "unknown":
                assert d.group_by is None
    line = (f"extremity groups: {counts['DETERMINISTIC']} lexicon, {counts['AI']} AI, {counts['HUMAN']} reviewer, "
            f"{counts['unknown']} unknown.  sides read from the letters: {sides}.  arithmetic: all deterministic")
    assert line in result.stdout


def test_the_triage_never_reads_as_a_finding_of_error(caseload):
    _, result, _ = caseload
    out = result.stdout.lower()
    for phrase in FORBIDDEN:
        assert phrase not in out
    if "recomputed higher than the letter" in out or "recomputed lower than the letter" in out:
        assert "not findings of error" in out
    if "recomputed lower than the letter" in out:
        assert "downward review" in out


def test_a_second_sweep_reports_the_same_triage_and_changes_nothing(caseload):
    """S1."""
    store_root, first, snapshot = caseload
    second = cli("sweep", CASELOAD, "--scripted", "--classifications", CASELOAD_MAP, store=store_root, timeout=900)
    assert second.returncode == first.returncode
    assert "TypeError" not in second.stdout + second.stderr
    assert _triage(second.stdout) == _triage(first.stdout)
    assert f"{len(snapshot)} case(s) were already on file" in second.stdout
    assert {p.name: (p / "case.json").read_bytes() for p in store_root.iterdir()} == snapshot


# --------------------------------------------------------------------------
# Small directories: the edges
# --------------------------------------------------------------------------

def test_a_fresh_sweep_re_audits_and_agrees_with_itself(tmp_path):
    work = tmp_path / "work"
    tabular_letter(work / "pair.txt", PAIR, stated=70)
    shutil.copy(LETTERS / "06_agrees.txt", work / "agrees.txt")
    store_root = tmp_path / "runs"
    code1, out1, err1 = main("sweep", work, store=store_root)
    before = (store_root / "agrees" / "case.json").read_bytes()
    code2, out2, err2 = main("sweep", work, "--fresh", store=store_root)
    assert code1 == code2 == EXIT_AWAITING_HUMAN
    assert "TypeError" not in out1 + err1 + out2 + err2
    assert _triage(out1) == _triage(out2)
    assert "already on file" not in out2
    assert json.loads(before)["status"] == json.loads((store_root / "agrees" / "case.json").read_bytes())["status"]


def test_a_sweep_of_only_unusable_documents_exits_3_and_claims_nothing(tmp_path):
    """S2."""
    work = tmp_path / "junk"
    work.mkdir()
    (work / "letter.txt").write_text("Dear veteran,\n\nThank you.\n", encoding="utf-8")
    (work / "empty.txt").write_text("", encoding="utf-8")
    (work / "binary.txt").write_bytes(b"\xff\xfe\x00\x81 not utf-8")
    code, out, _ = main("sweep", work, store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert SUMMARY.search(out).groups() == ("3", "0", "0", "0", "0", "3")
    assert "questions asked: 0" in out
    assert "resolved" not in out.lower()


def test_a_refused_document_keeps_its_reason_when_the_sweep_is_run_again(tmp_path):
    """S1 for refusals. A scanned PDF's reason used to live only in the first
    run's output; the case stayed "open", and the second sweep reported "the
    run stopped with the case in state 'open'" instead."""
    from fpdf import FPDF

    work = tmp_path / "refused"
    work.mkdir()
    pdf = FPDF()
    pdf.add_page()
    pdf.rect(20, 20, 100, 60, style="F")
    pdf.output(str(work / "scan.pdf"))
    (work / "binary.txt").write_bytes(b"\xff\xfe\x00\x81 not utf-8")
    store_root = tmp_path / "runs"

    code1, out1, _ = main("sweep", work, store=store_root)
    code2, out2, _ = main("sweep", work, store=store_root)
    assert code1 == code2 == EXIT_CANNOT_PROCEED
    assert _triage(out1) == _triage(out2)
    assert "no extractable text layer" in out2
    assert "state 'open'" not in out2
    store = CaseStore(store_root)
    assert store.load("scan").status == store.load("binary").status == "unparsed"
    code, shown, _ = main("show", "--case", "scan", store=store_root)
    assert "COULD NOT READ THE LETTER" in shown and "requires OCR" in shown


def test_documents_that_map_to_one_case_id_are_reported_not_merged(tmp_path):
    """S3."""
    work = tmp_path / "dupes"
    work.mkdir()
    shutil.copy(LETTERS / "06_agrees.txt", work / "a b.txt")
    tabular_letter(work / "a_b.txt", PAIR, stated=70)
    store_root = tmp_path / "runs"
    code, out, _ = main("sweep", work, store=store_root)
    assert code == EXIT_CANNOT_PROCEED
    assert "map to the same case id" in out
    assert [p.name for p in store_root.iterdir()] == ["a_b"]
    case = CaseStore(store_root).load("a_b")
    assert case.source_path.endswith("a b.txt")
    assert (case.stated_combined, case.recomputed_degree) == (70, 70)


def test_a_case_id_already_used_by_another_document_is_not_overwritten(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    shutil.copy(LETTERS / "06_agrees.txt", first / "letter.txt")
    tabular_letter(second / "letter.txt", PAIR, stated=70)
    store_root = tmp_path / "runs"
    assert main("sweep", first, store=store_root)[0] == EXIT_OK
    before = (store_root / "letter" / "case.json").read_bytes()

    code, out, _ = main("sweep", second, store=store_root)
    assert code == EXIT_CANNOT_PROCEED
    assert "already used by" in out
    assert (store_root / "letter" / "case.json").read_bytes() == before

    code, _, _ = main("sweep", second, "--fresh", store=store_root)
    assert code == EXIT_AWAITING_HUMAN


def test_one_unreadable_document_does_not_stop_the_sweep(tmp_path):
    work = tmp_path / "mixed"
    work.mkdir()
    shutil.copy(LETTERS / "06_agrees.txt", work / "good_one.txt")
    shutil.copy(LETTERS / "01_tabular.txt", work / "good_two.txt")
    (work / "unparseable.txt").write_text("Dear veteran,\n\nThank you.\n", encoding="utf-8")
    store_root = tmp_path / "runs"
    code, out, _ = main("sweep", work, store=store_root)
    assert code == EXIT_CANNOT_PROCEED
    assert SUMMARY.search(out).groups() == ("3", "1", "1", "0", "0", "1")
    store = CaseStore(store_root)
    assert store.load("good_one").status == store.load("good_two").status == "complete"


def test_an_undetermined_case_is_its_own_triage_bucket(tmp_path):
    work = tmp_path / "und"
    tabular_letter(work / "both.txt", [("Migraine headaches", 20), ("Bilateral knee strain", 30)], stated=40)
    code, out, _ = main("sweep", work, store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert SUMMARY.search(out).groups() == ("1", "0", "0", "0", "1", "0")
    assert re.search(r"both\s+could be 40% or 50%", out)


def test_a_question_from_a_sweep_is_answered_by_a_separate_process(tmp_path):
    work = tmp_path / "work"
    tabular_letter(work / "pair.txt", PAIR, stated=70)
    store_root = tmp_path / "runs"
    code, out, _ = main("sweep", work, store=store_root)
    assert code == EXIT_AWAITING_HUMAN
    (command,) = [l.strip() for l in out.splitlines() if l.strip().startswith("recheck") and " resume " in l]
    # Printed with forward slashes, which every shell (and Windows) accepts.
    assert command == f'recheck --store {store_root.as_posix()} resume --case pair --answer "2=<left|right|both|unknown>"'

    result = cli("resume", "--case", "pair", "--answer", "2=left", store=store_root)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert CaseStore(store_root).load("pair").recomputed_degree == 80


def test_sweep_of_an_empty_directory_fails_cleanly(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    code, _, err = main("sweep", empty, store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert "no .txt or .pdf documents" in err


def test_sweep_of_a_missing_directory_fails_cleanly(tmp_path):
    code, _, err = main("sweep", tmp_path / "does-not-exist", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert "not a directory" in err
