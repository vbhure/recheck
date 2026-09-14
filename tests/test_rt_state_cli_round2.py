"""Second round: the reviewer's open items against durable state, the CLI and the report.

Each test fails on the code before its fix (commit baa590f).

  CASE-COLLISION     a sweep discarded a reviewer's answered case when PAIR.txt
                     and pair.pdf shared one directory on a case-insensitive
                     filesystem; `show --case a` refused a case audited as A
  FILES-P4-02        a forged "Final degree of disability" trace entry printed
                     a false figure under the correct verdict
  FILES-P4-03        an actor field set to a list crashed with TypeError; a
                     re-derivation ValueError stopped the whole sweep
  HUMAN-F1           on POSIX, lock() released its lock before deleting the file
  MODEL-RT-P2P6-05   the MAP fixture classified from a listed term joined to
                     the rated condition by " - ", "/" or "with"
  Z2 cross-lane      a follow-up question from resume exited 3, not 2
  HUMAN-F2           a resume that raised lost its question for good, and
                     show/sweep still offered a question that could not be answered
  WORDING            an over-the-limit UNDETERMINED case read "The rating could
                     be not established" and its triage row was cut off
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import pathlib
import re
import shutil
import uuid

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    ROOT,
    case_json,
    main,
    require_lexicon_abstains,
    tabular_letter,
)
from recheck import case as case_module
from recheck.case import Case, CaseCorrupt, CaseStore
from recheck.graph import build_graph, outstanding_interrupt

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]
CASELOAD_MAP = ROOT / "fixtures" / "classifications" / "caseload.json"


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


def _commands(text: str, verb: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("recheck") and f" {verb} " in line]


def _case_insensitive(directory: pathlib.Path) -> bool:
    probe = directory / f"CaseProbe{uuid.uuid4().hex[:6]}"
    probe.mkdir(parents=True)
    try:
        return pathlib.Path(str(probe).lower()).exists()
    finally:
        probe.rmdir()


# ==========================================================================
# CASE-COLLISION: ids that differ only in letter case
# ==========================================================================

def test_a_sweep_never_discards_an_answered_case_whose_id_collides_ignoring_case(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "PAIR.txt", PAIR, stated=70)
    if not _case_insensitive(work):
        pytest.skip("needs a case-insensitive filesystem, where PAIR and pair are one directory")
    assert main("sweep", work, store=store)[0] == EXIT_AWAITING_HUMAN
    code, out, err = main("resume", "--case", "PAIR", "--answer", "2=left", store=store)
    assert code == EXIT_OK, out + err
    answered = (store / "PAIR" / "case.json").read_bytes()

    # A second document whose id differs only in letter case arrives. It sorts
    # first, so it is the one that reaches the case on file.
    (work / "pair.pdf").write_bytes(b"%PDF-1.4\n% not a decision letter\n")
    for _ in range(2):
        code, out, err = main("sweep", work, store=store)
        assert "Traceback" not in out + err
        flat = _flat(out)
        assert "ignoring letter case" in flat and "already used by PAIR.txt" in flat
        case = case_json(store, "PAIR")
        assert case["status"] == "complete" and case["recomputed_degree"] == 80
        assert case["human_answers"] == {"2": "lower-left"}
        assert (store / "PAIR" / "case.json").read_bytes() == answered


def test_a_sweep_without_fresh_reports_a_case_it_cannot_load_and_deletes_nothing(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "pair.txt", PAIR, stated=70)
    assert main("sweep", work, store=store)[0] == EXIT_AWAITING_HUMAN
    assert main("resume", "--case", "pair", "--answer", "2=left", store=store)[0] == EXIT_OK
    _edit(store, "pair", lambda raw: raw.update(schema_version=2))
    damaged = (store / "pair" / "case.json").read_bytes()

    code, out, err = main("sweep", work, store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    flat = _flat(out)
    assert "COULD NOT PROCEED 1" in flat
    assert "Nothing was deleted; re-audit with --fresh" in flat, "the row keeps its way forward, uncut"
    assert (store / "pair" / "case.json").read_bytes() == damaged
    assert (store / "pair" / "session").is_dir()

    code, out, _ = main("sweep", work, "--fresh", store=store)
    assert code == EXIT_AWAITING_HUMAN
    assert case_json(store, "pair")["schema_version"] == case_module.SCHEMA_VERSION


def test_a_case_is_shown_and_resumed_by_its_id_in_another_letter_case(store, tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "A", store=store)[0] == EXIT_AWAITING_HUMAN
    if not _case_insensitive(store):
        pytest.skip("needs a case-insensitive filesystem, where A and a are one directory")

    code, out, err = main("show", "--case", "a", store=store)
    assert code == EXIT_OK, out + err
    assert _commands(out, "resume") and all("--case A " in c for c in _commands(out, "resume"))

    code, out, err = main("resume", "--case", "a", "--answer", "2=left", store=store)
    assert code == EXIT_OK, out + err
    assert case_json(store, "A")["case_id"] == "A" and case_json(store, "A")["recomputed_degree"] == 80


def test_an_id_differing_only_in_case_is_refused_where_it_names_another_directory(store, tmp_path, monkeypatch):
    """On a case-sensitive filesystem "A" and "a" are two cases; the redirect check still holds there."""
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "b", store=store)[0] == EXIT_AWAITING_HUMAN
    _edit(store, "b", lambda raw: raw.update(case_id="B"))
    monkeypatch.setattr(os.path, "samefile", lambda a, b: False)
    with pytest.raises(CaseCorrupt, match="different case id"):
        CaseStore(store).load("b")


# ==========================================================================
# FILES-P4-02: trace entries that state the result must match it
# ==========================================================================

FINAL = "Final degree of disability"


def _final_entry(raw: dict) -> dict:
    return next(e for e in raw["trace"] if e["action"] == FINAL)


def _forge_trace_value(raw):
    _final_entry(raw)["value"] = "90%"


def _forge_trace_detail(raw):
    entry = _final_entry(raw)
    entry["detail"] = re.sub(r"\d+", "87", entry["detail"], count=1)


def _forge_lookalike(raw):
    raw["trace"].append({**_final_entry(raw), "action": "FINAL DEGREE OF DISABILITY.", "value": "90%"})


def _forge_arithmetic(raw):
    entry = next(e for e in raw["trace"] if e["action"] == "Arithmetic")
    entry["value"] = 88


def _forge_actor(raw):
    raw["trace"].append({**_final_entry(raw), "actor": "HUMAN"})


@pytest.mark.parametrize("forge", [_forge_trace_value, _forge_trace_detail, _forge_lookalike, _forge_arithmetic,
                                   _forge_actor], ids=lambda f: f.__name__)
def test_a_forged_result_line_in_the_trace_is_never_printed(store, tmp_path, forge):
    letter = tabular_letter(tmp_path / "tabular.txt", PAIR[:2] + [("Left knee strain", 10), ("Tinnitus", 10)],
                            stated=70)
    code, out, _ = main("audit", letter, "--case", "t", store=store)
    assert code == EXIT_OK and re.search(rf"{FINAL}: (\d+)%", out)
    genuine = int(re.search(rf"{FINAL}: (\d+)%", out).group(1))
    assert genuine != 90

    _edit(store, "t", forge)
    # A look-alike action name is now refused before the trace is checked (RG-06).
    with pytest.raises(CaseCorrupt, match="decision trace|no Recheck node writes"):
        CaseStore(store).load("t")
    code, out, err = main("show", "--case", "t", store=store)
    assert code == EXIT_CANNOT_PROCEED and re.search("decision trace|no Recheck node writes", err)
    assert "90%" not in out and FINAL not in out


def test_arithmetic_on_a_case_that_is_not_complete_is_refused(store, tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "p", store=store)[0] == EXIT_AWAITING_HUMAN
    _edit(store, "p", lambda raw: raw["trace"].append(
        {"actor": "DETERMINISTIC", "action": FINAL, "detail": "combined value 90", "value": "90%",
         "confidence": None, "evidence": None, "rule": "38 CFR 4.25(a)"}))
    with pytest.raises(CaseCorrupt, match="arithmetic in its trace"):
        CaseStore(store).load("p")


# ==========================================================================
# FILES-P4-03: unhashable actors; a re-derivation that raises
# ==========================================================================

def _set_decision(field, value):
    return lambda raw: raw["decisions"][0].__setitem__(field, value)


@pytest.mark.parametrize("change", [
    pytest.param(_set_decision("group_by", ["AI"]), id="group_by list"),
    pytest.param(_set_decision("side_by", {"x": 1}), id="side_by object"),
    pytest.param(lambda raw: raw["trace"][0].__setitem__("actor", ["DETERMINISTIC"]), id="trace actor list"),
])
def test_an_unhashable_actor_is_refused_as_corrupt_and_the_sweep_goes_on(store, tmp_path, change):
    work = tmp_path / "docs"
    tabular_letter(work / "a.txt", PAIR, stated=70)
    tabular_letter(work / "b.txt", PAIR, stated=70)
    assert main("sweep", work, store=store)[0] == EXIT_AWAITING_HUMAN
    _edit(store, "a", change)

    with pytest.raises(CaseCorrupt, match="malformed"):
        CaseStore(store).load("a")
    for command in ("show", "resume"):
        code, out, err = main(command, "--case", "a", *(["--answer", "2=left"] if command == "resume" else []),
                              store=store)
        assert code == EXIT_CANNOT_PROCEED and "Traceback" not in out + err
    code, out, err = main("sweep", work, store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert "CASELOAD TRIAGE" in out and re.search(r"\bb\s+could be 70% or 80%", out)


def test_facts_the_engine_refuses_on_a_complete_case_do_not_stop_a_sweep(store, tmp_path):
    work = tmp_path / "docs"
    tabular_letter(work / "a.txt", [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=60)
    tabular_letter(work / "b.txt", PAIR, stated=70)
    assert main("sweep", work, store=store)[0] == EXIT_AWAITING_HUMAN
    many = [{"condition": f"{side} knee strain {i}", "percent": 10, "extremity_group": "lower", "laterality": side,
             "group_by": "DETERMINISTIC", "side_by": "DETERMINISTIC", "confidence": None, "note": None,
             "evidence": None} for i in range(7) for side in ("left", "right")]
    # Without the letter's own "Extracted rating" entries and ratings, which
    # the store would otherwise find contradicted (tests/test_deep_state.py).
    _edit(store, "a", lambda raw: raw.update(
        decisions=many, ratings=[], trace=[e for e in raw["trace"] if e["action"] != "Extracted rating"]))

    with pytest.raises(CaseCorrupt, match="engine refuses|could change its result"):
        CaseStore(store).load("a")
    code, out, err = main("sweep", work, store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    flat = _flat(out)
    assert "CASELOAD TRIAGE" in flat and "COULD NOT PROCEED 1" in flat and "Nothing was deleted" in flat


# ==========================================================================
# HUMAN-F1: on POSIX the lock file is removed while the lock is still held
# ==========================================================================

def test_the_lock_file_is_removed_before_the_lock_is_released_on_posix(tmp_path, monkeypatch):
    events: list[str] = []
    real_release, real_unlink = case_module._release, pathlib.Path.unlink

    def release(fd):
        events.append("release")
        real_release(fd)

    def unlink(self, *args, **kwargs):
        if self.suffix == ".lock":
            events.append("unlink")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(case_module, "_UNLINK_WHILE_LOCKED", True, raising=False)
    monkeypatch.setattr(case_module, "_release", release)
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    with CaseStore(tmp_path / "runs").lock("c"):
        pass
    assert events == ["unlink", "release"]


# ==========================================================================
# MODEL-RT-P2P6-05: joined conditions are not classified from a listed term
# ==========================================================================

@pytest.mark.parametrize("name", [
    "Left Lisfranc injury - cubital tunnel syndrome",
    "Left Lisfranc injury/cubital tunnel syndrome",
    "Left Lisfranc injury with cubital tunnel syndrome",
    "Left Lisfranc injury and cubital tunnel syndrome",
])
def test_the_map_fixture_does_not_classify_a_joined_condition(store, tmp_path, name):
    require_lexicon_abstains(name)
    letter = tabular_letter(tmp_path / "joined.txt", [("Right wrist strain", 30), (name, 30)], stated=50)
    code, out, err = main("audit", letter, "--case", "joined", "--scripted", "--classifications", CASELOAD_MAP,
                          store=store)
    decision = case_json(store, "joined")["decisions"][1]
    assert decision["condition"] == name, "precondition: the letter row was read whole"
    assert decision["extremity_group"] == "unknown" and decision["group_by"] is None
    assert code == EXIT_AWAITING_HUMAN, out + err
    assert "POTENTIAL DISCREPANCY" not in out


def test_the_map_fixture_still_classifies_a_single_listed_condition(store, tmp_path):
    name = "Left cubital tunnel syndrome"
    letter = tabular_letter(tmp_path / "single.txt", [("Right wrist strain", 30), (name, 30)], stated=50)
    main("audit", letter, "--case", "single", "--scripted", "--classifications", CASELOAD_MAP, store=store)
    decision = case_json(store, "single")["decisions"][1]
    assert decision["extremity_group"] == "upper" and decision["group_by"] == "AI"


# ==========================================================================
# Z2 cross-lane: a follow-up question from resume exits 2
# ==========================================================================

def test_a_follow_up_question_from_resume_exits_2_and_a_rejected_answer_exits_3(store, tmp_path, monkeypatch):
    from recheck.graph import AssessNode

    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "f", store=store)[0] == EXIT_AWAITING_HUMAN

    code, out, err = main("resume", "--case", "f", "--answer", "2=sideways", store=store)
    assert code == EXIT_CANNOT_PROCEED and "not accepted" in out

    # Stand-in for a graph that asks again for a fact still missing after an
    # accepted answer (Z2's HUMAN-F11): the settle step treats it as a first pass.
    settle = AssessNode._settle
    monkeypatch.setattr(AssessNode, "_settle",
                        lambda self, *args, after_answers, **kwargs: settle(
                            self, *args, after_answers=False, **kwargs))
    code, out, err = main("resume", "--case", "f", "--answer", "2=unknown", store=store)
    assert case_json(store, "f")["status"] == "awaiting_human"
    assert case_json(store, "f")["rejected_answer"] is None
    assert "QUESTION FOR THE REVIEWER" in out and "not accepted" not in out
    assert code == EXIT_AWAITING_HUMAN, out + err


# ==========================================================================
# HUMAN-F2: a failed resume keeps its question; a stranded one is not offered
# ==========================================================================

@pytest.mark.parametrize("failure", [PermissionError("case.json is held by another program"), KeyboardInterrupt()],
                         ids=["OSError", "KeyboardInterrupt"])
def test_a_resume_that_raises_part_way_leaves_the_question_answerable(store, tmp_path, monkeypatch, failure):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "r", store=store)[0] == EXIT_AWAITING_HUMAN
    before = (store / "r" / "case.json").read_bytes()
    real_save = CaseStore.save

    def save(self, case):
        if case.human_answers:
            raise failure
        return real_save(self, case)

    with monkeypatch.context() as patch:
        patch.setattr(CaseStore, "save", save)
        if isinstance(failure, Exception):
            code, _, failed_err = main("resume", "--case", "r", "--answer", "2=left", store=store)
            assert code == EXIT_CANNOT_PROCEED
        else:
            failed_err = None
            with pytest.raises(KeyboardInterrupt):
                main("resume", "--case", "r", "--answer", "2=left", store=store)
    after_failure = (store / "r" / "case.json").read_bytes()

    code, out, err = main("show", "--case", "r", store=store)
    assert code == EXIT_OK and _commands(out, "resume"), out + err
    code, out, err = main("resume", "--case", "r", "--answer", "2=left", store=store)
    assert code == EXIT_OK, out + err
    assert case_json(store, "r")["recomputed_degree"] == 80
    assert after_failure == before
    assert failed_err is None or "put back" in failed_err


def test_show_and_sweep_do_not_offer_a_stranded_question(store, tmp_path, monkeypatch):
    work = tmp_path / "docs"
    letter = tabular_letter(work / "pair.txt", PAIR, stated=70)
    assert main("sweep", work, store=store)[0] == EXIT_AWAITING_HUMAN

    # A run that raised part-way outside cmd_resume's protection (a process
    # killed mid-run behaves the same): the session keeps an activated
    # interrupt with no node waiting on it.
    real_save = CaseStore.save

    def save(self, case):
        if case.human_answers:
            raise PermissionError("case.json is held by another program")
        return real_save(self, case)

    with monkeypatch.context() as patch:
        patch.setattr(CaseStore, "save", save)
        graph = build_graph(CaseStore(store), "pair", str(letter), None)
        pending = outstanding_interrupt(graph)
        with pytest.raises(PermissionError):
            asyncio.run(graph.invoke_async([{"interruptResponse": {"interruptId": pending.id,
                                                                    "response": {"2": "left"}}}]))
    assert case_json(store, "pair")["status"] == "awaiting_human"

    code, out, err = main("show", "--case", "pair", store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "AWAITING YOUR ANSWER" not in out and "QUESTION FOR THE REVIEWER" not in out
    assert not _commands(out, "resume") and "--fresh" in err

    code, out, err = main("sweep", work, store=store)
    assert code == EXIT_CANNOT_PROCEED
    flat = _flat(out)
    assert "NEEDS YOUR ANSWER 0" in flat and not _commands(out, "resume")
    assert "question not in its Strands session; re-audit with --fresh" in flat

    code, out, err = main("resume", "--case", "pair", "--answer", "2=left", store=store)
    assert code == EXIT_CANNOT_PROCEED and "--fresh" in err and "deleted?" not in err


# ==========================================================================
# WORDING: an UNDETERMINED case whose possibilities were not enumerated
# ==========================================================================

REASON = "too many facts are unknown to try every possibility (the unknown facts can be completed 16807 ways)"


def _over_the_limit(store_root: pathlib.Path) -> CaseStore:
    store = CaseStore(store_root)
    store.save(Case("over", "over.txt", stated_combined=70, status="undetermined",
                    undetermined_reason=REASON, possible_degrees=[]))
    return store


def test_an_undetermined_case_with_no_possible_degrees_is_worded_properly(store):
    from recheck.report import render
    from recheck.sweep import Outcome, render_triage

    case_store = _over_the_limit(store)
    report = _flat(render(case_store.load("over"), show_trace=False))
    assert "could be not established" not in report
    assert "No possible ratings were computed" in report
    assert report.count("Too many facts are unknown to try every possibility") == 1

    triage = render_triage(case_store, [Outcome("over", Outcome.UNDETERMINED)])
    row = next(line for line in triage.splitlines() if line.strip().startswith("over"))
    assert row.strip() == "over         not computed:"
    assert "Too many facts are unknown" in triage or "too many facts are unknown" in triage


def test_the_question_text_survives_an_empty_set_of_possible_degrees(store, tmp_path, monkeypatch):
    from recheck import cli, materiality

    assert cli._or([]) and "%" not in cli._or([])
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    assert main("audit", letter, "--case", "q", store=store)[0] == EXIT_AWAITING_HUMAN
    real = materiality.assess
    monkeypatch.setattr(materiality, "assess", lambda decisions: dataclasses.replace(real(decisions), possible=()))
    text = cli.question_text(CaseStore(store), "q", "", exiting=False)
    assert "could not be enumerated" in text and "could be ." not in text
