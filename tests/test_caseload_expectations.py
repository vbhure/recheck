"""The committed caseload, checked against outcomes derived by hand.

The previous caseload's stated values were computed by the engine under test,
so a sweep could only ever agree with the code it was supposed to check. The
letters pinned here instead carry values worked out by hand, with every Table
I step written down in fixtures/caseload/EXPECTED.md. The expectations below
are copied from that file, not from tools/make_caseload.py, and the test reads
only committed fixtures: the letters and fixtures/classifications/caseload.json.

Each letter pins one behaviour a sweep has to get right:

  (i)    three leg disabilities - the whole group enters the factor
  (ii)   four extremities form one group (4.26(b))
  (iii)  a linked clause does not decide the rated condition's extremity
  (iv)   an unstated side that cannot change the result asks nobody
  (v)    unstated sides that can change the result become a question
  (vi)   a clinical term the classifier fixture does not list stays unknown,
         and becomes a question for its extremity group
  (vii)  a letter stating MORE than the regulation gives is reported lower
  (ix)   clinical names with no side: the classifier supplies the group, a
         reviewer the side

The sweep runs once, in a subprocess, exactly as an operator would run it.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASELOAD = ROOT / "fixtures" / "caseload"
CASELOAD_MAP = ROOT / "fixtures" / "classifications" / "caseload.json"

EXIT_AWAITING_HUMAN = 2


@pytest.fixture(scope="module")
def swept(tmp_path_factory) -> pathlib.Path:
    store = tmp_path_factory.mktemp("caseload") / "runs"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(store), "sweep", str(CASELOAD),
         "--scripted", "--classifications", str(CASELOAD_MAP)],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=900,
    )
    assert result.returncode == EXIT_AWAITING_HUMAN, result.stdout + result.stderr
    return store


def case(store: pathlib.Path, case_id: str) -> dict:
    return json.loads((store / case_id / "case.json").read_text(encoding="utf-8"))


def ratings(c: dict) -> list[tuple[str, int]]:
    return [(d["condition"], d["percent"]) for d in c["decisions"]]


# Hand-derived expectations, copied from fixtures/caseload/EXPECTED.md.
# (case id, evaluations as the letter writes them, stated, expected outcome)
COMPLETE = {
    "i_three_legs": ("case_020", [
        ("post-traumatic stress disorder", 30),
        ("limitation of flexion of the left knee", 20),
        ("limited motion of the left ankle", 20),
        ("limitation of flexion of the right knee", 30),
        ("tinnitus", 10),
    ], 80, {"combined": 76, "degree": 80, "bilateral": True}),
    "ii_four_extremities": ("case_021", [
        ("lumbosacral strain", 20),
        ("right cubital tunnel syndrome", 10),
        ("left cubital tunnel syndrome", 10),
        ("radiculopathy of the right lower extremity", 20),
        ("radiculopathy of the left lower extremity", 10),
    ], 50, {"combined": 57, "degree": 60, "bilateral": True}),
    "iii_linked_clause": ("case_010", [
        ("Lumbosacral strain", 20),
        ("Radiculopathy, left lower extremity, associated with lumbosacral strain", 20),
        ("Radiculopathy, right lower extremity, associated with lumbosacral strain", 10),
    ], 50, {"combined": 45, "degree": 50, "bilateral": True}),
    "iv_sideless_immaterial": ("case_001", [
        ("post-traumatic stress disorder", 70),
        ("limitation of flexion of the knee", 10),
        ("tinnitus", 10),
    ], 80, {"combined": 76, "degree": 80, "bilateral": False}),
    "vii_stated_higher": ("case_015", [
        ("post-traumatic stress disorder", 50),
        ("tinnitus", 10),
        ("limitation of flexion of the right knee", 10),
        ("limited motion of the right ankle", 10),
    ], 70, {"combined": 64, "degree": 60, "bilateral": False}),
}

AWAITING = {
    "v_sideless_material": ("case_012", [
        ("post-traumatic stress disorder", 50),
        ("limitation of flexion of the knee", 20),
        ("plantar fasciitis", 10),
    ], 60, {"possible": [60, 70], "asked": {1: ["side"], 2: ["side"]}}),
    "vi_unlisted_long_tail": ("case_019", [
        ("post-traumatic stress disorder", 50),
        ("limitation of flexion of the right knee", 20),
        ("left Lisfranc injury", 10),
    ], 60, {"possible": [60, 70], "asked": {2: ["extremity group"]}}),
    "ix_sideless_long_tail_pair": ("case_014", [
        ("post-traumatic stress disorder", 50),
        ("cubital tunnel syndrome", 20),
        ("De Quervain's tenosynovitis", 10),
    ], 60, {"possible": [60, 70], "asked": {1: ["side"], 2: ["side"]}}),
}


def _missing(decision: dict) -> list[str]:
    facts = []
    if decision["extremity_group"] == "unknown":
        facts.append("extremity group")
    if decision["laterality"] == "unknown" and decision["extremity_group"] in ("upper", "lower", "unknown"):
        facts.append("side")
    return facts


@pytest.mark.parametrize("key", sorted(COMPLETE))
def test_hand_derived_letters_complete_with_the_hand_derived_degree(swept, key):
    case_id, evaluations, stated, expected = COMPLETE[key]
    c = case(swept, case_id)
    assert ratings(c) == evaluations, f"{case_id} is not the letter EXPECTED.md describes"
    assert c["stated_combined"] == stated
    assert c["status"] == "complete"
    assert c["recomputed_combined"] == expected["combined"]
    assert c["recomputed_degree"] == expected["degree"]
    assert c["bilateral_applied"] is expected["bilateral"]


@pytest.mark.parametrize("key", sorted(AWAITING))
def test_hand_derived_letters_ask_only_for_the_facts_that_matter(swept, key):
    case_id, evaluations, stated, expected = AWAITING[key]
    c = case(swept, case_id)
    assert ratings(c) == evaluations, f"{case_id} is not the letter EXPECTED.md describes"
    assert c["stated_combined"] == stated
    assert c["status"] == "awaiting_human"
    assert c["recomputed_degree"] is None, "nothing is computed while a material fact is open"
    assert c["possible_degrees"] == expected["possible"]
    asked = {i: _missing(d) for i, d in enumerate(c["decisions"]) if _missing(d)}
    assert asked == expected["asked"]


def test_i_every_leg_disability_enters_the_factor(swept):
    """A single-pair engine gets 70% here and flags a correct letter."""
    c = case(swept, "case_020")
    assert c["alternative_degree"] == 70
    assert c["recomputed_degree"] == c["stated_combined"] == 80


def test_ii_the_classifier_fixture_supplies_the_arm_group_and_the_letter_the_sides(swept):
    decisions = case(swept, "case_021")["decisions"]
    arms = [d for d in decisions if "cubital tunnel" in d["condition"]]
    assert [(d["extremity_group"], d["group_by"], d["side_by"]) for d in arms] == [
        ("upper", "AI", "DETERMINISTIC"), ("upper", "AI", "DETERMINISTIC")]
    legs = [d for d in decisions if "radiculopathy" in d["condition"]]
    assert all(d["group_by"] == "DETERMINISTIC" for d in legs)


def test_iii_the_linked_condition_does_not_make_a_leg_disability_none(swept):
    decisions = case(swept, "case_010")["decisions"]
    assert [(d["extremity_group"], d["laterality"]) for d in decisions] == [
        ("none", "unknown"), ("lower", "left"), ("lower", "right")]


def test_iv_the_unstated_side_is_recorded_as_immaterial(swept):
    c = case(swept, "case_001")
    assert c["immaterial_unknowns"] == [1]
    assert c["human_answers"] == {}


def test_vi_an_unlisted_term_is_refused_not_guessed(swept):
    lisfranc = case(swept, "case_019")["decisions"][2]
    assert lisfranc["extremity_group"] == "unknown"
    assert lisfranc["group_by"] is None
    assert lisfranc["laterality"] == "left" and lisfranc["side_by"] == "DETERMINISTIC"


def test_vii_a_stated_value_above_the_regulation_is_reported_lower(swept):
    c = case(swept, "case_015")
    assert c["recomputed_degree"] < c["stated_combined"]


def test_the_rest_of_the_caseload_finishes_without_a_human(swept):
    """Only the three hand-derived question letters wait on a reviewer."""
    statuses = {p.name: case(swept, p.name)["status"] for p in swept.iterdir()}
    assert len(statuses) == 24
    waiting = sorted(k for k, v in statuses.items() if v == "awaiting_human")
    assert waiting == ["case_012", "case_014", "case_019"]
    assert all(v in ("complete", "awaiting_human") for v in statuses.values())


def test_the_classifier_fixture_works_on_some_letters_not_all(swept):
    by_letter = {}
    for p in swept.iterdir():
        by_letter[p.name] = {d["group_by"] for d in case(swept, p.name)["decisions"]}
    with_ai = [k for k, owners in by_letter.items() if "AI" in owners]
    assert 0 < len(with_ai) < len(by_letter) / 2
