"""What a reviewer reads.

Recheck is an audit aid. It cannot see the evidence or the criteria the VA
applied, so a difference is "a question to raise in review", never "the VA
is wrong"; a LOWER recomputation carries a warning that raising it could
invite a downward review; a result that depends on something nobody
established is UNDETERMINED, with what it could be and no recomputed figure;
and AI decisions replayed from a fixture say so, so a scripted demo can
never be mistaken for a model run.
"""

from __future__ import annotations

import json

import pytest

from _support import UNLISTED, UNLISTED_NERVE, main, require_lexicon_abstains, tabular_letter
from recheck.case import Case, CaseStore
from recheck.report import classifier_note, render, verdict
from recheck.sweep import Outcome, render_triage

FORBIDDEN = ("va is wrong", "va erred", "incorrect decision", "you are owed", "was wrong", "mistake by")


def _case(**fields) -> Case:
    base = dict(case_id="r", source_path="letter.txt", stated_combined=70, status="complete",
                recomputed_combined=75, recomputed_degree=80, bilateral_applied=True)
    base.update(fields)
    return Case(**base)


CASES = {
    "higher": _case(),
    "lower": _case(recomputed_degree=60, recomputed_combined=62, bilateral_applied=False),
    "agree": _case(recomputed_degree=70),
    "awaiting": _case(status="awaiting_human", recomputed_degree=None, recomputed_combined=None,
                      possible_degrees=[70, 80]),
    "undetermined": _case(status="undetermined", recomputed_degree=None, recomputed_combined=None,
                          possible_degrees=[40, 50], undetermined_reason="a single evaluation names both sides"),
    "unparsed": _case(status="unparsed", stated_combined=None, recomputed_degree=None),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_report_ever_says_the_va_is_wrong(name):
    text = render(CASES[name]).lower()
    for phrase in FORBIDDEN:
        assert phrase not in text
    assert "not a conclusion that the decision is wrong" in text, "the disclaimer is on every report"


def test_a_higher_recomputation_is_a_question_for_review():
    headline, explanation = verdict(CASES["higher"])
    assert headline == "POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED"
    assert "not a finding of error" in explanation
    assert "downward" not in explanation


def test_a_lower_recomputation_carries_the_downward_review_caution():
    headline, explanation = verdict(CASES["lower"])
    assert headline == "POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED"
    assert "lower" in explanation
    assert "prompt VA to review the rating downward" in explanation


def test_an_undetermined_report_shows_what_it_could_be_and_no_recomputed_figure():
    text = render(CASES["undetermined"])
    assert "UNDETERMINED - NOT COMPUTED" in text
    assert "possible final degrees" in text and "40% or 50%" in text
    assert "Recheck does not pick one" in text
    assert "recomputed final degree" not in text
    assert "NO DISCREPANCY FOUND" not in text


def test_a_report_awaiting_an_answer_is_not_a_result():
    text = render(CASES["awaiting"])
    assert "AWAITING YOUR ANSWER" in text and "70% or 80%" in text
    assert "recomputed final degree" not in text


def test_human_supplied_facts_are_named_as_such():
    headline, explanation = verdict(_case(human_answers={"2": "lower-left"}))
    assert "the facts you supplied for [2]" in explanation


@pytest.mark.parametrize(
    "label,fragment",
    [("scripted: fixtures/classifications/caseload.json", "no model was called"),
     ("none", "no classifier"),
     ("bedrock:global.anthropic.claude-haiku-4-5", "live model")],
)
def test_the_classifier_is_declared_on_every_report(label, fragment):
    assert fragment in classifier_note(_case(classifier=label))


def test_a_scripted_run_is_labelled_as_a_replayed_fixture_end_to_end(tmp_path):
    """Through the real CLI and the MAP fixture: the AI entry is labelled as
    replayed, and a term the fixture does not list is refused at confidence
    0.0 - left unknown, not called 'none'."""
    listed = f"{UNLISTED_NERVE}, left"
    unlisted = UNLISTED
    require_lexicon_abstains(listed, unlisted)
    fixture = tmp_path / "map.json"
    fixture.write_text(json.dumps({"anatomy": {"zorblatt nerve": "lower"}, "confidence": 0.9}), encoding="utf-8")
    letter = tabular_letter(tmp_path / "scripted.txt", [("Post-traumatic stress disorder", 60),
                            ("Right knee strain", 20), (listed, 10), (unlisted, 10)], stated=70)

    code, out, err = main("audit", letter, "--case", "s", "--scripted", "--classifications", fixture,
                          store=tmp_path / "runs")
    assert code == 0, out + err
    assert "[AI - replayed fixture]" in out
    assert "[AI]" not in out
    assert "no model was called" in out
    case = CaseStore(tmp_path / "runs").load("s")
    assert case.classifier.startswith("scripted:")
    decisions = case.load_decisions()
    assert (decisions[2].extremity_group, decisions[2].laterality) == ("lower", "left")
    assert decisions[3].extremity_group == "unknown"
    assert case.recomputed_degree == 80


def test_the_triage_warns_about_a_lower_recomputation(tmp_path):
    store = CaseStore(tmp_path / "runs")
    store.save(_case(case_id="low", recomputed_degree=60))
    store.save(_case(case_id="high"))
    text = render_triage(store, [Outcome("low", Outcome.COMPLETE), Outcome("high", Outcome.COMPLETE)])
    assert "recomputed LOWER than the letter - raising it could prompt a downward review" in text
    assert "questions to raise in review, not findings of error" in text
    for phrase in FORBIDDEN:
        assert phrase not in text.lower()
