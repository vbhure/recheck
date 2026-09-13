"""A reviewer's answer is validated against the case, not trusted.

A reviewer supplies FACTS the letter does not establish - a side, an
extremity group - and only for the conditions Recheck asked about. The
grammar has no place for a number, and a fact the letter states is never
asked, so it cannot be overridden.

A rejected answer must leave the question OPEN: the same interrupt is raised
again, and a correct answer afterwards completes the case. Stranding a case
because someone mistyped is its own failure.

Defects regression-locked here:

  A1  Overriding the letter. At 0d018c9 a condition sent to a reviewer for
      any reason - low model confidence, say - was asked for its side, and
      the answer replaced the side even when the letter stated one. Now only
      missing facts are asked, and an answer for anything else is rejected.
  A2  A non-ASCII digit ("²=left") passed str.isdigit(), crashed int(), failed
      the graph mid-resume and left the case unable to take a correct answer.
"""

from __future__ import annotations

import pytest

from _support import answer, nodes_run, run_audit, tabular_letter
from recheck.classify import Decision
from recheck.graph import accepted_answers, build_graph, interrupt_id, outstanding_interrupt, parse_answers
from recheck.provenance import Actor


def D(percent, group, side, condition="c"):
    return Decision(condition, percent, group, side, None, None, None, None, None)


# 0 PTSD (none), 1 right knee (stated side), 2 knee (side unknown),
# 3 unlisted (group and side unknown), 4 unlisted, left (group unknown)
DECISIONS = [
    D(60, "none", "unknown"),
    D(20, "lower", "right"),
    D(10, "lower", "unknown"),
    D(10, "unknown", "unknown"),
    D(10, "unknown", "left"),
]
FULL = "2=left,3=upper-right,4=lower"


def test_accepted_answers_depend_on_what_is_missing():
    assert accepted_answers(DECISIONS[0]) == []
    assert accepted_answers(DECISIONS[1]) == []
    assert accepted_answers(DECISIONS[2]) == ["left", "right", "both", "unknown"]
    assert accepted_answers(DECISIONS[3]) == ["upper-left", "upper-right", "upper-both", "lower-left", "lower-right", "lower-both", "upper", "lower", "none", "unknown"]
    assert accepted_answers(DECISIONS[4]) == ["upper", "lower", "none", "unknown"]


def test_a_complete_answer_is_interpreted_keeping_established_facts():
    answers, problems = parse_answers(FULL, DECISIONS)
    assert problems == []
    assert answers == {2: ("lower", "left"), 3: ("upper", "right"), 4: ("lower", "left")}


def test_the_mapping_form_is_equivalent():
    assert parse_answers({"2": "left", "3": "upper-right", "4": "lower"}, DECISIONS) == \
        parse_answers(FULL, DECISIONS)


def test_unknown_keeps_what_the_letter_says_and_changes_nothing_else():
    answers, problems = parse_answers("2=unknown,3=unknown,4=unknown", DECISIONS)
    assert problems == []
    assert answers == {2: ("lower", "unknown"), 3: ("unknown", "unknown"), 4: ("unknown", "left")}


@pytest.mark.parametrize(
    "response,fragment",
    [
        # A1: the letter states the right knee's side; it is not asked.
        (FULL + ",1=left", "condition 1 was not asked about"),
        (FULL + ",0=upper", "condition 0 was not asked about"),
        # a side where a group is asked, and a group where a side is asked
        ("2=left,3=upper-right,4=left", "'left' is not an accepted answer for condition 4"),
        ("2=lower,3=upper-right,4=lower", "'lower' is not an accepted answer for condition 2"),
        # numbers have no place in the grammar
        ("2=10,3=upper-right,4=lower", "'10' is not an accepted answer"),
        ("2=20%,3=upper-right,4=lower", "'20%' is not an accepted answer"),
        ("2=left,3=upper-right,4=lower,80", "cannot parse '80'"),
        # duplicates, even when one of them is right
        ("2=left,2=right,3=upper-right,4=lower", "condition 2 answered more than once"),
        ("2=left,2=left,3=upper-right,4=lower", "condition 2 answered more than once"),
        # indices
        ("99=left,2=left,3=upper-right,4=lower", "no condition at index 99"),
        ("-1=left,2=left,3=upper-right,4=lower", "'-1' is not a condition index"),
        ("²=left,3=upper-right,4=lower", "'²' is not a condition index"),  # A2
        ("=left", "'' is not a condition index"),
        # incomplete or empty
        ("2=left", "no answer supplied for condition(s) [3, 4]"),
        ("", "empty answer"),
        ("garbage", "cannot parse 'garbage'"),
        (["2=left"], "answer must be text"),
        (None, "answer must be text"),
    ],
)
def test_invalid_answers_are_rejected_with_a_specific_reason(response, fragment):
    _, problems = parse_answers(response, DECISIONS)
    assert problems, response
    assert any(fragment in p for p in problems), problems


def test_the_parser_never_raises_on_hostile_text():
    """A2 generalised: the parser's output is a rejection, never an exception."""
    hostile = ["٣=left", "1²=left", "\x00=left", "2=​left", "2==left", "2=left=right",
               "①=left", "2" * 500 + "=left", "2=" + "x" * 10_000, ";;;,,,", "2=left\n3=right"]
    for response in hostile:
        answers, problems = parse_answers(response, DECISIONS)
        assert isinstance(answers, dict) and isinstance(problems, list)
        assert problems, f"hostile answer accepted: {response[:40]!r}"


def test_answers_are_case_and_whitespace_insensitive_but_nothing_else():
    answers, problems = parse_answers(" 2 = LEFT ; 3 = Upper-Right ; 4 = lower ", DECISIONS)
    assert problems == []
    assert answers[2] == ("lower", "left")


# --------------------------------------------------------------------------
# Through the graph: a rejection keeps the question open
# --------------------------------------------------------------------------

def _pair_case(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", [("Post-traumatic stress disorder", 60),
                            ("Right knee strain", 20), ("Limitation of motion of the knee", 10),
                            ("Tinnitus", 10)], stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    assert store.load("pair").status == "awaiting_human"
    return store


@pytest.mark.parametrize(
    "bad",
    ["1=left", "2=left,1=left", "2=left,2=right", "2=10", "²=left", {"2": "80"}, "nonsense"],
)
def test_a_rejected_answer_leaves_the_question_open_and_a_correct_one_completes(tmp_path, bad):
    store = _pair_case(tmp_path)
    result = answer(store, "pair", bad)

    case = store.load("pair")
    assert case.status == "awaiting_human"
    assert case.rejected_answer
    assert case.recomputed_degree is None
    assert [i.id for i in result.interrupts] == [interrupt_id("pair")]
    assert "Human answer REJECTED" in [e["action"] for e in case.trace]
    decisions = case.load_decisions()
    assert (decisions[1].laterality, decisions[1].side_by) == ("right", Actor.DETERMINISTIC)
    assert decisions[2].laterality == "unknown"
    # The restored session still holds the same question.
    graph = build_graph(store, "pair", case.source_path, None)
    assert outstanding_interrupt(graph).id == interrupt_id("pair")

    answer(store, "pair", "2=left")
    case = store.load("pair")
    assert case.status == "complete"
    assert case.rejected_answer is None
    assert case.human_answers == {"2": "lower-left"}
    assert case.recomputed_degree == 80
    assert nodes_run(store, "pair")[-2:] == ["assess", "compute"]


def test_answering_unknown_to_a_material_question_is_undetermined_not_a_guess(tmp_path):
    store = _pair_case(tmp_path)
    answer(store, "pair", "2=unknown")
    case = store.load("pair")
    assert case.status == "undetermined"
    assert case.possible_degrees == [70, 80]
    assert case.recomputed_degree is None
    assert "compute" not in nodes_run(store, "pair")
