"""Red-team regressions: graph semantics, materiality and 4.26(d) selection.

Each test names the red-team finding it locks, and fails on the release
candidate the red team attacked (af714cb):

  ARITH-F3, HUMAN-F6, MODEL-RT-P2P6-03
      Four ordinary conditions outside the lexicon pushed the enumeration
      over its cap, and the over-cap path crashed the assess node with
      IndexError, leaving the case stuck in 'classified'.
  ARITH-F4
      The 4.26(d) search left a single "bilateral" evaluation ALONE in the
      factor - the case M21-1 gives no factor - and reported 100% where the
      settled reading gives 90%.
  ARITH-F6, HUMAN-F12
      A lone "bilateral" evaluation that M21-1 settles (no factor) was still
      enumerated both ways, giving false UNDETERMINED results and questions
      whose every answer led to the same rating.
  ARITH-F8
      The completion cap did not bound time: a 12-row letter took 8.4 minutes.
  HUMAN-F2
      A resume that failed part-way left a session that re-printed the same
      question forever and accepted no answer.
  HUMAN-F11
      A group-only answer the question offers ("lower") ended the case as
      UNDETERMINED, with no way to give the side next.
  HUMAN-F13
      A JSON null answer became the fact "none" (not an arm or leg).
  SECRETS-F4, FILES-P4-06
      The Strands session path repeated the case id, overrunning the Windows
      path limit with a realistic store and file name, and a too-long path
      was found out only after a half-created 'open' case was left behind.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    actions,
    answer,
    main,
    nodes_run,
    require_lexicon_abstains,
    run_audit,
    tabular_letter,
)
import recheck.cfr.rating as rating
import recheck.graph as graph
from recheck.case import CaseStore
from recheck.cfr.rating import Paired, evaluate
from recheck.classify import Decision
from recheck.graph import build_graph, outstanding_interrupt, parse_answers
from recheck.materiality import MAX_COMPLETIONS, assess, evaluate_for_report

INVENTED = ("Zorblatt syndrome", "Quibble disorder", "Frobnitz condition", "Wibble syndrome", "Glorp syndrome")


def D(percent: int, group: str, side: str, condition: str = "c") -> Decision:
    return Decision(condition, percent, group, side, None, None, None, None, None)


def _case(store_root, case_id):
    return CaseStore(store_root).load(case_id)


# --------------------------------------------------------------------------
# ARITH-F3, HUMAN-F6, MODEL-RT-P2P6-03: the enumeration cap
# --------------------------------------------------------------------------

def test_four_conditions_outside_the_lexicon_complete_instead_of_crashing(tmp_path):
    """ARITH-F3's letter. Every completion gives 80%, so nobody is asked.

    Before: 7^4 x 2 readings = 4802 completions, over the 4096 cap, and the
    over-cap path raised IndexError inside the assess node."""
    names = ("Hypertension", "Gastroesophageal reflux disease", "Irritable bowel syndrome", "Hemorrhoids")
    require_lexicon_abstains(*names)
    letter = tabular_letter(tmp_path / "four.txt", [("Post-traumatic stress disorder", 70), (names[0], 10),
                            (names[1], 10), (names[2], 10), (names[3], 0)], stated=80)
    code, out, err = main("audit", letter, "--case", "four", store=tmp_path / "runs")
    assert "IndexError" not in out + err
    assert code == EXIT_OK, out + err
    assert "NO DISCREPANCY FOUND" in out
    case = _case(tmp_path / "runs", "four")
    assert (case.status, case.recomputed_degree) == ("complete", 80)


def test_a_zero_percent_unknown_does_not_multiply_the_enumeration():
    """A 0% evaluation can never enter the factor (4.26(c)); its options are one."""
    decisions = [D(70, "none", "unknown"), D(10, "unknown", "unknown"), D(10, "unknown", "unknown"),
                 D(10, "unknown", "unknown"), D(0, "unknown", "unknown")]
    m = assess(decisions)
    assert m.exhaustive and m.settled and m.possible == (80,)
    assert m.unknown == (1, 2, 3, 4), "still honestly unknown"


def test_four_invented_terms_and_a_knee_ask_the_reviewer_and_the_answers_finish_it(tmp_path):
    """HUMAN-F6's letter, which crashed with IndexError: it now asks, and answers complete it."""
    require_lexicon_abstains(*INVENTED[:4])
    rows = [("Post-traumatic stress disorder", 50), ("Right knee strain", 20)] + [(n, 10) for n in INVENTED[:4]]
    letter = tabular_letter(tmp_path / "many.txt", rows, stated=70)
    store_root = tmp_path / "runs"
    code, out, err = main("audit", letter, "--case", "many", store=store_root)
    assert "IndexError" not in out + err
    assert code == EXIT_AWAITING_HUMAN, out + err
    case = _case(store_root, "many")
    assert case.status == "awaiting_human" and case.possible_degrees == [70, 80]

    # Right knee 20 and a left leg 10: 28 + 2.8 -> 31; 50, 31, 10, 10, 10 -> 75 -> 80.
    code, out, err = main("resume", "--case", "many", "--answer", "2=none,3=none,4=none,5=lower-left",
                          store=store_root)
    assert code == EXIT_OK, out + err
    assert _case(store_root, "many").recomputed_degree == 80


def test_a_letter_over_the_cap_is_undetermined_not_an_exception(tmp_path):
    """Five invented terms: 7^5 completions, over the cap even now. No figure,
    no sampling, no crash - and the case reaches a real terminal state, which
    a later `show` can report (it was stuck in 'classified')."""
    require_lexicon_abstains(*INVENTED)
    assert 7 ** 5 > MAX_COMPLETIONS
    rows = [("Post-traumatic stress disorder", 50), ("Right knee strain", 20)] + [(n, 10) for n in INVENTED]
    letter = tabular_letter(tmp_path / "over.txt", rows, stated=70)
    store_root = tmp_path / "runs"
    code, out, err = main("audit", letter, "--case", "over", store=store_root)
    assert "IndexError" not in out + err
    assert code == EXIT_CANNOT_PROCEED
    assert "UNDETERMINED - NOT COMPUTED" in out
    case = _case(store_root, "over")
    assert case.status == "undetermined"
    assert case.recomputed_degree is None and case.possible_degrees == []
    assert "too many facts are unknown" in case.undetermined_reason
    assert "compute" not in nodes_run(CaseStore(store_root), "over")
    assert main("show", "--case", "over", store=store_root)[0] == EXIT_OK


def test_the_graph_phrase_helper_survives_an_empty_set():
    assert graph._or(()) == "not established"


# --------------------------------------------------------------------------
# Round 2 (review of MODEL-RT-P2P6-03): more arm and leg ratings than the
# 4.26(d) search is verified for
# --------------------------------------------------------------------------

SIDED_ELEVEN = [("Left knee strain", 10), ("Right knee strain", 10), ("Left ankle strain", 10),
                ("Right ankle strain", 10), ("Left hip strain", 10), ("Right hip strain", 10),
                ("Left shoulder strain", 10), ("Right shoulder strain", 10), ("Left elbow strain", 10),
                ("Right elbow strain", 10), ("Left wrist strain", 10)]
OVER_MEMBERS = {
    "thirteen_sided": SIDED_ELEVEN + [("Right wrist strain", 10), ("Left thumb strain", 10)],
    "eleven_sided_two_sideless": SIDED_ELEVEN + [("Knee strain", 10), ("Ankle strain", 10)],
}


@pytest.mark.parametrize("name", sorted(OVER_MEMBERS))
def test_more_arm_and_leg_ratings_than_the_search_is_verified_for_are_undetermined(tmp_path, name):
    """The engine refuses more than MAX_BILATERAL_MEMBERS members in the factor
    rather than truncating 4.26(d). That ValueError escaped the assess node:
    the case stayed 'classified', the report said INCOMPLETE, exit 3 - the
    dead state MODEL-RT-P2P6-03 was about, by another route."""
    assert len(OVER_MEMBERS[name]) == rating.MAX_BILATERAL_MEMBERS + 1
    letter = tabular_letter(tmp_path / f"{name}.txt", OVER_MEMBERS[name], stated=90)
    store_root = tmp_path / "runs"
    code, out, err = main("audit", letter, "--case", name, store=store_root)
    assert "ValueError" not in out + err
    assert code == EXIT_CANNOT_PROCEED
    assert "UNDETERMINED - NOT COMPUTED" in out
    case = _case(store_root, name)
    assert case.status == "undetermined"
    assert case.recomputed_degree is None and case.possible_degrees == []
    assert "verified for" in case.undetermined_reason
    assert "compute" not in nodes_run(CaseStore(store_root), name)
    assert main("show", "--case", name, store=store_root)[0] == EXIT_OK


def test_the_member_limit_is_refused_before_any_arithmetic(monkeypatch):
    """Found from the planned engine inputs, like the work budget - whether
    the letter states every side or leaves some to the enumeration."""
    calls = _count_engine_folds(monkeypatch, limit=0)
    thirteen = [D(10, g, s) for g, s in [("lower", "left"), ("lower", "right")] * 4
                + [("upper", "left"), ("upper", "right")] * 2 + [("upper", "left")]]
    eleven_and_two = thirteen[:11] + [D(10, "lower", "unknown"), D(10, "lower", "unknown")]
    for decisions in (thirteen, eleven_and_two):
        m = assess(decisions)
        assert not m.exhaustive and not m.settled and m.possible == ()
        assert "13 arm and leg disabilities" in m.limit
    assert calls[0] == 0


def test_the_compute_node_refuses_a_ready_case_over_the_member_limit(tmp_path):
    """Defence in depth: a case marked ready (a hand-edited case file) with
    13 arm and leg ratings raised ValueError inside compute. It now gets no
    figure and a terminal state."""
    import asyncio

    store = CaseStore(tmp_path / "runs")
    case = graph.open_case(store, "ready13", str(tmp_path / "none.txt"))
    case.store_decisions([D(10, g, s) for g, s in [("lower", "left"), ("lower", "right")] * 4
                          + [("upper", "left"), ("upper", "right")] * 2 + [("upper", "left")]])
    case.status = "ready"
    store.save(case)
    asyncio.run(graph.ComputeNode(store, "ready13").invoke_async("compute"))
    after = store.load("ready13")
    assert after.status == "undetermined"
    assert after.recomputed_degree is None and after.possible_degrees == []
    assert "verified for" in after.undetermined_reason


# --------------------------------------------------------------------------
# ARITH-F4: a both-sides evaluation left alone in the factor by 4.26(d)
# --------------------------------------------------------------------------

D_BOTH_ALONE = [("Post-traumatic stress disorder", 70), ("Obstructive sleep apnea", 50),
                ("Bilateral pes planus", 50), ("Limitation of flexion of the right knee", 10)]


def test_the_4_26_d_search_honours_the_strict_reading_of_m21_1():
    """With the knee in the factor: 50, 10 -> 55 + 5.5 -> 61; 70, 61, 50 -> 94 -> 90.
    Leaving the knee out leaves the pes planus alone in the factor (55):
    70, 55, 50, 10 -> 95 -> 100. M21-1 gives a lone both-sides evaluation no
    factor, so the strict reading is 90%."""
    ratings = [70, 50, 50, 10]
    paired = [Paired(50, "lower", "both"), Paired(10, "lower", "right")]
    assert evaluate(ratings, paired=paired, lone_both_in_factor=False).final_degree == 90
    assert evaluate(ratings, paired=paired, lone_both_in_factor=True).final_degree == 100


def test_a_result_that_depends_on_a_lone_both_sides_evaluation_in_the_factor_is_undetermined(tmp_path):
    """Before: COMPLETE, 100%, 'POTENTIAL DISCREPANCY', exit 0."""
    decisions = [D(70, "none", "unknown"), D(50, "none", "unknown"), D(50, "lower", "both"),
                 D(10, "lower", "right")]
    m = assess(decisions)
    assert m.possible == (90, 100) and m.reading_matters and not m.answers_matter

    letter = tabular_letter(tmp_path / "d.txt", D_BOTH_ALONE, stated=90)
    store_root = tmp_path / "runs"
    code, out, _ = main("audit", letter, "--case", "d", store=store_root)
    assert code == EXIT_CANNOT_PROCEED
    case = _case(store_root, "d")
    assert case.status == "undetermined"
    assert case.possible_degrees == [90, 100]
    assert case.recomputed_degree is None
    assert "POTENTIAL DISCREPANCY" not in out and "NO DISCREPANCY FOUND" not in out


def test_the_reported_derivation_uses_the_settled_reading():
    """70, 60, 20, bilateral pes planus 30 and a right knee 10 give 100% under
    either reading, so the case is settled - but the derivation shown under it
    put the pes planus alone in the factor (30 + 3), a calculation M21-1 does
    not make. The report's derivation now reaches 100% without the factor."""
    decisions = [D(70, "none", "unknown"), D(60, "none", "unknown"), D(20, "none", "unknown"),
                 D(30, "lower", "both"), D(10, "lower", "right")]
    assert assess(decisions).settled
    evaluation, _ = evaluate_for_report(decisions)
    assert evaluation.final_degree == 100
    lone = [m for m in evaluation.bilateral_members
            if m.side == "both" and [n.extremity for n in evaluation.bilateral_members].count(m.extremity) == 1]
    assert lone == [], [m.label() for m in evaluation.bilateral_members]


# --------------------------------------------------------------------------
# ARITH-F6, HUMAN-F12: enumerate the open reading only where M21-1 leaves it open
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rows,stated,expected",
    [
        # (a) a lone bilateral pes planus: no factor under M21-1. 30, 20 -> 44 -> 40.
        ([("Bilateral pes planus", 30), ("Post-traumatic stress disorder", 20)], 40, 40),
        # (b) two both-sides evaluations of the feet are each other's other
        #     disability: 30, 30 -> 51 + 5.1 -> 56; 56, 20 -> 65 -> 70.
        ([("Bilateral pes planus", 30), ("Bilateral plantar fasciitis", 30),
          ("Post-traumatic stress disorder", 20)], 70, 70),
        # (c) a sideless knee that, if both knees, would still be alone in the
        #     legs with only one arm rated: 50, 20, 10 -> 64 -> 60, no question.
        ([("Post-traumatic stress disorder", 50), ("Limitation of flexion of the knee", 20),
          ("Left shoulder strain", 10)], 60, 60),
        # (d) one rating on the whole letter. It was asked "90% or 100%".
        ([("Hemorrhoids", 90)], 90, 90),
        # a stated bilateral wrist with one leg rated (HUMAN-F12's bilat2.txt): 70.
        ([("Post-traumatic stress disorder", 60), ("Right knee strain", 20), ("Bilateral wrist strain", 10),
          ("Tinnitus", 10)], 70, 70),
    ],
    ids=["lone_both", "dup_both", "lone_both_q", "single_rating", "bilateral_wrist"],
)
def test_a_both_sides_reading_m21_1_settles_completes_without_a_question(tmp_path, rows, stated, expected):
    letter = tabular_letter(tmp_path / "l.txt", rows, stated=stated)
    store_root = tmp_path / "runs"
    code, out, err = main("audit", letter, "--case", "l", store=store_root)
    assert code == EXIT_OK, out + err
    case = _case(store_root, "l")
    assert (case.status, case.recomputed_degree) == ("complete", expected)
    assert "Question for a reviewer" not in actions(CaseStore(store_root), "l")


def test_answering_upper_both_where_m21_1_settles_it_completes(tmp_path):
    """HUMAN-F12: the question offered 'upper-both'; that answer gave UNDETERMINED."""
    require_lexicon_abstains(INVENTED[0])
    letter = tabular_letter(tmp_path / "u.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                            (INVENTED[0], 10), ("Tinnitus", 10)], stated=70)
    store, _ = run_audit(tmp_path / "runs", "u", letter)
    assert store.load("u").status == "awaiting_human"
    answer(store, "u", "2=upper-both")
    case = store.load("u")
    assert (case.status, case.recomputed_degree) == ("complete", 70)


# --------------------------------------------------------------------------
# ARITH-F8: time is bounded
# --------------------------------------------------------------------------

def _count_engine_folds(monkeypatch, limit: int) -> list[int]:
    """Count 4.26 candidate calculations; fail fast past `limit` rather than
    letting an unbounded enumeration run for minutes."""
    calls = [0]
    real = rating._with_factor

    def counted(others, members):
        calls[0] += 1
        if calls[0] > limit:
            raise AssertionError(f"more than {limit} bilateral-factor calculations")
        return real(others, members)

    monkeypatch.setattr(rating, "_with_factor", counted)
    rating._degree_with_factor.cache_clear()
    return calls


PERF_ROWS = [("Left knee strain", 10), ("Right knee strain", 10), ("Left shoulder strain", 10),
             ("Right shoulder strain", 10), ("Left ankle strain", 10), ("Right elbow strain", 10),
             ("Limitation of flexion of the knee", 10), ("Limitation of motion of the elbow", 10),
             ("Limitation of motion of the ankle", 10), ("Limitation of motion of the elbow", 10),
             ("Limitation of flexion of the knee", 10), ("Limitation of motion of the elbow", 10)]


def test_twelve_arm_and_leg_ratings_with_six_unstated_sides_finish_quickly(tmp_path, monkeypatch):
    """The red team's perf12_6 letter: 8.4 minutes of CPU, one engine run per
    completion and a 4.26(d) search of up to 4096 subsets in each. The
    searches' arithmetic repeats across completions; it is now done once."""
    calls = _count_engine_folds(monkeypatch, limit=20_000)
    letter = tabular_letter(tmp_path / "perf.txt", PERF_ROWS, stated=80)
    store, _ = run_audit(tmp_path / "runs", "perf", letter)
    case = store.load("perf")
    assert (case.status, case.recomputed_degree) == ("complete", 80)
    assert calls[0] < 20_000


def test_a_letter_past_the_work_budget_is_refused_before_any_arithmetic(monkeypatch):
    """Twelve distinct arm and leg ratings, six sides unstated: about 5.7 million
    4.26(d) subsets. Refused up front - not after spending the budget."""
    calls = _count_engine_folds(monkeypatch, limit=0)
    known = [D(10, "lower", "left"), D(20, "lower", "right"), D(30, "upper", "left"), D(40, "upper", "right"),
             D(50, "lower", "left"), D(60, "upper", "right")]
    unknown = [D(p, g, "unknown") for p, g in zip((10, 20, 30, 40, 50, 60), ("lower", "upper") * 3)]
    m = assess(known + unknown)
    assert not m.exhaustive and m.possible == () and "4.26(d)" in m.limit
    assert calls[0] == 0


# --------------------------------------------------------------------------
# HUMAN-F2: a failed resume must not strand the question
# --------------------------------------------------------------------------

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20), ("Limitation of motion of the knee", 10),
        ("Tinnitus", 10)]


def test_a_session_stranded_part_way_is_not_offered_as_an_open_question(tmp_path, monkeypatch):
    """Before: a run that failed part-way left an activated interrupt with no
    node waiting on it, and every later resume - right answer or garbage - ran
    no node, printed the same question again and exited 3, forever.

    The session is stranded here through the graph directly. Through `recheck
    resume` the same failure no longer strands anything: cmd_resume restores
    the case and its session, so the question stays answerable (see
    test_a_resume_that_raises_part_way_keeps_the_question_answerable below).
    """
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store_root = tmp_path / "runs"
    assert main("audit", letter, "--case", "pair", store=store_root)[0] == EXIT_AWAITING_HUMAN
    store = CaseStore(store_root)

    async def locked(self, task, invocation_state=None, **kwargs):
        raise PermissionError("case.json is locked by another program")

    with monkeypatch.context() as patch:
        patch.setattr(graph.AssessNode, "invoke_async", locked)
        restored = build_graph(store, "pair", str(letter), None)
        pending = outstanding_interrupt(restored)
        with pytest.raises(PermissionError):
            asyncio.run(restored.invoke_async(
                [{"interruptResponse": {"interruptId": pending.id, "response": {"2": "left"}}}]))

    case = store.load("pair")
    assert case.status == "awaiting_human" and case.recomputed_degree is None
    assert outstanding_interrupt(build_graph(store, "pair", case.source_path, None)) is None

    code, out, _ = main("resume", "--case", "pair", "--answer", "2=left", store=store_root)
    assert code == EXIT_CANNOT_PROCEED
    assert "QUESTION FOR THE REVIEWER" not in out
    assert store.load("pair").recomputed_degree is None


def test_a_resume_that_raises_part_way_keeps_the_question_answerable(tmp_path, monkeypatch):
    """The same failure through `recheck resume`: the case and its session are
    put back, so a later resume with the right answer finishes the case."""
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store_root = tmp_path / "runs"
    assert main("audit", letter, "--case", "pair", store=store_root)[0] == EXIT_AWAITING_HUMAN

    async def locked(self, task, invocation_state=None, **kwargs):
        raise PermissionError("case.json is locked by another program")

    with monkeypatch.context() as patch:
        patch.setattr(graph.AssessNode, "invoke_async", locked)
        code, _, err = main("resume", "--case", "pair", "--answer", "2=left", store=store_root)
    assert code == EXIT_CANNOT_PROCEED

    store = CaseStore(store_root)
    case = store.load("pair")
    assert case.status == "awaiting_human"
    assert outstanding_interrupt(build_graph(store, "pair", case.source_path, None)) is not None
    code, _, _ = main("resume", "--case", "pair", "--answer", "2=left", store=store_root)
    assert code == 0
    assert store.load("pair").recomputed_degree is not None


def test_a_normally_interrupted_session_still_offers_its_question(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    pending = outstanding_interrupt(build_graph(store, "pair", str(letter), None))
    assert pending is not None and pending.id == graph.interrupt_id("pair")


# --------------------------------------------------------------------------
# HUMAN-F11: a partial answer is asked about again, not ended
# --------------------------------------------------------------------------

def test_a_group_only_answer_is_followed_by_a_question_for_the_side(tmp_path):
    require_lexicon_abstains(INVENTED[0])
    letter = tabular_letter(tmp_path / "u.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                            (INVENTED[0], 10), ("Tinnitus", 10)], stated=70)
    store, _ = run_audit(tmp_path / "runs", "p", letter)

    result = answer(store, "p", "2=lower")
    case = store.load("p")
    assert case.status == "awaiting_human", "the question offered 'lower'; it must not end the case"
    assert case.possible_degrees == [70, 80]
    assert case.human_answers == {"2": "lower-unknown"}
    (interrupt,) = result.interrupts
    (asked,) = interrupt.reason["conditions"]
    assert (asked["index"], asked["missing"]) == (2, ["side"])
    assert asked["outcomes"] == {"lower-left": [80], "lower-right": [70], "lower-both": [80]}

    answer(store, "p", "2=left")
    case = store.load("p")
    assert (case.status, case.recomputed_degree) == ("complete", 80)
    assert case.human_answers == {"2": "lower-left"}
    decision = case.load_decisions()[2]
    assert (decision.extremity_group, decision.laterality) == ("lower", "left")


def test_a_round_that_establishes_nothing_still_ends_undetermined(tmp_path):
    """No loop: 'unknown' after a follow-up question ends the case, as before."""
    require_lexicon_abstains(INVENTED[0])
    letter = tabular_letter(tmp_path / "u.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                            (INVENTED[0], 10), ("Tinnitus", 10)], stated=70)
    store, _ = run_audit(tmp_path / "runs", "p", letter)
    answer(store, "p", "2=lower")
    answer(store, "p", "2=unknown")
    case = store.load("p")
    assert case.status == "undetermined" and case.recomputed_degree is None


# --------------------------------------------------------------------------
# HUMAN-F13: only text is an answer
# --------------------------------------------------------------------------

DECISIONS = [D(60, "none", "unknown"), D(20, "lower", "right"), D(10, "unknown", "unknown"), D(10, "none", "unknown")]


@pytest.mark.parametrize("response", [{"2": None}, {2: None}, {"2": True}, {"2": 2}, {"2": ["left"]}, {True: "left"}])
def test_a_mapping_answer_that_is_not_text_is_rejected(response):
    answers, problems = parse_answers(response, DECISIONS)
    assert problems
    assert answers == {}


def test_a_null_answer_through_the_graph_keeps_the_question_open(tmp_path):
    """Before: {'2': None} completed at 70% on the 'fact' that [2] is not an arm or leg."""
    require_lexicon_abstains(INVENTED[0])
    letter = tabular_letter(tmp_path / "u.txt", [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
                            (INVENTED[0], 10), ("Tinnitus", 10)], stated=70)
    store, _ = run_audit(tmp_path / "runs", "n", letter)
    answer(store, "n", {"2": None})
    case = store.load("n")
    assert case.status == "awaiting_human"
    assert case.recomputed_degree is None and case.human_answers == {}
    assert "must be text" in case.rejected_answer
    answer(store, "n", {"2": "lower-left"})
    assert store.load("n").recomputed_degree == 80


# --------------------------------------------------------------------------
# SECRETS-F4, FILES-P4-06: path length
# --------------------------------------------------------------------------

LONG_ID = "2026-08-19_rating_decision_narrative_file_00-000-006_" + "x" * 11
assert len(LONG_ID) == 64


def _store_with_deepest_path(tmp_path, length: int):
    """A store directory sized so this case's deepest session path is `length` characters."""
    probe = CaseStore(tmp_path / "s")
    pad = length - len(graph.deepest_session_path(probe, LONG_ID)) + 1
    if pad < 1:
        pytest.skip("the temporary directory is too deep for this test")
    return tmp_path / ("s" * pad)


def test_the_session_path_does_not_repeat_the_case_id(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", LONG_ID, letter)
    session = store.session_dir(LONG_ID)
    written = [p for p in session.rglob("*")]
    assert written, "the session was persisted"
    assert not any(LONG_ID in p.name for p in written)
    deepest = max(len(os.path.abspath(p)) for p in written)
    assert deepest <= len(graph.deepest_session_path(store, LONG_ID)), "the bound open_case checks is the real one"


def test_a_long_case_id_under_a_deep_store_is_audited(tmp_path):
    """SECRETS-F4. The deepest session path is 250 characters. With the case id
    repeated in it, it was 313 - over Windows' 260 without long-path support -
    and the audit failed with WinError 206 and a report blaming the letter."""
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store_root = _store_with_deepest_path(tmp_path, 250)
    code, out, err = main("audit", letter, "--case", LONG_ID, store=store_root)
    assert code == EXIT_AWAITING_HUMAN, out + err
    code, out, err = main("resume", "--case", LONG_ID, "--answer", "2=left", store=store_root)
    assert code == EXIT_OK, out + err


def test_a_path_too_long_for_the_platform_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    """FILES-P4-06. Before: case.json was written (status 'open'), the session
    write then failed, and a later sweep reported a stopped run."""
    monkeypatch.setattr(graph, "_path_limit", lambda: 259, raising=False)
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store_root = _store_with_deepest_path(tmp_path, 270)
    code, out, err = main("audit", letter, "--case", LONG_ID, store=store_root)
    assert code == EXIT_CANNOT_PROCEED
    assert "too long" in err
    assert not (store_root / LONG_ID).exists()
