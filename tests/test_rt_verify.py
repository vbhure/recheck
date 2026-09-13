"""Verification of the final regression pass: what its fixes broke or missed.

Each test fails on the code before its fix (commit d29fddc), except the
controls, which say so.

  RG01-BYPASS-STATE-CODE  The salutation split matched its joining words in
          any case, so an address ending "Gary, IN", "Portland, OR" or
          "Apt A" read as "in", "or" and "a" and joined the name and address
          to the rating below: "Jane Right" gave a knee its side, 50% at exit 0.
  RG01-REFUSES-WRAPPED-LEADIN-SENTENCES  The split refused ordinary wrapped
          condition names inside a lead-in ("Service connection for right" /
          "Achilles tendonitis is granted ...") that the code before it read.
  RG01-MISLEADING-REFUSAL-REASON  A rating refused for starting mid-sentence
          was blamed on its name ("contains a long number").
  RG03-HYPHENATED-CODE  A restatement citing one part of a hyphenated code
          (DC 5260 for a row under DC 5003-5260) was refused as a second
          evaluation.
  RG11-REFINED-ZERO  A volunteered partial answer for a 0% condition, which is
          never asked, re-asked the unchanged question.
  RG05-FINISH-READY  A hand-edited 'ready' case whose facts leave the result
          open was computed and written 'complete' with one possible figure.
  RG11-FACT-COUNT  "2 fact(s) needed" for two conditions each missing a group
          and a side.
"""

from __future__ import annotations

import json
import re

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    main,
    require_lexicon_abstains,
    tabular_letter,
)
from recheck.case import CaseStore
from recheck.extract.deterministic import parse

HEAD = "DEPARTMENT OF VETERANS AFFAIRS\nRegional Office\n\nName: R. SYNTHETIC\nFile Number: 00-000-002\n\nDECISION\n\n"
TINNITUS = "Service connection for tinnitus is granted with an evaluation of 10 percent effective January 9, 2026.\n\n"
STATED = "Your combined evaluation for compensation is {} percent.\n"


def _read(text: str) -> list[tuple[str, int]]:
    extraction = parse(text)
    assert extraction.ok, extraction.unparsed_reason
    return [(r.condition, r.percent) for r in extraction.ratings]


# ==========================================================================
# RG01-BYPASS-STATE-CODE
# ==========================================================================

ADDRESSES = ["Jane Right\n12 Elm Street, Gary, IN", "Jane Right\n12 Elm Street, Portland, OR",
             "Jane Right\n12 Elm Street Apt A\nPortland OR", "Jane Right\n12 Elm Street Unit A\nSalem OR"]
BODY = ("\nKnee strain is continued as 20 percent disabling.\n"
        "Left knee strain is continued as 10 percent disabling.\n"
        "Post-traumatic stress disorder is continued as 20 percent disabling.\n\n" + STATED.format(40))


@pytest.mark.parametrize("address", ADDRESSES)
def test_an_address_ending_in_a_state_code_is_not_joined_to_a_rating(address):
    extraction = parse(address + BODY)
    assert not extraction.ok and not extraction.ratings
    assert "salutation" in extraction.unparsed_reason


def test_an_address_block_never_reaches_a_figure_through_the_cli(tmp_path):
    letter = tmp_path / "address.txt"
    letter.write_text(ADDRESSES[0] + BODY, encoding="utf-8")
    code, out, err = main("audit", letter, "--case", "address", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "50%" not in out and "Elm Street" not in out


# ==========================================================================
# RG01-REFUSES-WRAPPED-LEADIN-SENTENCES
# ==========================================================================

WRAPPED = [
    ("Service connection for diabetes mellitus associated with herbicide exposure (Agent\nOrange) is granted "
     "with an evaluation of 20 percent effective January 9, 2026.", 30),
    ("Service connection for diabetes mellitus type\nII is granted with an evaluation of 20 percent effective "
     "January 9, 2026.", 30),
    ("Service connection for residuals of left knee injury, status post\nACL reconstruction, is granted with an "
     "evaluation of 10 percent effective January 9, 2026.", 20),
    ("Service connection for right\nAchilles tendonitis is granted with an evaluation of 10 percent effective "
     "January 9, 2026.", 20),
    ("Service connection for right De\nQuervain's tenosynovitis is granted with an evaluation of 10 percent "
     "effective January 9, 2026.", 20),
    ("Evaluation of limitation of flexion, left knee,\nDiagnostic Code 5260, which is currently 10 percent "
     "disabling, is increased to 20 percent effective January 9, 2026.", 30),
    ("Service connection for Post Traumatic\nStress Disorder is granted with an evaluation of 50 percent "
     "effective January 9, 2026.", 60),
    ("Service connection for chronic fatigue syndrome due to service in the Persian\nGulf is granted with an "
     "evaluation of 20 percent effective January 9, 2026.", 30),
    ("Service connection for right shoulder\nAC joint separation is granted with an evaluation of 20 percent "
     "effective January 9, 2026.", 30),
    ("Service connection for chronic\nPTSD is granted with an evaluation of 50 percent effective January 9, 2026.",
     60),
    ("Service connection for left foot\nMorton's neuroma is granted with an evaluation of 10 percent effective "
     "January 9, 2026.", 20),
    ("Service connection for radiculopathy, right\nUpper extremity, is granted with an evaluation of 20 percent "
     "effective January 9, 2026.", 30),
    ("Service connection for bilateral hand\nRaynaud's syndrome is granted with an evaluation of 10 percent "
     "effective January 9, 2026.", 20),
]


@pytest.mark.parametrize("sentence, stated", WRAPPED, ids=lambda v: v.split("\n")[0][-24:] if isinstance(v, str) else "")
def test_a_condition_name_wrapped_once_inside_a_lead_in_reads_as_if_unwrapped(sentence, stated):
    wrapped = HEAD + sentence + "\n\n" + TINNITUS + STATED.format(stated)
    unwrapped = wrapped.replace(sentence, sentence.replace("\n", " "))
    assert _read(wrapped) == _read(unwrapped)


SUBJECT_THEN_SALUTATION = ("Your Claim for Compensation\nDear Ms Right,\n"
                           "Knee strain is continued as 20 percent disabling.\n"
                           "Left knee strain is continued as 10 percent disabling.\n"
                           "Post-traumatic stress disorder is continued as 20 percent disabling.\n\n"
                           + STATED.format(40))


def test_a_lead_in_bridges_one_break_so_a_subject_line_cannot_reach_a_salutation(tmp_path):
    """Control (passes before and after): it guards the one-break bridge. A
    first proposed fix bridged every break inside an open lead-in, and this
    letter then printed 50% at exit 0."""
    extraction = parse(SUBJECT_THEN_SALUTATION)
    assert not extraction.ok and not extraction.ratings
    letter = tmp_path / "subject.txt"
    letter.write_text(SUBJECT_THEN_SALUTATION, encoding="utf-8")
    code, out, err = main("audit", letter, "--case", "subject", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "50%" not in out and "Dear" not in out


# ==========================================================================
# RG01-MISLEADING-REFUSAL-REASON
# ==========================================================================

def test_a_rating_refused_for_starting_mid_sentence_is_not_blamed_on_its_name():
    letter = (HEAD + "Service connection for left knee strain is granted effective\n"
              "January 9, 2026, with an evaluation of 10 percent.\n\n" + TINNITUS + STATED.format(20))
    extraction = parse(letter)
    assert not extraction.ok
    assert "salutation" in extraction.unparsed_reason
    assert "long number" not in extraction.unparsed_reason


# ==========================================================================
# RG03-HYPHENATED-CODE
# ==========================================================================

HYPHENATED = """DEPARTMENT OF VETERANS AFFAIRS

RATING DECISION

  1. Post-traumatic stress disorder (DC 9411) ........ 30%
  2. Degenerative arthritis, right knee (DC 5003-5260) ....... 10%

COMBINED EVALUATION FOR COMPENSATION: 40%

REASONS FOR DECISION

Service connection for degenerative arthritis, right knee ({codes}) is granted with an evaluation of 10 percent.
"""


@pytest.mark.parametrize("codes", ["DC 5260", "Diagnostic Code 5003"])
def test_a_restatement_citing_part_of_a_hyphenated_code_is_read(codes):
    assert [p for _, p in _read(HYPHENATED.format(codes=codes))] == [30, 10]


def test_a_restatement_citing_the_same_codes_in_another_order_is_read():
    """Control (passes before and after)."""
    assert [p for _, p in _read(HYPHENATED.format(codes="DC 5260-5003"))] == [30, 10]


@pytest.mark.parametrize("codes", ["DC 5260-5261", "DC 5010"])
def test_a_statement_under_overlapping_or_other_codes_is_still_refused(codes):
    """Control (passes before and after)."""
    extraction = parse(HYPHENATED.format(codes=codes))
    assert not extraction.ok and "REASONS FOR DECISION" in extraction.unparsed_reason


# ==========================================================================
# RG11-REFINED-ZERO
# ==========================================================================

def test_a_volunteered_partial_answer_for_a_zero_percent_condition_does_not_re_ask(tmp_path):
    require_lexicon_abstains("Zorblatt syndrome")
    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "z2.txt", [("Bronchial asthma", 60), ("Degenerative arthritis of the knee", 20),
                                                 ("Limitation of motion of the knee", 10), ("Zorblatt syndrome", 0),
                                                 ("Tinnitus", 10)], stated=70)
    assert main("audit", letter, "--case", "z2", store=store)[0] == EXIT_AWAITING_HUMAN
    code, out, err = main("resume", "--case", "z2", "--answer", "1=unknown,2=unknown,3=upper", store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    case = CaseStore(store).load("z2")
    assert case.status == "undetermined" and case.human_answers["3"] == "upper-unknown"
    assert [e["action"] for e in case.trace].count("Question for a reviewer") == 1


def test_a_partial_answer_for_an_asked_condition_still_brings_a_follow_up(tmp_path):
    """Control (passes before and after)."""
    require_lexicon_abstains("Zorblatt syndrome")
    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "z7.txt", [("Bronchial asthma", 60), ("Left knee strain", 20),
                                                 ("Zorblatt syndrome", 10), ("Tinnitus", 10)], stated=70)
    assert main("audit", letter, "--case", "z7", store=store)[0] == EXIT_AWAITING_HUMAN
    code, out, _ = main("resume", "--case", "z7", "--answer", "2=lower", store=store)
    assert code == EXIT_AWAITING_HUMAN and "left, right, both, unknown" in out
    code, out, err = main("resume", "--case", "z7", "--answer", "2=right", store=store)
    assert code == EXIT_OK, out + err
    assert "Final degree of disability: 80%" in out


# ==========================================================================
# RG05-FINISH-READY
# ==========================================================================

def _decision(percent, group, side):
    return {"condition": f"{side} {group} {percent}", "percent": percent, "extremity_group": group,
            "laterality": side, "group_by": "DETERMINISTIC", "side_by": "DETERMINISTIC", "confidence": None,
            "note": None, "evidence": None}


def test_a_ready_case_whose_facts_leave_the_result_open_is_not_computed(tmp_path):
    store = tmp_path / "runs"
    letter = tabular_letter(tmp_path / "a.txt", [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=60)
    assert main("audit", letter, "--case", "a", store=store)[0] == EXIT_OK
    path = store / "a" / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["decisions"] = [_decision(20, "upper", "left"), _decision(10, "upper", "right"), _decision(30, "lower", "both")]
    raw["trace"] = [e for e in raw["trace"] if e["action"] not in ("Arithmetic", "Note", "Final degree of disability")]
    raw["status"] = "ready"
    raw.update(recomputed_combined=None, recomputed_degree=None, alternative_degree=None, bilateral_applied=False,
               bilateral_note=None)
    path.write_text(json.dumps(raw), encoding="utf-8")
    CaseStore(store).load("a")  # a ready case with these facts loads

    code, out, err = main("resume", "--case", "a", store=store)
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "could change its result" in err and err.count("could change its result") == 1
    case = CaseStore(store).load("a")
    assert case.status == "ready" and case.recomputed_degree is None


def test_the_compute_node_refuses_facts_that_leave_the_result_open(tmp_path):
    import asyncio

    from recheck.classify import Decision
    from recheck.graph import ComputeNode
    from recheck.provenance import Actor
    from strands.multiagent.base import Status

    store = CaseStore(tmp_path / "runs")
    letter = tabular_letter(tmp_path / "b.txt", [("Bronchial asthma", 60), ("Tinnitus", 10)], stated=70)
    assert main("audit", letter, "--case", "b", store=tmp_path / "runs")[0] == EXIT_OK
    case = store.load("b")
    decisions = [Decision("Knee strain", 20, "lower", "unknown", Actor.DETERMINISTIC, None, None, None, None),
                 Decision("Left knee strain", 20, "lower", "left", Actor.DETERMINISTIC, Actor.DETERMINISTIC,
                          None, None, None),
                 Decision("Bronchial asthma", 60, "none", "unknown", Actor.DETERMINISTIC, None, None, None, None)]
    case.status = "ready"
    case.recomputed_combined = case.recomputed_degree = case.alternative_degree = None
    case.trace = [e for e in case.trace if e["action"] not in ("Arithmetic", "Note", "Final degree of disability")]
    case.store_decisions(decisions)
    store.save(case)
    result = asyncio.run(ComputeNode(store, "b").invoke_async("compute"))
    assert result.status == Status.FAILED
    after = store.load("b")
    assert after.recomputed_degree is None and after.status == "ready"
    assert after.trace[-1]["action"] == "Arithmetic REFUSED"


# ==========================================================================
# RG11-FACT-COUNT
# ==========================================================================

def test_the_question_counts_facts_and_conditions(tmp_path):
    store = tmp_path / "runs"
    code, _, _ = main("audit", LETTERS / "08_clinical_terms_no_side.txt", "--case", "c8", store=store)
    assert code == EXIT_AWAITING_HUMAN
    entry = next(e for e in CaseStore(store).load("c8").trace if e["action"] == "Question for a reviewer")
    assert entry["value"] == "4 fact(s) needed for 2 condition(s)"
    assert re.search(r"\[1\].*\[2\]", entry["detail"])
