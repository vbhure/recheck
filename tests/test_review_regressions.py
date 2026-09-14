"""Regression cover for defects found by reviewing the redesign itself.

Each test names the review finding it locks. They are grouped here, rather
than scattered, so the review's outcome can be read in one place.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket

import pytest

from _support import (
    EXIT_AWAITING_HUMAN,
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    main,
    run_audit,
    tabular_letter,
)
from recheck.case import CaseCorrupt, CaseStore
from recheck.classify import Decision
from recheck.extract.deterministic import parse
from recheck.materiality import assess, evaluate_established

L, R = "left", "right"


def D(percent, group, side):
    return Decision("c", percent, group, side, None, None, None, None, None)


# --------------------------------------------------------------------------
# F1: a side in an adjacent sentence must not reach the rated condition,
# tested through the graph, not only through derive_laterality().
# --------------------------------------------------------------------------

def test_a_side_in_the_previous_sentence_does_not_become_the_conditions_side(tmp_path):
    letter = tmp_path / "prose.txt"
    letter.write_text(
        "DECISION\n\n"
        "The examiner noted pain on the left side. Service connection for limitation of flexion of\n"
        "the knee is granted with an evaluation of 10 percent.\n\n"
        "Service connection for right knee strain is granted with an evaluation of 20 percent.\n\n"
        "Service connection for bronchial asthma is granted with an evaluation of 60 percent.\n\n"
        "Service connection for tinnitus is granted with an evaluation of 10 percent.\n\n"
        "Your combined evaluation for compensation is 70 percent.\n",
        encoding="utf-8",
    )
    store, _ = run_audit(tmp_path / "runs", "prose", letter)
    case = store.load("prose")
    knee = next(d for d in case.load_decisions() if d.condition.startswith("limitation of flexion"))
    assert knee.laterality == "unknown" and knee.side_by is None
    assert case.status == "awaiting_human"


# --------------------------------------------------------------------------
# F2: a second sweep over a store holding an OPEN question - independent of
# whatever the committed caseload happens to contain.
# --------------------------------------------------------------------------

def test_a_second_sweep_over_an_open_question_changes_nothing(tmp_path):
    folder = tmp_path / "letters"
    tabular_letter(folder / "pair.txt", [("Bronchial asthma", 60), ("Right knee strain", 20),
                                        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)], stated=70)
    tabular_letter(folder / "agrees.txt", [("Post-traumatic stress disorder", 50), ("Cervical strain", 30)],
                   stated=70)
    store_root = tmp_path / "runs"
    first = main("sweep", folder, store=store_root)
    before = (store_root / "pair" / "case.json").read_bytes()
    second = main("sweep", folder, store=store_root)
    assert first[0] == second[0] == EXIT_AWAITING_HUMAN
    assert "TypeError" not in second[1] + second[2]
    triage = lambda out: out[out.index("CASELOAD TRIAGE"):].replace("2 case(s) were already on file", "")
    assert triage(first[1]).split("----")[0] == triage(second[1]).split("----")[0]
    assert (store_root / "pair" / "case.json").read_bytes() == before
    code, _, _ = main("resume", "--case", "pair", "--answer", "2=left", store=store_root)
    assert code == EXIT_OK


# --------------------------------------------------------------------------
# One evaluation covering both extremities: M21-1 V.iv.1.C.4.b
# --------------------------------------------------------------------------

def test_a_single_bilateral_evaluation_alone_gets_no_factor():
    """"only apply the bilateral factor if there is/are an independently
    ratable condition in one of the involved extremities" - bilateral pes
    planus by itself is combined normally (the Board called adding the factor
    pyramiding, Citation Nr 0003457)."""
    decisions = [D(30, "lower", "both"), D(50, "none", "unknown")]
    assert assess(decisions).settled
    ev = evaluate_established(decisions)
    assert ev.bilateral_applied is False
    assert ev.final_degree == 70  # 50, 30 -> 65 -> 70


def test_a_single_bilateral_evaluation_joins_other_ratings_of_the_same_pair():
    """Citation Nr 1519449: bilateral feet 10, right knee 10, left knee 10 -> 27 + 2.7."""
    decisions = [D(10, "lower", "both"), D(10, "lower", R), D(10, "lower", L)]
    m = assess(decisions)
    assert m.settled
    ev = evaluate_established(decisions)
    assert ev.bilateral_subtotal_value == 30
    assert len(ev.bilateral_members) == 3


def test_the_case_m21_1_leaves_open_is_undetermined_when_it_matters(tmp_path):
    """Both uninvolved extremities rated, nothing else in the pair: M21-1 says
    the factor applies but not how the calculation runs. Bilateral pes planus
    10, left shoulder 20, right shoulder 30: the two readings give 50% and
    60%, so no figure is reported."""
    decisions = [D(10, "lower", "both"), D(20, "upper", L), D(30, "upper", R)]
    m = assess(decisions)
    assert m.possible == (50, 60)
    assert m.reading_matters and not m.answers_matter
    letter = tabular_letter(tmp_path / "open.txt", [("Bilateral pes planus", 10), ("Left shoulder strain", 20),
                                                    ("Right shoulder strain", 30)], stated=50)
    store, _ = run_audit(tmp_path / "runs", "open", letter)
    case = store.load("open")
    assert case.status == "undetermined"
    assert case.recomputed_degree is None
    assert case.possible_degrees == [50, 60]


def test_a_reviewer_can_answer_both_for_a_sideless_condition(tmp_path):
    letter = tabular_letter(tmp_path / "pf.txt", [("Bronchial asthma", 60), ("Right knee strain", 20),
                                                  ("Plantar fasciitis", 10), ("Tinnitus", 10)], stated=70)
    store_root = tmp_path / "runs"
    code, out, _ = main("audit", letter, "--case", "pf", store=store_root)
    assert code == EXIT_AWAITING_HUMAN and "both" in out
    code, out, _ = main("resume", "--case", "pf", "--answer", "2=both", store=store_root)
    case = CaseStore(store_root).load("pf")
    assert code == EXIT_OK
    assert case.load_decisions()[2].laterality == "both"
    assert case.recomputed_degree == 80  # joins the right knee under M21-1: 20, 10 -> 28 + 2.8 = 31


# --------------------------------------------------------------------------
# A numbered list with a gap is a partial list, and is refused
# --------------------------------------------------------------------------

def test_a_gap_in_numbered_rows_refuses_rather_than_dropping_an_evaluation():
    text = ("RATING DECISION\n\n"
            "  1. Bronchial asthma ........................ 60%\n"
            "  2. Limitation of flexion, right knee\n      (continued on next line) 20%\n"
            "  3. Limitation of flexion, left knee ......... 10%\n\n"
            "COMBINED EVALUATION FOR COMPENSATION: 70%\n")
    extraction = parse(text)
    assert not extraction.ok
    assert extraction.ratings == []
    assert "not contiguous" in extraction.unparsed_reason


# --------------------------------------------------------------------------
# Values inside case.json are validated, not only its shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,value",
    [("percent", "ten"), ("percent", 150), ("laterality", "north"), ("extremity_group", "torso"), ("confidence", 3)],
)
def test_a_tampered_decision_value_is_refused_as_corrupt(tmp_path, field, value):
    letter = tabular_letter(tmp_path / "a.txt", [("Bronchial asthma", 60), ("Right knee strain", 20),
                                                 ("Left knee strain", 10)], stated=70)
    store, _ = run_audit(tmp_path / "runs", "a", letter)
    path = tmp_path / "runs" / "a" / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["decisions"][1][field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CaseCorrupt):
        store.load("a")


# --------------------------------------------------------------------------
# A defect in the parser is not reported as an unreadable letter
# --------------------------------------------------------------------------

def test_a_parser_defect_surfaces_instead_of_blaming_the_document(tmp_path, monkeypatch):
    import recheck.graph as graph

    letter = tabular_letter(tmp_path / "a.txt", [("Bronchial asthma", 60)], stated=60)
    store = CaseStore(tmp_path / "runs")
    graph.open_case(store, "a", str(letter))

    def broken(_text):
        raise IndexError("a bug in the parser")

    monkeypatch.setattr(graph, "parse", broken)
    with pytest.raises(IndexError):
        asyncio.run(graph.ExtractNode(store, "a", str(letter)).invoke_async("audit"))


# --------------------------------------------------------------------------
# 4.26(d) took effect April 16, 2023: say so wherever it decides a result
# --------------------------------------------------------------------------

def test_a_result_decided_by_4_26_d_names_its_effective_date(tmp_path):
    # 90 and 30 combine to 93; with the two leg 10s in the factor: 90%; combined separately: 100%.
    letter = tabular_letter(tmp_path / "d.txt", [("Bronchial asthma", 90), ("Cervical strain", 30),
                                                 ("Right knee strain", 10), ("Left knee strain", 10)], stated=90)
    store, _ = run_audit(tmp_path / "runs", "d", letter)
    case = store.load("d")
    assert case.recomputed_degree == 100
    entries = [e for e in case.trace if e["action"] == "38 CFR 4.26(d) decides this result"]
    assert entries and "April 16, 2023" in entries[0]["detail"]


# --------------------------------------------------------------------------
# The time budget reaches each provider's own network client
# --------------------------------------------------------------------------

@pytest.fixture
def network_attempts(monkeypatch):
    """Record, and refuse, every name lookup and connection."""
    attempts: list[str] = []

    def refuse_lookup(host, *args, **kwargs):
        attempts.append(f"getaddrinfo {host}")
        raise socket.gaierror("network refused by test")

    def refuse_connect(address, *args, **kwargs):
        attempts.append(f"connect {address}")
        raise OSError("network refused by test")

    monkeypatch.setattr(socket, "getaddrinfo", refuse_lookup)
    monkeypatch.setattr(socket, "create_connection", refuse_connect)
    return attempts


@pytest.fixture
def no_aws_configuration(monkeypatch, tmp_path):
    """No AWS configuration or credentials from this machine, and no instance-metadata probe."""
    for key in [k for k in os.environ if k.startswith("AWS_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))


def test_the_bedrock_client_gets_the_time_budget_and_no_retries(no_aws_configuration, network_attempts):
    from recheck.models.factory import ProviderConfig, build_model

    model = build_model(ProviderConfig("bedrock", "global.anthropic.claude-haiku-4-5", region="us-west-2",
                                       timeout_s=7.0))
    config = model.client.meta.config
    assert config.read_timeout == 7.0
    assert config.retries["total_max_attempts"] == 1  # one attempt, no retry
    # RC-01: building the client must not go looking for credentials on the network
    # (instance metadata at 169.254.169.254); the suite is network-free.
    assert network_attempts == [], f"building the bedrock client used the network: {network_attempts}"


def test_the_anthropic_client_gets_the_time_budget_and_no_retries(monkeypatch):
    pytest.importorskip("anthropic")
    from recheck.models.factory import ProviderConfig, build_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder-not-a-key")
    model = build_model(ProviderConfig("anthropic", "claude-haiku-4-5", timeout_s=7.0))
    assert model.client.max_retries == 0
    assert float(getattr(model.client.timeout, "read", model.client.timeout) or 0) == 7.0
