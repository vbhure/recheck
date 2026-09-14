"""deploy/agentcore/app.py: the AgentCore Runtime entrypoint, exercised without a server.

Every test runs Recheck's zero-model path (RECHECK_AGENTCORE_SCRIPTED) or stops
at the provider seam's preflight, so no test uses the network or a credential.
The entrypoint must reach the CLI's own pipeline, not a copy of it: the
figures, the question and the resume below are the ones `recheck audit` and
`recheck resume` produce for the same letters.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import socket
from types import SimpleNamespace

import pytest

from _support import LETTERS, ROOT

APP_PATH = ROOT / "deploy" / "agentcore" / "app.py"
FIXTURE = ROOT / "fixtures" / "classifications" / "07_clinical_terms.json"
SESSION = "test-session-000000000000000000000000001"


def _load_app():
    spec = importlib.util.spec_from_file_location("recheck_agentcore_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("RECHECK_AGENTCORE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RECHECK_AGENTCORE_SCRIPTED", str(FIXTURE))
    return _load_app()


def _letter(name: str) -> str:
    return (LETTERS / name).read_text(encoding="utf-8")


def test_a_letter_that_agrees_completes_without_a_model_call(app):
    r = app.handle({"letter_text": _letter("06_agrees.txt"), "case_id": "agrees"}, SESSION)
    assert r["status"] == "complete"
    assert (r["stated_combined"], r["recomputed_degree"]) == (70, 70)
    assert r["verdict"] == "NO DISCREPANCY FOUND"
    assert r["question"] is None
    assert r["model"]["model_call_made"] is False  # the lexicon settled every name
    assert "NO DISCREPANCY FOUND" in r["report"]
    assert all(x["extremity_group_decided_by"] == "lexicon" for x in r["ratings"])


def test_names_outside_the_lexicon_go_to_one_classifier_call(app):
    r = app.handle({"letter_text": _letter("07_clinical_terms.txt"), "case_id": "clinical"}, SESSION)
    assert r["status"] == "complete"
    assert (r["stated_combined"], r["recomputed_degree"], r["bilateral_applied"]) == (70, 80, True)
    assert r["verdict"] == "POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED"
    assert r["model"]["model_call_made"] is True
    assert r["model"]["provider"] == "scripted"
    by = {x["condition"]: (x["extremity_group"], x["extremity_group_decided_by"], x["side_decided_by"])
          for x in r["ratings"]}
    assert by["bronchial asthma"] == ("none", "lexicon", None)
    assert by["left cubital tunnel syndrome"] == ("upper", "AI (replayed fixture)", "letter")


def test_an_open_question_is_returned_and_a_later_invocation_resumes_it(app):
    first = app.handle({"letter_text": _letter("05_missing_side.txt"), "case_id": "side"}, SESSION)
    assert first["status"] == "awaiting_human"
    assert first["recomputed_degree"] is None
    assert first["possible_degrees"] == [70, 80]
    question = first["question"]
    assert question["answer_format"] == "1=<left|right|both|unknown>,2=<left|right|both|unknown>"
    assert "QUESTION FOR THE REVIEWER" in question["text"]
    assert "recheck resume" not in question["text"]  # the answer goes back through the runtime

    done = app.invoke({"case_id": "side", "answers": "1=left,2=right"}, SimpleNamespace(session_id=SESSION))
    assert done["status"] == "complete"
    assert (done["recomputed_degree"], done["bilateral_applied"]) == (80, True)
    assert done["reviewer_answers"] == {"1": "lower-left", "2": "lower-right"}
    sides = {x["index"]: x["side_decided_by"] for x in done["ratings"]}
    assert sides[1] == sides[2] == "reviewer"
    assert done["model"]["model_call_made"] is False


def test_a_rejected_answer_keeps_the_question_open(app):
    app.handle({"letter_text": _letter("05_missing_side.txt"), "case_id": "side"}, SESSION)
    r = app.handle({"case_id": "side", "answers": "1=up"}, SESSION)
    assert r["status"] == "awaiting_human"
    assert "not" in r["question"]["rejected_answer"]  # Recheck's reason, e.g. "'up' is not ..."
    again = app.handle({"case_id": "side", "answers": "1=left,2=right"}, SESSION)
    assert again["status"] == "complete" and again["recomputed_degree"] == 80


def test_a_case_belongs_to_its_runtime_session(app):
    app.handle({"letter_text": _letter("05_missing_side.txt"), "case_id": "side"}, SESSION)
    r = app.handle({"case_id": "side", "answers": "1=left,2=right"}, "another-session-00000000000000000000002")
    assert r["status"] == "rejected"
    assert "no case side in this runtime session" in r["error"]


def test_a_resume_with_a_different_letter_is_refused(app):
    app.handle({"letter_text": _letter("05_missing_side.txt"), "case_id": "side"}, SESSION)
    r = app.handle({"letter_text": _letter("06_agrees.txt"), "case_id": "side", "answers": "1=left,2=right"},
                   SESSION)
    assert r["status"] == "rejected"
    assert "differs" in r["error"]


def test_text_that_is_not_a_decision_letter_is_unparsed(app):
    r = app.handle({"letter_text": "Hello. This states no evaluations at all.", "case_id": "junk"}, SESSION)
    assert r["status"] == "unparsed"
    assert r["verdict"] == "COULD NOT READ THE LETTER"
    assert r["recomputed_degree"] is None


def test_an_existing_case_is_not_audited_over(app):
    app.handle({"letter_text": _letter("06_agrees.txt"), "case_id": "dup"}, SESSION)
    r = app.handle({"letter_text": _letter("06_agrees.txt"), "case_id": "dup"}, SESSION)
    assert r["status"] == "rejected" and "already exists" in r["error"]


def test_an_oversize_letter_is_rejected_before_anything_is_written(app, tmp_path):
    r = app.handle({"letter_text": "x" * (app.MAX_LETTER_BYTES + 1), "case_id": "big"}, SESSION)
    assert r["status"] == "rejected"
    assert "limit" in r["error"]
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize("payload, fragment", [
    ("just a string", "JSON object"),
    ({"letter_text": "a", "extra": 1}, "unknown payload keys"),
    ({"letter_text": 5}, "must be a string"),
    ({"letter_text": "   "}, "empty"),
    ({"answers": "1=left"}, "need the case_id"),
    ({"case_id": "c"}, "send letter_text"),
    ({"letter_text": "a", "case_id": "../escape"}, "case id"),
    ({"case_id": "c", "answers": "1=left" * 1000}, "over"),
])
def test_malformed_payloads_are_rejected(app, tmp_path, payload, fragment):
    r = app.handle(payload, SESSION)
    assert r["status"] == "rejected"
    assert fragment in r["error"]
    assert not (tmp_path / "store").exists()


#: The framework loggers the runtime holds at CRITICAL, as the CLI does without
#: --debug (recheck.cli._configure_logging, which app.py's __main__ calls).
#: Strands logs a model's tool input at DEBUG, and that holds condition names.
SILENCED = ("strands", "botocore", "urllib3", "httpx", "anthropic")


def test_the_letter_text_is_never_logged(app, caplog):
    caplog.set_level(logging.DEBUG)
    text = _letter("07_clinical_terms.txt")
    app.handle({"letter_text": text, "case_id": "quiet"}, SESSION)
    app.handle({"letter_text": text + "\nsecret-marker-zqx", "case_id": "quiet"}, SESSION)  # refused: exists
    app.handle({"letter_text": "secret-marker-zqx " * 50_000, "case_id": "big"}, SESSION)  # refused: size
    app.handle({"case_id": "quiet", "answers": "1=secret-marker-zqx"}, SESSION)  # refused: not waiting
    logged = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith(SILENCED))
    for fragment in ("cubital", "De Quervain", "asthma", "SYNTHETIC", "secret-marker-zqx"):
        assert fragment not in logged
    assert "case=quiet" in logged


def test_the_runtime_silences_the_framework_loggers():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "cli._configure_logging(False)" in source.split('if __name__ == "__main__":')[1]
    from recheck import cli

    levels = {name: logging.getLogger(name).level for name in SILENCED}
    try:
        cli._configure_logging(False)
        assert all(logging.getLogger(name).level == logging.CRITICAL for name in SILENCED)
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


@pytest.fixture
def no_aws(monkeypatch, tmp_path):
    for key in [k for k in os.environ if k.startswith("AWS_") or k.startswith("RECHECK_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RECHECK_AGENTCORE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    attempts: list[str] = []

    def refuse(*args, **kwargs):
        attempts.append(repr(args[:2]))
        raise OSError("network refused by test")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return attempts


def test_bedrock_without_credentials_is_refused_before_the_letter_is_written(no_aws, tmp_path):
    app = _load_app()
    r = app.handle({"letter_text": _letter("07_clinical_terms.txt"), "case_id": "live"}, SESSION)
    assert r["status"] == "error"
    assert "not ready" in r["error"] and "credentials" in r["error"]
    assert not list((tmp_path / "store").rglob("*.txt"))
    assert no_aws == [], f"the refusal used the network: {no_aws}"


def test_the_bedrock_model_id_comes_from_the_environment(no_aws, monkeypatch):
    import recheck.models.factory as factory

    monkeypatch.setenv("RECHECK_BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
    monkeypatch.setattr(factory, "preflight", lambda config: [])  # stand-in for a ready account
    app = _load_app()
    make, label, model = app._resolve_classifier()
    assert label == "bedrock:us.amazon.nova-lite-v1:0"
    assert model == {"provider": "bedrock", "model_id": "us.amazon.nova-lite-v1:0", "region": "us-east-1"}
    assert make.config.model_id == "us.amazon.nova-lite-v1:0"
    assert no_aws == []


def test_the_entrypoint_is_registered_with_the_agentcore_app(app):
    pytest.importorskip("bedrock_agentcore")
    assert app.app is not None
    assert app.app.handlers["main"] is app.invoke
    assert app.app._takes_context(app.invoke)  # the runtime session id reaches the handler
