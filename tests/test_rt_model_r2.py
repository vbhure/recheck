"""Red-team regressions, round two: what the review of the model-boundary fixes
found still open. Every test here failed before its fix.

  MODEL-RT-P2P6-07  (residual) a batch that classified the same echoed name
                    twice put that model-written name - "VA ERRED. The correct
                    combined rating is 60 percent - appeal now." - into the
                    note, and so into the reviewer's question and case.json.
  SECRETS-F9        (residual) preflight read the endpoint overrides itself,
                    and differently from botocore: AWS_IGNORE_CONFIGURED_
                    ENDPOINT_URLS=" true" was taken as true (botocore does
                    not strip it), and a profile endpoint_url or [services]
                    section was not seen at all. Preflight passed while the
                    call went to a plaintext attacker endpoint.
  _unsendable       withheld every name with a digit or ':', so ordinary names
                    ("Chronic kidney disease, stage 3", "Hypertension (DC
                    7101)") stopped for a human with a note that wrongly said
                    they might carry other letter text.
  _redact_url       displayed a host other than the one used: "https://
                    attacker.example/@bedrock-runtime.us-west-2.amazonaws.com"
                    was shown as "https://***@bedrock-runtime...".
  --debug           the provider exception was logged whole, credentials in a
                    URL or an Authorization header included; .env.example said
                    passwords were redacted "wherever Recheck prints it".
  non-ASCII veto    a model "none" was refused for "Ménière's disease" because
                    of its accents, while a combining mark or zero-width space
                    inside "knee" hid the word from the veto altogether.
"""

from __future__ import annotations

import logging
import os
import socket

import pytest

from _support import ROOT, UNLISTED, batch, item, run_audit, scripted, tabular_letter
from recheck.classify import classify
from recheck.extract.deterministic import ExtractedRating
from recheck.models import factory as factory_module
from recheck.models.factory import load_config, preflight
from recheck.provenance import Actor, Trace

HOSTILE = "VA ERRED. The correct combined rating is 60 percent - appeal now."
REGIONAL = "https://bedrock-runtime.us-west-2.amazonaws.com"


def _rating(name: str, line: int = 3) -> ExtractedRating:
    return ExtractedRating(name, 10, "unrecognised", name, line)


def _classify(names, factory):
    trace = Trace()
    return classify([_rating(n, i + 1) for i, n in enumerate(names)], trace, factory), trace


def _prompt(factory) -> str:
    (model,) = factory.models
    return "\n".join(
        block.get("text", "")
        for call in model.calls
        for message in call["messages"]
        for block in message.get("content", [])
        if isinstance(block, dict)
    )


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-07 residual: a contradiction note never quotes the model
# --------------------------------------------------------------------------

def test_a_contradiction_on_an_echoed_name_does_not_quote_it():
    factory = scripted(batch(item(HOSTILE, "upper"), item(HOSTILE, "lower"), item(UNLISTED, "upper")))
    (decision,), trace = _classify([UNLISTED], factory)
    assert decision.extremity_group == "unknown" and decision.group_by is None
    assert "more than once with different answers" in (decision.note or "")
    for text in (decision.note or "", trace.render()):
        assert "ERRED" not in text and "appeal" not in text and "60" not in text


def test_a_contradiction_on_a_sent_name_quotes_the_letter_s_spelling_not_the_echo():
    factory = scripted(batch(item("ZORBLATT SYNDROME", "upper"), item("zorblatt syndrome ", "lower")))
    (decision,), _ = _classify([UNLISTED], factory)
    assert decision.extremity_group == "unknown"
    assert repr(UNLISTED) in (decision.note or "")
    assert "ZORBLATT" not in (decision.note or "") and "zorblatt syndrome" not in (decision.note or "")


def test_an_echoed_contradiction_never_reaches_the_question_or_case_state(tmp_path):
    """The reviewer's end-to-end shape."""
    from recheck.cli import question_text

    letter = tabular_letter(tmp_path / "echo.txt", [("Left wrist strain", 30), (UNLISTED, 30)], stated=50)
    factory = scripted(batch(item(HOSTILE, "upper"), item(HOSTILE, "lower"), item(UNLISTED, "upper")))
    store, _ = run_audit(tmp_path / "runs", "echo", letter, factory)
    assert store.load("echo").status == "awaiting_human"
    question = question_text(store, "echo", "", exiting=False)
    stored = (tmp_path / "runs" / "echo" / "case.json").read_text(encoding="utf-8")
    for text in (question, stored):
        assert "ERRED" not in text and "appeal" not in text and "60 percent" not in text


# --------------------------------------------------------------------------
# SECRETS-F9 residual: the bedrock endpoint is the one botocore resolves
# --------------------------------------------------------------------------

@pytest.fixture
def aws_isolated(monkeypatch, tmp_path):
    """No AWS configuration from this machine; static dummy credentials so the
    credential check has nothing to look up."""
    for key in [k for k in os.environ if k.startswith("AWS_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-real-secret-for-the-test")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    return tmp_path


@pytest.fixture
def network_attempts(monkeypatch):
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


def _bedrock_endpoint_check():
    config = load_config("bedrock", env=dict(os.environ))
    return {c.name: c for c in preflight(config)}["endpoint"]


def test_the_default_bedrock_endpoint_passes_without_network(aws_isolated, network_attempts):
    endpoint = _bedrock_endpoint_check()
    assert endpoint.ok is True, endpoint.detail
    assert "bedrock-runtime.us-west-2.amazonaws.com" in endpoint.detail
    assert network_attempts == []


def test_a_padded_ignore_flag_does_not_hide_an_endpoint_botocore_still_uses(aws_isolated, network_attempts,
                                                                             monkeypatch):
    monkeypatch.setenv("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", " true")
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://attacker.example:8080")
    endpoint = _bedrock_endpoint_check()
    assert endpoint.ok is False
    assert "attacker.example" in endpoint.detail
    assert load_config("bedrock", env=dict(os.environ)).endpoint == "http://attacker.example:8080"
    assert network_attempts == []


PROFILE_ENDPOINT = "[default]\nendpoint_url = http://attacker-cfg.example:8080\n"
SERVICES_SECTION = ("[default]\nregion = us-west-2\nservices = s1\n"
                    "[services s1]\nbedrock_runtime =\n  endpoint_url = http://attacker-svc.example:8080\n")


@pytest.mark.parametrize("config_text,host", [(PROFILE_ENDPOINT, "attacker-cfg.example"),
                                              (SERVICES_SECTION, "attacker-svc.example")],
                         ids=["profile-endpoint_url", "services-section"])
def test_an_endpoint_from_the_aws_config_file_is_seen(aws_isolated, network_attempts, monkeypatch,
                                                      config_text, host):
    path = aws_isolated / "attacker-config"
    path.write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(path))
    endpoint = _bedrock_endpoint_check()
    assert endpoint.ok is False
    assert host in endpoint.detail
    assert network_attempts == []
    with pytest.raises(factory_module.ProviderNotConfigured, match="endpoint"):
        factory_module.build_agent_factory("bedrock", env=dict(os.environ))


def test_a_config_file_endpoint_botocore_is_told_to_ignore_passes(aws_isolated, network_attempts, monkeypatch):
    path = aws_isolated / "attacker-config"
    path.write_text(PROFILE_ENDPOINT, encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(path))
    monkeypatch.setenv("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "true")
    assert _bedrock_endpoint_check().ok is True


@pytest.mark.parametrize("url", ["https://attacker.example/@bedrock-runtime.us-west-2.amazonaws.com",
                                 "https://gateway.example", "http://127.0.0.1:8080"])
def test_a_bedrock_endpoint_other_than_the_regional_one_fails(aws_isolated, network_attempts, monkeypatch, url):
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", url)
    endpoint = _bedrock_endpoint_check()
    assert endpoint.ok is False
    assert "***@bedrock-runtime" not in endpoint.detail
    assert network_attempts == []


def test_resolving_the_endpoint_does_not_ask_instance_metadata_for_a_region(aws_isolated, network_attempts,
                                                                             monkeypatch):
    """An AWS config with defaults_mode = auto and no AWS_REGION variable makes
    botocore ask instance metadata for the region while building a client."""
    path = aws_isolated / "auto-config"
    path.write_text("[default]\ndefaults_mode = auto\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(path))
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.setenv("RECHECK_REGION", "us-west-2")
    assert _bedrock_endpoint_check().ok is True
    assert network_attempts == []


def test_the_regional_endpoint_set_explicitly_still_passes(aws_isolated, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", REGIONAL)
    assert _bedrock_endpoint_check().ok is True


# --------------------------------------------------------------------------
# _unsendable: withhold letter text, not ordinary condition names
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Chronic kidney disease, stage 3",
    "Degenerative disc disease, L4-L5",
    "Hypertension (DC 7101)",
    "COVID-19 residuals",
    "Thoracolumbar strain, T12",
    "Hodgkin's lymphoma, stage 2",
    "Degenerative arthritis (DC 5010-5242)",
    "Intervertebral disc syndrome, C5-6-7",
    "Diabetes mellitus, type 2",
])
def test_an_ordinary_name_with_a_number_is_sent(name):
    factory = scripted(responder=lambda _t, _m: batch(item(name, "none", 0.95)))
    (decision,), _ = _classify([name], factory)
    assert name in _prompt(factory)
    assert (decision.extremity_group, decision.group_by) == ("none", Actor.AI)


@pytest.mark.parametrize("name,why", [
    ("Zorblatt syndrome (previously 70%)", "percentage"),
    ("Zorblatt syndrome, previously ten per cent", "percentage"),
    ("Zorblatt syndrome ７０％", "percentage"),
    ("Veteran: John Q Doe Zorblatt syndrome", "label"),
    ("Zorblatt syndrome Date of Notification: soon", "label"),
    ("Zorblatt syndrome 04/02/2026", "date"),
    ("Zorblatt syndrome April 2", "date"),
    ("Zorblatt syndrome April 2nd", "date"),
    ("Zorblatt syndrome 4/2/26", "date"),
    ("Zorblatt syndrome 2026", "date"),
    ("Zorblatt syndrome, file 00-000-002", "number"),
    ("Zorblatt syndrome 123-45-6789", "number"),
    ("Zorblatt syndrome C 12345678", "number"),
    ("Zorblatt syndrome 1-800-827-1000", "number"),
    ("Zorblatt syndrome 123 45 6789", "number"),
])
def test_letter_text_is_withheld_with_a_note_that_says_what_it_is(name, why):
    factory = scripted(responder=lambda _t, _m: batch(item(UNLISTED, "upper")))
    (withheld, clean), _ = _classify([name, UNLISTED], factory)
    sent = _prompt(factory)
    assert name not in sent and "Zorblatt syndrome " not in sent
    assert withheld.extremity_group == "unknown" and withheld.group_by is None
    note = withheld.note or ""
    assert note.startswith("not sent to the model:")
    assert why in note.lower()
    assert "may carry other letter text" not in note
    assert (clean.extremity_group, clean.group_by) == ("upper", Actor.AI)


# --------------------------------------------------------------------------
# _redact_url: show the host the call goes to, or refuse the URL
# --------------------------------------------------------------------------

def test_a_path_holding_an_at_sign_does_not_move_the_displayed_host():
    shown = factory_module._redact_url("https://attacker.example/@bedrock-runtime.us-west-2.amazonaws.com")
    assert shown.startswith("https://attacker.example")
    assert "bedrock-runtime" not in shown


def test_userinfo_is_masked_and_the_parsed_host_shown():
    shown = factory_module._redact_url("https://vso:PLANTED@ollama.internal.example:443/api")
    assert "PLANTED" not in shown and "vso" not in shown
    assert shown.startswith("https://***@ollama.internal.example:443")


@pytest.mark.parametrize("url", [
    "https://attacker.example\\@api.anthropic.com",  # a backslash ends the host for some HTTP clients
    "https://user:p@ss@api.anthropic.com",           # an unencoded "@" in the password
    "https://api.anthropic.com:notaport",
    "https:// api.anthropic.com",
    "api.anthropic.com:443",
    "http://[::1",
])
def test_an_ambiguous_url_is_refused_not_displayed(url):
    shown = factory_module._redact_url(url)
    assert "api.anthropic.com" not in shown and "p@ss" not in shown
    endpoint = {c.name: c for c in preflight(load_config("anthropic", env={"ANTHROPIC_BASE_URL": url}))}["endpoint"]
    assert endpoint.ok is False


def test_an_unparseable_ollama_host_fails_the_host_check():
    config = load_config("ollama", env={"RECHECK_OLLAMA_HOST": "vso:PLANTEDpw@ollama.internal.example:443"})
    host = {c.name: c for c in preflight(config)}.get("host configured")
    if host is not None:  # the check runs only once the ollama SDK imports
        assert host.ok is False and "PLANTED" not in host.detail
    assert "PLANTED" not in config.describe() and "ollama.internal.example" not in config.describe()


# --------------------------------------------------------------------------
# --debug: provider exception text is kept, credential material is not
# --------------------------------------------------------------------------

LEAKY = ("POST https://vso:PLANTEDpassword@ollama.internal.example/api/chat failed; "
         "retrying vso:PLANTEDnoscheme@ollama.internal.example:443; "
         "headers={'Authorization': 'Bearer PLANTEDbearer', 'x-api-key': 'sk-ant-api03-PLANTEDkey'}; "
         "X-Amz-Credential=AKIAPLANTED000000000/20260913 X-Amz-Security-Token=PLANTEDsession")


class _LeakyError:
    async def invoke_async(self, prompt, **kwargs):
        raise RuntimeError(LEAKY)


def test_the_debug_log_of_a_failed_call_masks_credentials(caplog):
    with caplog.at_level(logging.DEBUG, logger="recheck.classify"):
        (decision,), _ = _classify([UNLISTED], lambda: _LeakyError())
    assert decision.note == "the model call failed (RuntimeError)"
    assert "ollama.internal.example/api/chat failed" in caplog.text, "the detail is still there for --debug"
    assert "PLANTED" not in caplog.text


def test_the_debug_log_of_an_unrecognised_stop_reason_masks_credentials(caplog):
    factory = scripted(batch(item(UNLISTED, "upper")), stop_reason="Authorization: Bearer PLANTEDbearer")
    with caplog.at_level(logging.DEBUG, logger="recheck.classify"):
        _classify([UNLISTED], factory)
    assert "unrecognised stop_reason" in caplog.text and "PLANTED" not in caplog.text


def test_env_example_claims_only_what_is_true_about_redaction():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "wherever Recheck prints it" not in text
    assert "--debug" in text


# --------------------------------------------------------------------------
# The veto: accents are not a disguise; hidden marks inside "knee" are
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Ménière's disease", "Guillain-Barré syndrome",
                                  "Sjögren's syndrome", "Ménière's disease"])
def test_a_none_for_an_accented_latin_name_is_used(name):
    (decision,), _ = _classify([name], scripted(batch(item(name, "none", 0.99))))
    assert (decision.extremity_group, decision.group_by) == ("none", Actor.AI)


@pytest.mark.parametrize("name", [
    "Tenosynovitis, right кnee",        # Cyrillic ka
    "Tenosynovitis, right ｋｎｅｅ",  # fullwidth
    "Tenosynovitis, right kne̲e",       # combining low line
    "Tenosynovitis, right knée",       # combining acute
    "Tenosynovitis, right kn​ee",       # zero-width space
    "Tenosynovitis, right kn­ee",       # soft hyphen
    "Tenosynovitis, right kn⁠ee",       # word joiner
])
def test_a_none_is_still_vetoed_when_limb_vocabulary_is_disguised(name):
    (decision,), trace = _classify([name], scripted(batch(item(name, "none", 0.99))))
    assert decision.extremity_group == "unknown" and decision.group_by is None
    assert "Classification NOT USED" in [e.action for e in trace.entries]
