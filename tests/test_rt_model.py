"""Red-team regressions: the model boundary and the provider seam.

Every test here reproduces a defect an adversarial red team found in the
release candidate, and failed before its fix. Model payloads still go through
the REAL Strands structured-output path (ScriptedModel streams the events a
provider streams); provider tests make no network call - where a network
attempt is the defect, the socket layer is replaced by a recorder that
refuses.

  MODEL-RT-P2P6-06  two structured answers in one message, or a repeated JSON
                    key, was silently resolved (first wins / last wins)
                    instead of discarded - including a 0.10 confidence
                    overwritten by 0.95 above the floor.
  MODEL-RT-P2P6-07  provider exception text and a free-form stop_reason were
  SECRETS-F5        printed verbatim in the trace, case.json and the
                    reviewer's question ("VA ERRED ... 60 percent").
  MODEL-RT-P2P6-08  the time budget relied on the model honouring
                    cancellation: a blocking model held the run past it, and
                    a model that swallowed CancelledError had its late
                    answer USED.
  MODEL-RT-P2P6-09  confidence true became 1.0 and "0.9" became 0.9.
  MODEL-RT-P2P6-10  no bound on what is sent: 3,000 names went out in one
                    call the schema could never answer, and one over-long
                    name voided the answers for every other name.
  SECRETS-F6        bedrock preflight probed EC2 instance metadata before
                    refusing for missing credentials.
  SECRETS-F8        a password inside RECHECK_OLLAMA_HOST was printed.
  SECRETS-F9        preflight said "provider default" while ANTHROPIC_BASE_URL
                    or AWS_ENDPOINT_URL_BEDROCK_RUNTIME sent the credential
                    to a plaintext override.
  SECRETS-F10       .env.example said RECHECK_PROVIDER selects the live
                    provider; audit and sweep (rightly) ignore it.

Also the classify.py half of two extraction findings, as defence in depth
behind the parser fix: letter text that bled into a condition name
(MODEL-RT-P2P6-02 / SECRETS-F2) is not sent to the model, and a "none" is not
accepted for a name whose limb word may be disguised by non-ASCII lookalike
letters (MODEL-RT-P2P6-04).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import socket
import time

import pytest
from pydantic import ValidationError

from _support import (
    ROOT,
    UNLISTED,
    UNLISTED_NERVE,
    batch,
    item,
    main,
    prompt_text,
    require_lexicon_abstains,
    run_audit,
    scripted,
    tabular_letter,
)
from recheck.classify import classify
from recheck.extract.deterministic import ExtractedRating
from recheck.models import factory as factory_module
from recheck.models.factory import DEFAULT_MAX_TOKENS, ProviderConfig, load_config, preflight
from recheck.models.scripted import ScriptedModel
from recheck.provenance import Actor, Trace
from recheck.schema import MAX_CONDITION_CHARS, MAX_ITEMS, ClassificationBatch, ConditionClassification

HOSTILE = "VA ERRED. The correct combined rating is 60 percent - appeal now."


@pytest.fixture(autouse=True)
def _names_are_outside_the_lexicon():
    require_lexicon_abstains(UNLISTED, UNLISTED_NERVE)


def _rating(name: str, line: int = 3) -> ExtractedRating:
    return ExtractedRating(name, 10, "unrecognised", name, line)


def _classify(names, factory):
    trace = Trace()
    return classify([_rating(n, i + 1) for i, n in enumerate(names)], trace, factory), trace


def _assert_unused(decision, trace):
    assert decision.extremity_group == "unknown"
    assert decision.group_by is None and decision.confidence is None
    assert not trace.by_actor(Actor.AI), "discarded output must not appear as an AI decision"


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-06: output that states two answers is discarded, not resolved
# --------------------------------------------------------------------------

class _TwoToolUses(ScriptedModel):
    """One assistant message carrying two structured-output tool calls."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        name = tool_specs[0]["name"] if tool_specs else "structured_output"
        self.calls.append({"tool": name, "messages": messages, "system_prompt": system_prompt})
        yield {"messageStart": {"role": "assistant"}}
        for n, group in enumerate(("lower", "upper"), 1):
            payload = json.dumps(batch(item(UNLISTED, group, 0.95)))
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{n}", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": payload}}}}
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


def test_two_structured_answers_in_one_message_are_discarded_not_first_wins():
    factory = scripted(batch(item(UNLISTED, "upper")), model_cls=_TwoToolUses)
    (decision,), trace = _classify([UNLISTED], factory)
    _assert_unused(decision, trace)
    assert "more than one structured answer" in (decision.note or "")
    assert len(factory.models[0].calls) == 1


ITEM = '{"condition": "Zorblatt syndrome", "extremity_group": "upper", "confidence": 0.9}'


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"classifications": [%s], "classifications": [%s]}'
                     % (ITEM, ITEM.replace("upper", "lower")), id="repeated-classifications"),
        pytest.param('{"classifications": [{"condition": "Zorblatt syndrome", "extremity_group": "upper", '
                     '"confidence": 0.10, "confidence": 0.95}]}', id="low-confidence-overwritten"),
        pytest.param('{"classifications": [{"condition": "Zorblatt syndrome", "extremity_group": "none", '
                     '"extremity_group": "lower", "confidence": 0.9}]}', id="repeated-group"),
    ],
)
def test_a_repeated_json_key_is_discarded_not_last_wins(raw):
    factory = scripted(raw)
    (decision,), trace = _classify([UNLISTED], factory)
    _assert_unused(decision, trace)
    assert "discarded" in (decision.note or "")
    assert len(factory.models[0].calls) == 1


def test_a_single_well_formed_answer_is_still_used():
    """The control for the two tests above: the new checks refuse only output
    that says two things."""
    (decision,), _ = _classify([UNLISTED], scripted(ITEM.join(['{"classifications": [', ']}'])))
    assert (decision.extremity_group, decision.group_by, decision.confidence) == ("upper", Actor.AI, 0.9)


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-07 / SECRETS-F5: only Recheck-authored text reaches the report
# --------------------------------------------------------------------------

class _RaisesMidCall:
    async def invoke_async(self, prompt, **kwargs):
        raise RuntimeError(HOSTILE)


def _raises_on_build():
    raise ConnectionError(HOSTILE)


@pytest.mark.parametrize("factory", [lambda: _RaisesMidCall(), _raises_on_build], ids=["mid-call", "on-build"])
def test_provider_exception_text_is_not_copied_into_the_note(factory, caplog):
    with caplog.at_level(logging.DEBUG, logger="recheck.classify"):
        (decision,), trace = _classify([UNLISTED], factory)
    _assert_unused(decision, trace)
    assert decision.note in ("the model call failed (RuntimeError)", "the model call failed (ConnectionError)")
    assert "ERRED" not in trace.render() and "60" not in trace.render()
    # The detail is kept for --debug only, where the operator can see it.
    assert HOSTILE in caplog.text


def test_a_free_form_stop_reason_is_not_copied_into_the_note():
    factory = scripted(batch(item(UNLISTED, "upper")), stop_reason="VA ERRED; correct rating 60 percent")
    (decision,), trace = _classify([UNLISTED], factory)
    _assert_unused(decision, trace)
    assert "stop_reason='other'" in (decision.note or "")
    assert "ERRED" not in trace.render() and "60" not in trace.render()


def test_endpoint_text_never_reaches_the_question_or_case_state(tmp_path):
    """The red team's end-to-end shape: a letter that raises a question, and a
    provider whose error message asserts VA error and names a rating."""
    from recheck.cli import question_text

    letter = tabular_letter(tmp_path / "exc.txt", [("Left wrist strain", 30), (UNLISTED, 30)], stated=50)

    def factory():
        return _RaisesMidCall()

    store, _ = run_audit(tmp_path / "runs", "exc", letter, factory)
    case = store.load("exc")
    assert case.status == "awaiting_human"
    question = question_text(store, "exc", "", exiting=False)
    stored = (tmp_path / "runs" / "exc" / "case.json").read_text(encoding="utf-8")
    for text in (question, stored):
        assert "ERRED" not in text and "appeal" not in text
    assert "the model call failed (RuntimeError)" in question


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-08: the budget holds whatever the model does with cancellation
# --------------------------------------------------------------------------

class _IgnoresCancellation(ScriptedModel):
    """Catches the cancellation, keeps working, then answers anyway."""

    async def stream(self, *args, **kwargs):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(1.5)
        async for event in super().stream(*args, **kwargs):
            yield event


def test_an_answer_that_arrives_after_the_budget_is_not_used():
    factory = scripted(batch(item(UNLISTED, "upper", 0.9)), model_cls=_IgnoresCancellation, timeout_s=0.3)
    start = time.monotonic()
    (decision,), trace = _classify([UNLISTED], factory)
    assert time.monotonic() - start < 1.2, "the run waited for a model that ignored cancellation"
    _assert_unused(decision, trace)
    assert "did not answer within 0.3s" in (decision.note or "")


def test_a_model_that_blocks_the_event_loop_does_not_hold_the_run():
    def responder(_tool, _messages):
        time.sleep(3)  # a synchronous client call inside an async stream
        return batch(item(UNLISTED, "upper", 0.9))

    factory = scripted(responder=responder, timeout_s=0.3)
    start = time.monotonic()
    (decision,), trace = _classify([UNLISTED], factory)
    assert time.monotonic() - start < 2.0, "a blocking model froze the timer"
    _assert_unused(decision, trace)
    assert "did not answer within 0.3s" in (decision.note or "")


def test_a_prompt_answer_inside_the_budget_is_still_used():
    (decision,), _ = _classify([UNLISTED], scripted(batch(item(UNLISTED, "lower", 0.9)), timeout_s=5))
    assert (decision.extremity_group, decision.group_by) == ("lower", Actor.AI)


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-09: confidence must be a JSON number
# --------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [True, False, "0.9", " 0.9 ", "1"])
def test_a_confidence_that_is_not_a_number_is_rejected(confidence):
    with pytest.raises(ValidationError):
        ConditionClassification(**item(UNLISTED, "upper", confidence))


@pytest.mark.parametrize("confidence", [True, "0.9"])
def test_a_non_numeric_confidence_through_strands_leaves_the_group_unknown(confidence):
    (decision,), trace = _classify([UNLISTED], scripted(batch(item(UNLISTED, "upper", confidence))))
    _assert_unused(decision, trace)


def test_a_json_integer_confidence_is_still_a_number():
    assert ConditionClassification(**item(UNLISTED, "upper", 1)).confidence == 1.0


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-10: nothing is sent that the model cannot validly answer
# --------------------------------------------------------------------------

def _names(n: int) -> list[str]:
    words = ["Alpha", "Bravo", "Delta", "Echo", "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima"]
    return [f"Zorblatt {words[i // 10]} {words[i % 10]} syndrome" for i in range(n)]


def test_more_names_than_one_answer_can_hold_are_not_sent():
    names = _names(MAX_ITEMS + 1)
    require_lexicon_abstains(*names)
    factory = scripted(batch(item(UNLISTED, "upper")))
    decisions, trace = _classify(names, factory)
    assert factory.models == [], "a call was made that the schema could never answer"
    assert all(d.extremity_group == "unknown" and d.group_by is None for d in decisions)
    assert f"more than the {MAX_ITEMS}" in (decisions[0].note or "")
    assert not trace.by_actor(Actor.AI)


def test_the_limit_counts_distinct_names_and_the_limit_itself_is_sent():
    names = _names(MAX_ITEMS) + [_names(1)[0].upper()]  # a repeat, in another case
    require_lexicon_abstains(*names)
    factory = scripted(responder=lambda _t, _m: batch(*[item(n, "lower") for n in names[:MAX_ITEMS]]))
    decisions, _ = _classify(names, factory)
    (model,) = factory.models
    listed = [line for line in prompt_text(model).splitlines() if line.startswith("- ")]
    assert len(listed) == MAX_ITEMS, "each distinct name is sent once"
    assert all((d.extremity_group, d.group_by) == ("lower", Actor.AI) for d in decisions)


def test_an_over_long_name_is_withheld_and_does_not_void_the_other_answers():
    long_name = "Zorblatt " + "syndrome " * 30
    assert len(long_name.strip()) > MAX_CONDITION_CHARS
    require_lexicon_abstains(long_name)
    factory = scripted(responder=lambda _t, _m: batch(item(UNLISTED, "upper")))
    (long_decision, short_decision), trace = _classify([long_name, UNLISTED], factory)
    assert long_name.strip() not in prompt_text(factory.models[0])
    assert long_decision.extremity_group == "unknown" and "not sent to the model" in (long_decision.note or "")
    assert (short_decision.extremity_group, short_decision.group_by) == ("upper", Actor.AI)


@pytest.mark.parametrize(
    "bled",
    [
        "SYNTHETIC File Number: 00-000-002 Date of Notification: April 2, 2026 DECISION Zorblatt syndrome",
        "Zorblatt syndrome (previously 70%; combined 90%)",
        "Your claim for Zorblatt syndrome, previously ten percent",
    ],
)
def test_a_name_carrying_other_letter_text_is_not_sent(bled):
    """Defence in depth for MODEL-RT-P2P6-02 / SECRETS-F2: whatever the parser
    captures, a name with digits, a percentage or a label is not a condition
    name, and the veteran's file number must not leave the machine in it."""
    require_lexicon_abstains(bled)
    factory = scripted(responder=lambda _t, _m: batch(item(UNLISTED, "upper")))
    (withheld, clean), trace = _classify([bled, UNLISTED], factory)
    text = prompt_text(factory.models[0])
    assert not any(ch.isdigit() for ch in text) and "percent" not in text.lower() and ":" not in text.split("\n", 1)[1]
    assert withheld.extremity_group == "unknown" and "not sent to the model" in (withheld.note or "")
    assert (clean.extremity_group, clean.group_by) == ("upper", Actor.AI)


def test_when_every_name_is_withheld_no_model_is_built():
    factory = scripted(batch(item(UNLISTED, "upper")))
    (decision,), _ = _classify(["Zorblatt syndrome, File Number 123-45-6789"], factory)
    assert factory.models == []
    assert decision.extremity_group == "unknown"


def test_the_output_cap_fits_the_largest_answer_the_schema_admits():
    """40 names at 200 characters is about 10,600 characters of JSON. At the
    old 1,024-token cap such a letter was sent and truncated every time."""
    largest = json.dumps(batch(*[item("x" * MAX_CONDITION_CHARS, "lower", 0.95)] * MAX_ITEMS))
    assert DEFAULT_MAX_TOKENS * 3 >= len(largest), "conservatively three characters per token"


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-04 (classify half): lookalike letters cannot slip a "none" past the veto
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Tenosynovitis, right кnee", "Tenosynovitis, right ｋｎｅｅ"])
def test_none_is_not_accepted_for_a_name_with_non_ascii_letters(name):
    require_lexicon_abstains(name)
    (decision,), trace = _classify([name], scripted(batch(item(name, "none", 0.99))))
    assert decision.extremity_group == "unknown" and decision.group_by is None
    assert "Classification NOT USED" in [e.action for e in trace.entries]


def test_a_plain_ascii_none_is_still_accepted():
    (decision,), _ = _classify([UNLISTED], scripted(batch(item(UNLISTED, "none", 0.99))))
    assert (decision.extremity_group, decision.group_by) == ("none", Actor.AI)


# --------------------------------------------------------------------------
# SECRETS-F6: no instance-metadata probe unless the operator opts in
# --------------------------------------------------------------------------

@pytest.fixture
def no_aws_credentials(monkeypatch, tmp_path):
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
                "AWS_EC2_METADATA_DISABLED", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))


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


def test_bedrock_preflight_refuses_without_contacting_instance_metadata(no_aws_credentials, network_attempts):
    checks = {c.name: c for c in preflight(load_config("bedrock", env={"AWS_REGION": "us-west-2"}))}
    assert network_attempts == [], f"preflight used the network: {network_attempts}"
    creds = checks["credentials resolvable"]
    assert creds.ok is False
    assert "instance metadata was not consulted" in creds.detail
    assert "AWS_EC2_METADATA_DISABLED=false" in creds.detail


def test_instance_metadata_is_consulted_when_the_operator_opts_in(no_aws_credentials, network_attempts, monkeypatch):
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "false")
    checks = {c.name: c for c in preflight(load_config("bedrock", env={"AWS_REGION": "us-west-2"}))}
    assert any("169.254.169.254" in a for a in network_attempts), "the opt-in was not honoured"
    assert checks["credentials resolvable"].ok is False


def test_the_refusal_is_documented_as_before_inference_not_before_network():
    doc = factory_module.ProviderNotConfigured.__doc__ or ""
    assert doc.startswith("Configuration is absent or invalid. Raised before any inference call.")


# --------------------------------------------------------------------------
# SECRETS-F8 / SECRETS-F9: what preflight shows about where the call goes
# --------------------------------------------------------------------------

PLANTED_HOST = "https://vso:PLANTEDollamaPASSWORD@ollama.internal.example:443"


def _with_sdk_importable(monkeypatch):
    import importlib

    real = importlib.import_module

    class _Stub:
        OllamaModel = AnthropicModel = BedrockModel = object

    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a: _Stub if name.startswith("strands.models.") else real(name, *a))


def test_a_password_in_the_ollama_host_is_not_printed(monkeypatch, tmp_path):
    _with_sdk_importable(monkeypatch)
    config = load_config("ollama", env={"RECHECK_OLLAMA_HOST": PLANTED_HOST})
    shown = config.describe() + " ".join(c.detail for c in preflight(config))
    assert "PLANTED" not in shown and "vso" not in shown
    assert "***@ollama.internal.example:443" in shown
    monkeypatch.setenv("RECHECK_OLLAMA_HOST", PLANTED_HOST)
    code, out, err = main("preflight", "--model", "ollama", store=tmp_path / "runs")
    assert "PLANTED" not in out + err


@pytest.mark.parametrize("host", ["vso:PLANTEDpw@ollama.internal.example:443", "https://a:PLANTED@b@host:1/x",
                                  "https://vso:PLANTED@host:notaport"])
def test_redaction_holds_for_awkward_hosts(host):
    assert "PLANTED" not in factory_module._redact_url(host)


def test_an_anthropic_base_url_override_is_shown_and_plaintext_fails(monkeypatch):
    config = load_config("anthropic", env={"ANTHROPIC_BASE_URL": "http://attacker.example:8080"})
    assert "provider default" not in config.describe()
    assert "attacker.example:8080" in config.describe()
    endpoint = {c.name: c for c in preflight(config)}["endpoint"]
    assert endpoint.ok is False and "ANTHROPIC_BASE_URL" in endpoint.detail


def test_a_bedrock_endpoint_override_is_shown_and_plaintext_fails():
    env = {"AWS_REGION": "us-west-2", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "http://attacker.example:8080"}
    config = load_config("bedrock", env=env)
    assert "attacker.example:8080" in config.describe()
    endpoint = {c.name: c for c in preflight(config)}["endpoint"]
    assert endpoint.ok is False and "AWS_ENDPOINT_URL_BEDROCK_RUNTIME" in endpoint.detail


@pytest.mark.parametrize(
    "env,ok",
    [
        ({"ANTHROPIC_BASE_URL": "https://gateway.example"}, True),
        ({"ANTHROPIC_BASE_URL": "http://127.0.0.1:18765"}, True),  # loopback: nothing crosses a network
        ({"ANTHROPIC_BASE_URL": "https://user:PLANTED@gateway.example"}, True),
        ({}, True),
        ({"ANTHROPIC_BASE_URL": "ftp://localhost"}, False),
        ({"ANTHROPIC_BASE_URL": "http://[::1"}, False),  # unparseable is refused, not a crash
        ({"ANTHROPIC_BASE_URL": "gateway.example:443"}, False),
    ],
)
def test_endpoint_check_accepts_https_loopback_and_the_default_only(env, ok):
    config = load_config("anthropic", env=env)
    endpoint = {c.name: c for c in preflight(config)}["endpoint"]
    assert endpoint.ok is ok
    assert "PLANTED" not in endpoint.detail + config.describe()


def test_a_plaintext_override_makes_the_factory_refuse(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-planted-for-the-test")
    with pytest.raises(factory_module.ProviderNotConfigured, match="endpoint"):
        factory_module.build_agent_factory("anthropic", env={"ANTHROPIC_BASE_URL": "http://attacker.example"})


def test_aws_ignore_configured_endpoint_urls_is_honoured():
    env = {"AWS_REGION": "us-west-2", "AWS_ENDPOINT_URL": "http://attacker.example",
           "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true"}
    assert load_config("bedrock", env=env).endpoint is None


# --------------------------------------------------------------------------
# SECRETS-F10: RECHECK_PROVIDER is documented as what it does
# --------------------------------------------------------------------------

def test_recheck_provider_is_documented_as_the_preflight_default_only(monkeypatch):
    from recheck.cli import _resolve_factory

    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    block = text[: text.index("RECHECK_PROVIDER=")].rsplit("\n\n", 1)[1]
    assert "Which live provider to use" not in block
    assert "preflight" in block and "--model" in block
    # ...and that is what the code does: a variable alone never starts paid calls.
    monkeypatch.setenv("RECHECK_PROVIDER", "anthropic")
    factory, _, label = _resolve_factory(argparse.Namespace(scripted=False, model=None))
    assert factory is None and label == "none"
