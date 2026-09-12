"""The live-provider seam, tested without a live provider.

Every test here runs with no network, no credentials and no inference. The
doubles are provider-independent: they stand in for "a provider that stalls",
"a provider that errors", "a provider that is not installed" rather than for
any particular vendor.

Two properties matter most and are asserted directly:
  - a provider failure of ANY kind routes the affected conditions to human
    review, and never substitutes fabricated data
  - nothing in this module reads, stores, logs or prints a credential
"""

from __future__ import annotations

import threading
import time

import pytest

from recheck.classify import classify
from recheck.extract.deterministic import parse
from recheck.models.factory import (
    DEFAULT_TIMEOUT_S,
    PROVIDERS,
    BoundedAgent,
    Check,
    ProviderConfig,
    ProviderNotConfigured,
    ProviderTimeout,
    build_agent_factory,
    build_model,
    load_config,
    preflight,
)
from recheck.provenance import Actor, Trace

LETTER = "fixtures/letters/07_nerve_terminology.txt"


def _read(path: str) -> str:
    import pathlib

    return (pathlib.Path(__file__).parent.parent / path).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_no_provider_selected_is_a_clear_refusal():
    with pytest.raises(ProviderNotConfigured, match="no provider selected"):
        load_config(env={})


def test_unknown_provider_lists_the_known_ones():
    with pytest.raises(ProviderNotConfigured, match="unknown provider"):
        load_config("quantum-oracle", env={})


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_every_provider_has_a_usable_default_config(provider):
    config = load_config(provider, env={})
    assert config.provider == provider
    assert config.model_id
    assert config.timeout_s == DEFAULT_TIMEOUT_S


def test_environment_selects_the_provider():
    config = load_config(env={"RECHECK_PROVIDER": "bedrock", "RECHECK_REGION": "eu-west-1"})
    assert config.provider == "bedrock"
    assert config.region == "eu-west-1"


def test_explicit_argument_beats_the_environment():
    config = load_config("ollama", env={"RECHECK_PROVIDER": "bedrock"})
    assert config.provider == "ollama"


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "abc", "1e-40x"])
def test_invalid_timeout_is_rejected(bad):
    with pytest.raises(ProviderNotConfigured):
        load_config("bedrock", env={"RECHECK_TIMEOUT_S": bad})


def test_ollama_gets_a_local_host_by_default():
    assert load_config("ollama", env={}).host == "http://localhost:11434"


def test_describe_is_safe_to_print():
    """The summary line is logged on every live run, so it must not be able
    to carry credential material."""
    config = load_config("bedrock", env={"RECHECK_REGION": "us-west-2"})
    import re

    text = config.describe()
    assert "us-west-2" in text
    assert "timeout" in text
    # Assert on credential-shaped VALUES rather than on English words. An
    # earlier version of this test banned the substring "token" and failed on
    # "max_tokens" - a false positive. The property that matters is that no
    # secret material can appear, not that certain words are forbidden.
    assert not re.search(r"AKIA[0-9A-Z]{12,}", text)
    assert not re.search(r"sk-[A-Za-z0-9_-]{12,}", text)


# --------------------------------------------------------------------------
# Preflight: free, and silent about secrets
# --------------------------------------------------------------------------

def test_preflight_makes_no_inference_and_returns_checks():
    checks = preflight(load_config("bedrock", env={"RECHECK_REGION": "us-west-2"}))
    assert all(isinstance(c, Check) for c in checks)
    names = {c.name for c in checks}
    assert "provider SDK importable" in names
    assert "single-call contract" in names
    assert "prompt minimisation" in names


def test_preflight_never_emits_credential_material(monkeypatch):
    """Even with credentials present, only the METHOD may be reported."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLEEXAMPLE12")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s3cr3t-do-not-log-this-value-xyz")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-be-printed")

    blob = " ".join(
        f"{c.name} {c.detail}"
        for provider in ("bedrock", "anthropic")
        for c in preflight(load_config(provider, env={"RECHECK_REGION": "us-west-2"}))
    )
    assert "AKIAEXAMPLEEXAMPLE12" not in blob
    assert "s3cr3t-do-not-log-this-value-xyz" not in blob
    assert "sk-ant-should-never-be-printed" not in blob


def test_missing_sdk_reports_the_install_command():
    config = ProviderConfig(provider="ollama", model_id="llama3.2:3b", host="http://x")
    checks = {c.name: c for c in preflight(config)}
    sdk = checks["provider SDK importable"]
    if not sdk.ok:  # ollama is not installed in the verification environment
        assert "pip install" in sdk.detail
        assert "strands-agents[ollama]" in sdk.detail


def test_bedrock_model_constructs_without_credentials_or_network():
    """Construction must not be a billable or networked operation."""
    model = build_model(
        ProviderConfig(provider="bedrock", model_id="global.anthropic.claude-haiku-4-5",
                       region="us-west-2")
    )
    config = model.get_config()
    assert config["model_id"] == "global.anthropic.claude-haiku-4-5"
    assert config["temperature"] == 0.0, "classification must be reproducible"


def test_factory_refuses_eagerly_when_not_ready():
    """A misconfiguration must surface before a run begins, not mid-workflow."""
    with pytest.raises(ProviderNotConfigured) as excinfo:
        build_agent_factory("bedrock", env={"RECHECK_REGION": "us-west-2"})
    message = str(excinfo.value)
    assert "not ready" in message
    assert "--scripted" in message, "the refusal must point at the zero-cost path"


# --------------------------------------------------------------------------
# BoundedAgent: time, not just turns
# --------------------------------------------------------------------------

class StallingAgent:
    """A provider that accepts the call and then never answers."""

    def __init__(self) -> None:
        self.cancelled = threading.Event()

    def __call__(self, prompt, **kwargs):
        signal = kwargs.get("cancel_signal")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if signal is not None and signal.is_set():
                self.cancelled.set()
                raise RuntimeError("cancelled")
            time.sleep(0.02)
        return "never reached"


class ExplodingAgent:
    def __call__(self, prompt, **kwargs):
        raise RuntimeError("provider returned HTTP 503")


def test_a_stalling_provider_raises_timeout_rather_than_hanging():
    agent = StallingAgent()
    bounded = BoundedAgent(agent, timeout_s=0.25)
    start = time.time()
    with pytest.raises(ProviderTimeout, match="did not answer within"):
        bounded("classify these")
    assert time.time() - start < 3.0, "the wall-clock budget was not enforced"


def test_timeout_signals_cancellation_to_the_underlying_call():
    """Abandoning the future is not enough; the work must be told to stop."""
    agent = StallingAgent()
    bounded = BoundedAgent(agent, timeout_s=0.2)
    with pytest.raises(ProviderTimeout):
        bounded("classify these")
    assert agent.cancelled.wait(timeout=2.0), "cancel_signal was never set"


def test_a_fast_provider_passes_through_untouched():
    bounded = BoundedAgent(lambda prompt, **kw: "done", timeout_s=5.0)
    assert bounded("x") == "done"


def test_provider_errors_propagate_unchanged():
    bounded = BoundedAgent(ExplodingAgent(), timeout_s=5.0)
    with pytest.raises(RuntimeError, match="503"):
        bounded("x")


# --------------------------------------------------------------------------
# Every provider failure ends in human review, never in fabricated data
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "agent_double,label",
    [
        (lambda: BoundedAgent(StallingAgent(), timeout_s=0.2), "timeout"),
        (lambda: BoundedAgent(ExplodingAgent(), timeout_s=5.0), "api error"),
        (lambda: (_ for _ in ()).throw(ProviderNotConfigured("gone")), "unavailable"),
    ],
)
def test_provider_failure_routes_to_human_review(agent_double, label):
    extraction = parse(_read(LETTER))
    trace = Trace()
    decisions = classify(extraction.ratings, trace, agent_double)

    nerve = [d for d in decisions if "nerve" in d.condition]
    assert nerve, "fixture must contain conditions the lexicon abstains on"
    for decision in nerve:
        assert decision.needs_human, f"{label} must require a human"
        assert decision.laterality == "unknown"
        assert decision.extremity_group == "none"
        assert decision.decided_by is Actor.DETERMINISTIC
    assert any("FAILED" in e.action or "REJECTED" in e.action for e in trace.entries)


def test_a_failed_provider_does_not_invent_a_classification():
    """The specific thing that must never happen."""
    extraction = parse(_read(LETTER))
    decisions = classify(
        extraction.ratings, Trace(), lambda: BoundedAgent(ExplodingAgent(), timeout_s=5.0)
    )
    assert not any(d.safe_for_pairing for d in decisions if "nerve" in d.condition)
    assert all(d.confidence is None for d in decisions), "no confidence may be fabricated"


def test_the_zero_model_path_is_unaffected_by_provider_configuration(monkeypatch):
    """Provider availability must never determine whether the core works."""
    monkeypatch.setenv("RECHECK_PROVIDER", "bedrock")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    extraction = parse(_read(LETTER))
    decisions = classify(extraction.ratings, Trace(), agent_factory=None)
    assert len(decisions) == len(extraction.ratings)
    assert all(d.decided_by is Actor.DETERMINISTIC for d in decisions)
