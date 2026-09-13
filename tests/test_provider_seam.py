"""The live-provider seam, tested without a live provider.

No network, no credentials, no inference. The doubles stand in for "a
provider that stalls", "a provider that errors", "a provider that is not
installed" rather than for any particular vendor.

What matters most, asserted directly:
  - any provider failure leaves the extremity group UNKNOWN; nothing is
    fabricated, and facts the letter states (sides) are unaffected
  - the time budget configured for a provider is the one enforced
  - nothing here reads, stores, logs or prints a credential
  - preflight reports only checks it actually performed. An earlier version
    printed two unconditional PASS rows ("single-call contract", "prompt
    minimisation") that verified nothing.
"""

from __future__ import annotations

import asyncio
import re
import time

import pytest

from _support import UNLISTED, main, require_lexicon_abstains
from recheck.classify import classify
from recheck.extract.deterministic import ExtractedRating
from recheck.models import factory as factory_module
from recheck.models.factory import (
    DEFAULT_TIMEOUT_S,
    PROVIDERS,
    Check,
    ProviderConfig,
    ProviderNotConfigured,
    build_agent_factory,
    build_model,
    load_config,
    preflight,
)
from recheck.provenance import Actor, Trace

PLANTED = {
    "AWS_ACCESS_KEY_ID": "AKIAEXAMPLEEXAMPLE12",
    "AWS_SECRET_ACCESS_KEY": "s3cr3t-do-not-log-this-value-xyz",
    "AWS_SESSION_TOKEN": "session-token-do-not-log-abc123",
    "ANTHROPIC_API_KEY": "sk-ant-should-never-be-printed",
}


@pytest.fixture
def offline_aws(monkeypatch, tmp_path):
    """Keep boto3's credential chain local: no shared files, no instance metadata."""
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


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
    assert (config.provider, config.region) == ("bedrock", "eu-west-1")


def test_explicit_argument_beats_the_environment():
    assert load_config("ollama", env={"RECHECK_PROVIDER": "bedrock"}).provider == "ollama"


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "abc", "1e-40x"])
def test_invalid_timeout_is_rejected(bad):
    with pytest.raises(ProviderNotConfigured):
        load_config("bedrock", env={"RECHECK_TIMEOUT_S": bad})


def test_ollama_gets_a_local_host_by_default():
    assert load_config("ollama", env={}).host == "http://localhost:11434"


def test_describe_is_safe_to_print():
    text = load_config("bedrock", env={"RECHECK_REGION": "us-west-2"}).describe()
    assert "us-west-2" in text and "timeout" in text
    # Credential-shaped VALUES, not English words: an earlier version banned
    # the substring "token" and failed on "max_tokens".
    assert not re.search(r"AKIA[0-9A-Z]{12,}", text)
    assert not re.search(r"sk-[A-Za-z0-9_-]{12,}", text)


# --------------------------------------------------------------------------
# Preflight: real checks only, and silent about secrets
# --------------------------------------------------------------------------

def test_preflight_reports_only_checks_it_performs(offline_aws):
    checks = preflight(load_config("bedrock", env={"RECHECK_REGION": "us-west-2"}))
    assert all(isinstance(c, Check) for c in checks)
    names = [c.name for c in checks]
    assert "provider SDK importable" in names
    assert "single-call contract" not in names and "prompt minimisation" not in names
    # With the credential chain emptied, the credential check must FAIL; a
    # check that passes regardless of the environment is not a check.
    creds = next(c for c in checks if c.name == "credentials resolvable")
    assert creds.ok is False


def test_the_anthropic_credential_check_follows_the_environment(monkeypatch):
    config = load_config("anthropic", env={})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    absent = {c.name: c for c in preflight(config)}
    if not absent["provider SDK importable"].ok:
        pytest.skip("anthropic extra not installed; the credential check is not reached")
    monkeypatch.setenv("ANTHROPIC_API_KEY", PLANTED["ANTHROPIC_API_KEY"])
    present = {c.name: c for c in preflight(config)}
    assert absent["credentials resolvable"].ok is False
    assert present["credentials resolvable"].ok is True


def test_preflight_never_emits_credential_material(monkeypatch, offline_aws):
    """Even with credentials present, only the METHOD may be reported."""
    for key, value in PLANTED.items():
        monkeypatch.setenv(key, value)
    blob = " ".join(
        f"{c.name} {c.detail}"
        for provider in ("bedrock", "anthropic", "ollama")
        for c in preflight(load_config(provider, env={"RECHECK_REGION": "us-west-2"}))
    )
    for value in PLANTED.values():
        assert value not in blob


def test_the_preflight_command_prints_no_credential_and_no_hard_coded_pass(monkeypatch, offline_aws, tmp_path):
    for key, value in PLANTED.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RECHECK_REGION", "us-west-2")
    code, out, err = main("preflight", "--model", "bedrock", store=tmp_path / "runs")
    text = out + err
    for value in PLANTED.values():
        assert value not in text
    for value in PLANTED.values():
        assert value[:8] not in text, "not even a prefix of a credential"
    assert "single-call contract" not in text and "prompt minimisation" not in text
    assert "no inference call is made" in text
    assert code in (0, 3)


def test_missing_sdk_reports_the_install_command():
    config = ProviderConfig(provider="ollama", model_id="llama3.2:3b", host="http://x")
    sdk = {c.name: c for c in preflight(config)}["provider SDK importable"]
    if not sdk.ok:  # ollama is not installed in the verification environment
        assert "pip install" in sdk.detail and "strands-agents[ollama]" in sdk.detail


def test_bedrock_model_constructs_without_credentials_or_network(offline_aws):
    model = build_model(ProviderConfig(provider="bedrock", model_id="global.anthropic.claude-haiku-4-5",
                                       region="us-west-2"))
    config = model.get_config()
    assert config["model_id"] == "global.anthropic.claude-haiku-4-5"
    assert config["temperature"] == 0.0, "classification must be reproducible"


def test_factory_refuses_eagerly_when_not_ready(offline_aws):
    """A misconfiguration surfaces before a run begins, not mid-workflow."""
    with pytest.raises(ProviderNotConfigured) as excinfo:
        build_agent_factory("bedrock", env={"RECHECK_REGION": "us-west-2"})
    assert "not ready" in str(excinfo.value)
    assert "--scripted" in str(excinfo.value), "the refusal must point at the zero-cost path"


def test_the_configured_timeout_is_carried_by_the_factory(monkeypatch):
    """RECHECK_TIMEOUT_S must reach the classifier, which reads factory.timeout_s."""
    monkeypatch.setattr(factory_module, "preflight", lambda config: [Check("stub", True, "")])
    make = build_agent_factory("bedrock", env={"RECHECK_REGION": "us-west-2", "RECHECK_TIMEOUT_S": "12.5"})
    assert make.timeout_s == 12.5
    assert make.config.timeout_s == 12.5


# --------------------------------------------------------------------------
# Every provider failure leaves the group unknown
# --------------------------------------------------------------------------

LEFT_UNLISTED = f"{UNLISTED}, left"


class _Stalling:
    async def invoke_async(self, prompt, **kwargs):
        await asyncio.sleep(60)


class _Exploding:
    async def invoke_async(self, prompt, **kwargs):
        raise RuntimeError("provider returned HTTP 503")


def _stalling_factory():
    def make():
        return _Stalling()

    make.timeout_s = 0.3
    return make


def _unavailable():
    raise ProviderNotConfigured("provider went away")


@pytest.mark.parametrize(
    "agent_factory,label",
    [
        (_stalling_factory(), "timeout"),
        (lambda: _Exploding(), "api error"),
        (_unavailable, "unavailable"),
    ],
)
def test_provider_failure_leaves_the_group_unknown_and_invents_nothing(agent_factory, label):
    require_lexicon_abstains(LEFT_UNLISTED)
    ratings = [ExtractedRating(LEFT_UNLISTED, 10, "unrecognised", LEFT_UNLISTED, 2),
               ExtractedRating("Tinnitus", 10, "none", "Tinnitus", 3)]
    trace = Trace()
    start = time.monotonic()
    unlisted, tinnitus = classify(ratings, trace, agent_factory)
    assert time.monotonic() - start < 2.0, label
    assert unlisted.extremity_group == "unknown", label
    assert unlisted.group_by is None and unlisted.confidence is None
    # What the letter states is untouched by a provider failure.
    assert (unlisted.laterality, unlisted.side_by) == ("left", Actor.DETERMINISTIC)
    assert (tinnitus.extremity_group, tinnitus.group_by) == ("none", Actor.DETERMINISTIC)
    assert not trace.by_actor(Actor.AI)
    assert "Classification REJECTED" in [e.action for e in trace.entries]


def test_the_default_budget_applies_when_a_factory_declares_none(monkeypatch):
    monkeypatch.setattr("recheck.classify.DEFAULT_TIMEOUT_S", 0.2)
    trace = Trace()
    decision = classify([ExtractedRating(UNLISTED, 10, "unrecognised", UNLISTED, 1)], trace, lambda: _Stalling())[0]
    assert "within 0.2s" in (decision.note or "")


def test_the_zero_model_path_is_unaffected_by_provider_configuration(monkeypatch):
    """Provider availability never determines whether the core works."""
    monkeypatch.setenv("RECHECK_PROVIDER", "bedrock")
    ratings = [ExtractedRating("Right knee strain", 20, "lower", "Right knee strain", 1),
               ExtractedRating(UNLISTED, 10, "unrecognised", UNLISTED, 2)]
    knee, unlisted = classify(ratings, Trace(), agent_factory=None)
    assert (knee.extremity_group, knee.laterality) == ("lower", "right")
    assert unlisted.extremity_group == "unknown"
    assert "no classifier is configured" in (unlisted.note or "")
