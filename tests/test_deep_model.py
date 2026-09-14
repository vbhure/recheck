"""Deep review, model lane: the AI / model boundary under hostile output and configuration.

Regression tests for what the review found (each fails on d29fddc), then
proofs of what held: AI output reaches nothing but an extremity group and a
confidence, and the zero-model path uses no network.

No live provider is called. Providers are replaced by Strands Models that
raise or stream what a hostile endpoint could, or by constructor stubs.
"""

from __future__ import annotations

import importlib
import socket
import sys
import time
import types
from dataclasses import asdict

import pytest
from strands.tools.structured_output import convert_pydantic_to_tool_spec
from strands.types.exceptions import ModelThrottledException

from _support import (
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    UNLISTED,
    batch,
    case_json,
    item,
    main,
    require_lexicon_abstains,
    require_no_extremity_markers,
    run_audit,
    scripted,
    tabular_letter,
)
from recheck.classify import classify, derive_laterality
from recheck.extract.deterministic import ExtractedRating
from recheck.materiality import assess as assess_materiality, evaluate_for_report
from recheck.models import factory as factory_module
from recheck.models.factory import Check, ProviderNotConfigured, build_agent_factory, build_model, load_config
from recheck.models.scripted import ScriptedModel
from recheck.provenance import Actor, Trace
from recheck.schema import ClassificationBatch


def _unlisted_rating(name: str = UNLISTED) -> ExtractedRating:
    return ExtractedRating(name, 20, "unrecognised", name, 3)


# --------------------------------------------------------------------------
# MODEL-1: the production Agent retried a throttled model call itself
# --------------------------------------------------------------------------

class _Throttled(ScriptedModel):
    """A provider answering every request with a throttle (Bedrock ThrottlingException, Anthropic 429)."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls.append({"tool": None, "messages": messages, "system_prompt": system_prompt})
        raise ModelThrottledException("Rate exceeded")
        yield  # pragma: no cover - makes this an async generator


def test_a_throttled_provider_is_called_once_per_letter_not_retried_by_the_agent(monkeypatch):
    """One call per letter means one attempt. The provider clients were built
    with retries off, but the Strands Agent build_agent_factory made kept its
    default ModelRetryStrategy (six attempts, 4 s, 8 s, 16 s apart): a
    throttled letter made 2 calls in a 6 s budget, 4 in the default 30 s, and
    was reported as "did not answer within 30s"."""
    models: list[_Throttled] = []

    def throttled_model(config):
        models.append(_Throttled(payload=batch(item(UNLISTED, "upper"))))
        return models[-1]

    monkeypatch.setattr(factory_module, "preflight", lambda config: [Check("stub", True, "")])
    monkeypatch.setattr(factory_module, "build_model", throttled_model)
    make = build_agent_factory("bedrock", env={"RECHECK_REGION": "us-west-2", "RECHECK_TIMEOUT_S": "6"})

    start = time.monotonic()
    decision = classify([_unlisted_rating()], Trace(), make)[0]
    elapsed = time.monotonic() - start

    assert sum(len(m.calls) for m in models) == 1, "a throttled call was retried"
    assert decision.extremity_group == "unknown" and decision.group_by is None
    assert decision.note == "the model call failed (ModelThrottledException)"
    assert elapsed < 4.0, f"a refused call held the letter for {elapsed:.1f}s"


# --------------------------------------------------------------------------
# MODEL-2: non-finite or truncating numbers from the environment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["RECHECK_TIMEOUT_S", "RECHECK_MAX_TOKENS"])
@pytest.mark.parametrize("bad", ["inf", "Infinity", "nan", "-nan", "1e400"])
def test_a_non_finite_number_is_refused(key, bad):
    with pytest.raises(ProviderNotConfigured):
        load_config("bedrock", env={key: bad})


def test_a_max_tokens_that_truncates_to_zero_is_refused():
    with pytest.raises(ProviderNotConfigured):
        load_config("bedrock", env={"RECHECK_MAX_TOKENS": "0.5"})


def test_ordinary_numbers_are_still_accepted():
    config = load_config("bedrock", env={"RECHECK_TIMEOUT_S": "12.5", "RECHECK_MAX_TOKENS": "2048"})
    assert (config.timeout_s, config.max_tokens) == (12.5, 2048)


def test_an_infinite_output_cap_is_a_refusal_not_a_traceback(monkeypatch, tmp_path):
    """RECHECK_MAX_TOKENS=inf raised OverflowError out of load_config: a traceback, exit 1."""
    monkeypatch.setenv("RECHECK_MAX_TOKENS", "inf")
    code, out, err = main("preflight", "--model", "ollama", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert "RECHECK_MAX_TOKENS" in err


def test_a_nan_budget_never_reaches_the_classifier(monkeypatch):
    """With RECHECK_TIMEOUT_S=nan every call was abandoned the moment it was
    made - after the request was already sent - and reported as "did not
    answer within nans"."""
    monkeypatch.setattr(factory_module, "preflight", lambda config: [Check("stub", True, "")])
    with pytest.raises(ProviderNotConfigured):
        build_agent_factory("bedrock", env={"RECHECK_REGION": "us-west-2", "RECHECK_TIMEOUT_S": "nan"})


# --------------------------------------------------------------------------
# MODEL-4: audit --fresh discarded the case before the classifier was resolved
# --------------------------------------------------------------------------

def _refusing_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(factory_module, "preflight",
                        lambda config: [Check("credentials resolvable", False, "ANTHROPIC_API_KEY is not set")])
    return ("--model", "anthropic")


def _unloadable_fixture(monkeypatch, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"classifications": [', encoding="utf-8")
    return ("--scripted", "--classifications", broken)


@pytest.mark.parametrize("refusal", [_refusing_provider, _unloadable_fixture], ids=["provider-not-ready", "bad-fixture"])
def test_a_fresh_audit_whose_classifier_is_refused_keeps_the_existing_case(monkeypatch, tmp_path, refusal):
    letter = tabular_letter(tmp_path / "k.txt", [("Right knee strain", 20), ("Left knee strain", 10),
                                                 ("Tinnitus", 10)], stated=40)
    store = tmp_path / "runs"
    code, _, _ = main("audit", letter, "--case", "k", store=store)
    assert code == EXIT_OK
    before = case_json(store, "k")

    code, out, err = main("audit", letter, "--case", "k", "--fresh", *refusal(monkeypatch, tmp_path), store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert (store / "k" / "case.json").exists(), "the refused re-audit deleted the case"
    assert case_json(store, "k") == before
