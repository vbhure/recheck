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
