"""AI JUSTIFICATION GATE - evidence for where the model is and is not used.

Recheck refuses to use a language model anywhere deterministic code is
adequate. These tests are the standing, regression-protected proof of where
that line falls.

WHAT IS CLAIMED, PRECISELY:

1. Extraction of percentages and combined values is deterministic.
   The parser handles all six heterogeneous fixture letters - tabular, prose,
   a historical-percentage trap, a cross-body-part pair, missing laterality,
   and a no-discrepancy control. No model is used.
   This evidence is CIRCULAR by construction: the fixtures and the parser
   were written together. It establishes sufficiency for the supported input
   scope. It says nothing about letters we have not seen.

2. The CURRENT LEXICAL BASELINE has low coverage of real VA terminology.
   Measured non-circularly against 138 real condition names from the rating
   schedule in 38 CFR Part 4, labeled by diagnostic-code range - an oracle
   this project does not control - the 29-term lexicon commits to a group
   for 22 names and abstains on 116. It asserts a wrong group zero times.

WHAT IS NOT CLAIMED:
   That no deterministic approach could solve this. A large curated
   anatomical ontology, or a mapping onto an external terminology such as
   SNOMED CT, might well achieve high coverage. That is a different system
   with an external dependency and a maintenance burden, and it is not what
   this project has. The honest statement is narrow: THIS lexical baseline
   does not scale to the terminology tested, which is why a semantic
   classifier is used for the residue.

The measured properties that make the split safe:
   - the deterministic layer never asserts a wrong group, so its positives
     can be trusted without a model call
   - the model's role is confined to extremity group and laterality
   - the model can express neither a percentage nor a combined rating,
     because those fields do not exist in its schema
   - genuine ambiguity goes to a human, never to the model
"""

import json
import pathlib

import pytest

from recheck.extract.deterministic import LEXICON_SIZE, _classify_extremity

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
CONDITIONS = json.loads((FIXTURES / "va_condition_names.json").read_text())

# The value the lexicon returns when it has no opinion. Distinct from "none",
# which is a positive finding that a condition is not an extremity disability.
ABSTAIN = "unrecognised"

# Measured baseline at the time the gate was established.
BASELINE_COVERAGE = 22 / 138
TOLERANCE = 0.08


def _coverage() -> float:
    """Fraction of real condition names the lexicon classifies CORRECTLY."""
    correct = sum(1 for c in CONDITIONS if _classify_extremity(c["name"]) == c["group"])
    return correct / len(CONDITIONS)


def _abstentions() -> int:
    return sum(1 for c in CONDITIONS if _classify_extremity(c["name"]) == ABSTAIN)


def test_dataset_is_real_and_externally_labeled():
    assert len(CONDITIONS) == 138
    assert {c["group"] for c in CONDITIONS} == {"upper", "lower", "none"}


def test_lexical_baseline_has_low_coverage_of_real_terminology():
    """The narrow, precise justification for a semantic classifier.

    If a future change makes the deterministic layer genuinely sufficient,
    this test must fail loudly - because the correct response would be to
    REMOVE the model, not to keep it for appearances.
    """
    coverage = _coverage()
    assert coverage < 0.35, (
        f"the lexicon now classifies {coverage:.1%} of real condition names correctly. "
        f"If it has become sufficient, remove the model from the product rather than "
        f"keeping it, and rewrite this gate."
    )
    assert abs(coverage - BASELINE_COVERAGE) < TOLERANCE


def test_most_real_terminology_is_outside_the_lexicon():
    """Abstention, not error, is the dominant outcome."""
    assert _abstentions() > 0.7 * len(CONDITIONS)


def test_lexicon_never_asserts_a_wrong_group():
    """The safety property that lets the deterministic layer be a fast path.

    When the lexicon commits to "upper" or "lower", that answer feeds 4.26
    pairing without a model call. That is only sound because it is never
    wrong - it abstains instead. A single false positive here would mean
    4.26 could be applied to the wrong pair of conditions silently.
    """
    wrong = [
        {"name": c["name"], "truth": c["group"], "said": _classify_extremity(c["name"])}
        for c in CONDITIONS
        if _classify_extremity(c["name"]) not in (c["group"], ABSTAIN)
    ]
    assert wrong == [], f"lexicon asserted a wrong group for: {wrong[:5]}"


def test_abstention_is_distinct_from_a_positive_none():
    """"I know this is not an extremity" and "I do not know this word" must
    not be the same value.

    Conflating them caused two real problems: the gate over-reported coverage
    as 49.3% by counting abstentions as correct "none" answers, and the
    classifier spent model calls on conditions the lexicon already knew.
    """
    assert _classify_extremity("tinnitus") == "none"
    assert _classify_extremity("Genu recurvatum") == ABSTAIN
    assert _classify_extremity("right knee, limitation of flexion") == "lower"


def test_the_lexicon_is_small_relative_to_the_problem():
    assert LEXICON_SIZE < 40
    assert len(CONDITIONS) > 4 * LEXICON_SIZE


@pytest.mark.parametrize(
    "name,group",
    [
        ("Genu recurvatum", "lower"),
        ("Os calcis or astragalus, malunion", "lower"),
        ("Astragalectomy", "lower"),
        ("Scapulohumeral articulation, ankylosis", "upper"),
        ("Radius and ulna, nonunion", "upper"),
        ("Median nerve, paralysis", "upper"),
        ("Sciatic nerve, paralysis", "lower"),
        ("External popliteal nerve (common peroneal), paralysis", "lower"),
    ],
)
def test_representative_terms_the_lexicon_abstains_on(name, group):
    """Real rating-schedule entries requiring anatomical or Latin knowledge.

    Each generalises semantically but not lexically. These are the named
    examples used in the README and the demo, so they are pinned here.
    """
    assert _classify_extremity(name) == ABSTAIN
    assert group in ("upper", "lower")
