"""AI JUSTIFICATION GATE - the evidence that the model earns its place.

Recheck refuses to use a language model anywhere deterministic code is
adequate. These tests are the standing proof of where that line actually
falls, and they are regression-protected so the claim cannot silently rot.

Two findings, both measured rather than asserted:

1. EXTRACTION OF RATINGS IS DETERMINISTIC. The parser handles all six
   heterogeneous fixture letters - tabular, prose, historical-percentage
   traps, cross-body-part pairs, missing laterality and a no-discrepancy
   control. No model is used or needed. (This alone is circular evidence,
   since the fixtures and the parser were written together; it establishes
   sufficiency for the supported scope, not generality.)

2. ANATOMICAL CLASSIFICATION IS NOT DETERMINISTIC AT SCALE. Measured
   NON-CIRCULARLY against 138 real VA condition names taken from the rating
   schedule in 38 CFR Part 4 and labeled by diagnostic-code range - an
   external oracle this project does not control - the hand-built lexicon
   classifies under 60% correctly, and every error is a MISS rather than a
   false positive.

Finding 2 is why a model exists in this product, and it is confined to
exactly that job: deciding which extremity group a condition belongs to.
Percentages are parsed deterministically, the arithmetic is deterministic,
and genuine ambiguity goes to a human rather than to the model.
"""

import json
import pathlib

import pytest

from recheck.extract.deterministic import LEXICON_SIZE, _classify_extremity

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
CONDITIONS = json.loads((FIXTURES / "va_condition_names.json").read_text())

# The measured baseline at the time this gate was established. If a future
# change to the lexicon moves this materially, the justification narrative in
# the README must be re-derived rather than quietly left stale.
BASELINE_ACCURACY = 0.493
TOLERANCE = 0.06


def _accuracy() -> float:
    correct = sum(1 for c in CONDITIONS if _classify_extremity(c["name"]) == c["group"])
    return correct / len(CONDITIONS)


def test_dataset_is_real_and_externally_labeled():
    assert len(CONDITIONS) == 138
    assert {c["group"] for c in CONDITIONS} == {"upper", "lower", "none"}


def test_deterministic_lexicon_is_measurably_insufficient():
    """The core justification: a lexicon does not scale to real VA vocabulary."""
    accuracy = _accuracy()
    assert accuracy < 0.60, (
        f"lexicon now classifies {accuracy:.1%} of real condition names. If it has "
        f"genuinely become sufficient, the model should be removed from the product, "
        f"not kept for appearances."
    )
    assert abs(accuracy - BASELINE_ACCURACY) < TOLERANCE


def test_the_lexicon_is_small_relative_to_the_problem():
    """29 terms against a rating schedule of several hundred conditions."""
    assert LEXICON_SIZE < 40
    assert len(CONDITIONS) > 4 * LEXICON_SIZE


def test_errors_are_misses_not_false_positives():
    """Safety property: the lexicon never confidently asserts a WRONG group.

    This is what makes the deterministic layer safe to keep as a fast path -
    when it is wrong it says "none", which routes to the model or to a human,
    rather than silently applying 4.26 to the wrong pair of conditions.
    """
    false_positives = [
        c
        for c in CONDITIONS
        if _classify_extremity(c["name"]) not in (c["group"], "none")
    ]
    assert false_positives == [], f"lexicon asserted a wrong group for: {false_positives[:5]}"


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
def test_representative_terms_the_lexicon_cannot_reach(name, group):
    """Named examples for the README and the demo.

    These are real entries from the VA rating schedule. Each requires
    anatomical or Latin knowledge that generalises semantically but not
    lexically - which is precisely a language model's job and precisely not
    a regex's.
    """
    assert _classify_extremity(name) == "none"
    assert group in ("upper", "lower")
