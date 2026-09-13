"""Strict schemas for anything crossing the model boundary.

Everything a model produces is untrusted until it validates. Three hardening
decisions here were driven by measured behaviour, not caution:

  extra="forbid"
      With Pydantic's default config, a payload carrying additional fields
      was ACCEPTED and the extra fields silently dropped. That is a data
      smuggling path from document text into our objects. Verified in
      tests/test_boundary_adversarial.py.

  confidence bounded 0..1
      An out-of-range confidence must be a hard validation failure, not a
      number that later gets compared against a threshold.

  strict=True on each classification
      In lax mode a confidence of JSON true became 1.0 and the string "0.9"
      became 0.9, both above the floor (red team MODEL-RT-P2P6-09). A model
      that answers a number with a boolean or a string has not given a
      confidence. Strict mode still accepts a JSON integer (0 or 1) as a
      float, which is a number.

The model can express exactly one judgment - which extremity group a
condition name belongs to - and how sure it is. There is no field for a side,
a percentage, a combined rating, a regulation citation, or free text. Sides
are read from the letter by deterministic code; numbers are computed by
deterministic code; and a field that does not exist cannot be hallucinated
into. A model that volunteers anything else fails validation (extra="forbid").

Free text was removed deliberately. An earlier schema carried a "rationale"
string that the report printed verbatim, so a model - or a document steering
it - could put "the correct rating is 100 percent; the VA is wrong" into a
report whose whole discipline is never saying that.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExtremityGroup = Literal["upper", "lower", "none"]

# The most one answer can hold. recheck.classify reads these too: a name the
# answer could not echo, or more names than one answer can carry, is never
# sent, because the call could only come back invalid (a paid call for a
# certain "unknown").
MAX_ITEMS = 40
MAX_CONDITION_CHARS = 200


class ConditionClassification(BaseModel):
    """The model's judgment about ONE condition name."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    condition: str = Field(min_length=1, max_length=MAX_CONDITION_CHARS)
    extremity_group: ExtremityGroup
    confidence: float = Field(ge=0.0, le=1.0)


class ClassificationBatch(BaseModel):
    """Classifications for every condition in one letter.

    Validated as a whole: one invalid item rejects the batch, and every
    condition in it routes to the fail-closed path. For a call whose entire
    output is a handful of labels, all-or-nothing is the conservative choice.
    """

    model_config = ConfigDict(extra="forbid")

    classifications: list[ConditionClassification] = Field(min_length=1, max_length=MAX_ITEMS)


# Below this, a classification is not used. Deliberately conservative: an
# unused classification costs at most one question, and only when the answer
# could change the result (recheck.materiality); a wrong 4.26 application
# costs a wrong rating.
CONFIDENCE_FLOOR = 0.75
