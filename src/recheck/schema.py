"""Strict schemas for anything crossing the model boundary.

Everything a model produces is untrusted until it validates. Two hardening
decisions here were driven by measured behaviour, not caution:

  extra="forbid"
      With Pydantic's default config, a payload carrying additional fields
      was ACCEPTED and the extra fields silently dropped. That is a data
      smuggling path from document text into our objects. Verified in
      tests/test_boundary_adversarial.py.

  confidence bounded 0..1
      An out-of-range confidence must be a hard validation failure, not a
      number that later gets compared against a threshold.

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


class ConditionClassification(BaseModel):
    """The model's judgment about ONE condition name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str = Field(min_length=1, max_length=200)
    extremity_group: ExtremityGroup
    confidence: float = Field(ge=0.0, le=1.0)


class ClassificationBatch(BaseModel):
    """Classifications for every condition in one letter.

    Validated as a whole: one invalid item rejects the batch, and every
    condition in it routes to the fail-closed path. For a call whose entire
    output is a handful of labels, all-or-nothing is the conservative choice.
    """

    model_config = ConfigDict(extra="forbid")

    classifications: list[ConditionClassification] = Field(min_length=1, max_length=40)


# Below this, a classification is not used. Deliberately conservative: an
# unused classification costs at most one question, and only when the answer
# could change the result (recheck.materiality); a wrong 4.26 application
# costs a wrong rating.
CONFIDENCE_FLOOR = 0.75
