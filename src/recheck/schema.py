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

The model is never permitted to express a percentage, a combined rating, or
a regulation citation in any of these schemas. Those are deterministic
outputs, and a field that does not exist cannot be hallucinated into.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExtremityGroup = Literal["upper", "lower", "none"]
Laterality = Literal["left", "right", "bilateral", "unknown"]


class ConditionClassification(BaseModel):
    """The model's judgment about ONE condition name.

    Scope is deliberately narrow: which extremity group, which side, and how
    sure. Nothing here can influence the arithmetic directly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str = Field(min_length=1, max_length=200)
    extremity_group: ExtremityGroup
    laterality: Laterality
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=400)

    @model_validator(mode="after")
    def _reject_contradictions(self) -> "ConditionClassification":
        """A non-extremity condition cannot have a side.

        Without this, a model could return group="none" with
        laterality="left", and a downstream pairing rule that only checked
        laterality would treat tinnitus as a paired extremity.
        """
        if self.extremity_group == "none" and self.laterality in ("left", "right", "bilateral"):
            raise ValueError(
                f"contradictory classification: extremity_group='none' cannot have "
                f"laterality='{self.laterality}'"
            )
        return self


class ClassificationBatch(BaseModel):
    """Classifications for every condition in one letter."""

    model_config = ConfigDict(extra="forbid")

    classifications: list[ConditionClassification] = Field(min_length=1, max_length=40)


# Below this, the system must not act on a classification without asking a
# human. Chosen to be deliberately conservative: the cost of an unnecessary
# question is one tap; the cost of a wrong 4.26 application is a wrong rating.
CONFIDENCE_FLOOR = 0.75
