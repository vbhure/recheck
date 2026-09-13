"""Provenance: who decided what, and on what basis.

Recheck's central claim is a division of responsibility - the model
interprets, deterministic code calculates, a human resolves genuine
ambiguity. A claim like that is worthless unless it is observable, so every
consequential decision is recorded with the actor that made it.

The trace is the product, not a debug log. It is what a reviewer reads to
answer "why did Recheck reach this result?"

Rules enforced here:
  - an entry always names its actor
  - evidence is a location in the source document or None. It is NEVER
    invented. If we cannot point at a line, we say so.
  - deterministic entries carry the regulation they applied
  - AI entries carry a confidence; deterministic and human entries do not,
    because a confidence number would be meaningless for them
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Actor(str, Enum):
    """Who made a decision. Rendered verbatim in the trace."""

    DETERMINISTIC = "DETERMINISTIC"
    AI = "AI"
    HUMAN = "HUMAN"


@dataclass(frozen=True)
class Entry:
    """One auditable decision."""

    actor: Actor
    action: str
    detail: str
    value: str | int | None = None
    confidence: float | None = None
    evidence: str | None = None
    rule: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None:
            if self.actor is not Actor.AI:
                raise ValueError(
                    f"confidence is only meaningful for AI decisions, got actor={self.actor}"
                )
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"confidence out of range: {self.confidence}")
        if self.rule is not None and self.actor is Actor.AI:
            raise ValueError("the model does not apply regulations; only deterministic code does")


@dataclass
class Trace:
    """An ordered, append-only record of decisions."""

    entries: list[Entry] = field(default_factory=list)

    def add(
        self,
        actor: Actor,
        action: str,
        detail: str,
        *,
        value: str | int | None = None,
        confidence: float | None = None,
        evidence: str | None = None,
        rule: str | None = None,
    ) -> Entry:
        entry = Entry(actor, action, detail, value, confidence, evidence, rule)
        self.entries.append(entry)
        return entry

    def by_actor(self, actor: Actor) -> list[Entry]:
        return [e for e in self.entries if e.actor is actor]

    def render(self, width: int = 76, *, ai_label: str = "AI") -> str:
        """Human-readable trace. This is what appears in the demo.

        `ai_label` lets a report say plainly when AI entries were replayed from
        a fixture rather than produced by a model.
        """
        lines: list[str] = []
        for entry in self.entries:
            lines.append(f"[{ai_label if entry.actor is Actor.AI else entry.actor.value}]")
            head = f"  {entry.action}"
            if entry.value is not None:
                head += f": {entry.value}"
            lines.append(head)
            if entry.detail:
                for chunk in wrap(entry.detail, width - 6):
                    lines.append(f"      {chunk}")
            meta: list[str] = []
            if entry.confidence is not None:
                meta.append(f"confidence {entry.confidence:.2f}")
            if entry.rule:
                meta.append(entry.rule)
            if entry.evidence:
                meta.append(f"evidence: {entry.evidence}")
            elif not entry.rule:
                # A rule is its own basis for arithmetic; anything else without
                # a location in the letter says so rather than inventing one.
                meta.append("evidence: none recorded")
            lines.append(f"      ({'; '.join(meta)})")
        return "\n".join(lines)


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out or [""]
