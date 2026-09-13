"""Does an unknown fact matter? Answer it by trying every possibility.

A letter that does not say which side a knee is on leaves a fact unknown. It
does not follow that a person has to be asked. If every possible answer gives
the same final degree - there is no other leg disability to pair with, the
rating is 0%, or 4.26(d) would exclude the pair anyway - the question is
immaterial, and asking it would waste the one resource the product exists to
protect: the reviewer's attention.

So before any interrupt, Recheck enumerates every combination of possible
answers to the unknown facts, runs each through the same verified 4.25/4.26
engine that produces the reported figure, and asks a human only if the
answers lead to different final degrees. The question then carries its own
stakes: the ratings the answers lead to.

This is deterministic code deciding when the human is needed. The model has
no part in it.

One further kind of uncertainty is not a question for a reviewer. A letter
may rate both extremities of a pair in ONE evaluation ("bilateral pes
planus"). Whether 4.26 includes such an evaluation in the bilateral factor is
a question of rating practice that Recheck does not decide. It is enumerated
the same way; if it changes the result, the case is reported as
UNDETERMINED rather than guessed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

from recheck.cfr.rating import Evaluation, Paired, evaluate
from recheck.classify import Decision

# Enumeration is exhaustive or it is not used. A letter with so many unknowns
# that this is exceeded is asked about rather than silently sampled.
MAX_COMPLETIONS = 4096

GROUPS = ("upper", "lower")
SIDES = ("left", "right")


def options_for(decision: Decision) -> list[tuple[str, str]]:
    """Every (extremity group, side) a condition could turn out to have."""
    group, side = decision.extremity_group, decision.laterality
    if group == "unknown":
        if side == "unknown":
            return [("none", "unknown")] + [(g, s) for g in GROUPS for s in SIDES]
        return [("none", side)] + [(g, side) for g in GROUPS]
    if group in GROUPS and side == "unknown":
        return [(group, s) for s in SIDES]
    return [(group, side)]


def paired_disabilities(
    facts: Sequence[tuple[int, str, str]], *, both_in_factor: bool
) -> list[Paired]:
    """Arm and leg disabilities whose side is established, as the engine takes them."""
    out: list[Paired] = []
    for percent, group, side in facts:
        if group not in GROUPS:
            continue
        if side in SIDES or (side == "both" and both_in_factor):
            out.append(Paired(percent, group, side))
    return out


def evaluate_established(decisions: Sequence[Decision], *, both_in_factor: bool = False) -> Evaluation:
    """The rating from established facts only. Unknown facts contribute no pairing."""
    facts = [(d.percent, d.extremity_group, d.laterality) for d in decisions]
    return evaluate(
        [d.percent for d in decisions],
        paired=paired_disabilities(facts, both_in_factor=both_in_factor),
    )


@dataclass(frozen=True)
class Materiality:
    """What the unknown facts could do to the final degree."""

    #: final degrees reachable, over every answer and every reading of "both"
    possible: tuple[int, ...]
    #: final degrees for each combination of answers, keyed by the answers
    by_answers: tuple[tuple[tuple[tuple[str, str], ...], tuple[int, ...]], ...]
    #: indices of the decisions that have something unknown
    unknown: tuple[int, ...]
    exhaustive: bool

    @property
    def settled(self) -> bool:
        """True when every possibility gives the same final degree."""
        return self.exhaustive and len(self.possible) == 1

    @property
    def answers_matter(self) -> bool:
        """True when a human's answers could change the final degree."""
        if not self.exhaustive:
            return bool(self.unknown)
        return len({degrees for _, degrees in self.by_answers}) > 1

    @property
    def reading_matters(self) -> bool:
        """True when the treatment of a single both-sides evaluation changes the result."""
        return any(len(degrees) > 1 for _, degrees in self.by_answers)

    def outcomes_for(self, index: int) -> dict[tuple[str, str], tuple[int, ...]]:
        """For one unknown condition: each possible answer -> final degrees it can lead to."""
        if index not in self.unknown:
            return {}
        position = self.unknown.index(index)
        out: dict[tuple[str, str], set[int]] = {}
        for answers, degrees in self.by_answers:
            out.setdefault(answers[position], set()).update(degrees)
        return {k: tuple(sorted(v)) for k, v in out.items()}


def assess(decisions: Sequence[Decision]) -> Materiality:
    """Enumerate every completion of the unknown facts and collect final degrees."""
    unknown = tuple(i for i, d in enumerate(decisions) if d.missing)
    choices = [options_for(decisions[i]) for i in unknown]
    readings = (False, True) if any(d.laterality == "both" and d.extremity_group != "none"
                                    for d in decisions) else (False,)

    count = len(readings)
    for c in choices:
        count *= len(c)
    if count > MAX_COMPLETIONS:
        return Materiality(possible=(), by_answers=(), unknown=unknown, exhaustive=False)

    base = [(d.percent, d.extremity_group, d.laterality) for d in decisions]
    percents = [d.percent for d in decisions]
    by_answers = []
    possible: set[int] = set()
    for answers in itertools.product(*choices):
        facts = list(base)
        for i, (group, side) in zip(unknown, answers):
            facts[i] = (facts[i][0], group, side)
        degrees = set()
        for both_in_factor in readings:
            paired = paired_disabilities(facts, both_in_factor=both_in_factor)
            degrees.add(evaluate(percents, paired=paired).final_degree)
        possible |= degrees
        by_answers.append((tuple(answers), tuple(sorted(degrees))))
    return Materiality(
        possible=tuple(sorted(possible)),
        by_answers=tuple(by_answers),
        unknown=unknown,
        exhaustive=True,
    )
