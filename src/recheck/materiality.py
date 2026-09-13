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

A letter may rate both extremities of a pair in ONE evaluation ("bilateral
pes planus", DC 5276). VA's adjudication manual says when such an evaluation
enters the bilateral factor:

    "When a specific DC provides one evaluation for a bilateral condition,
    only apply the bilateral factor if there is/are an independently ratable
    condition in one of the involved extremities ... or independently
    ratable conditions of both uninvolved extremities"
    - M21-1, Part V, Subpart iv, 1.C.4.b

The first case is applied here: the single evaluation joins the group when
another compensable disability of the same pair of extremities is rated (the
Board did exactly this in Citation Nr 1519449: bilateral feet 10, right knee
10, left knee 10 -> 27, plus 2.7). The second case - the rest of the
calculation is not spelled out - is enumerated both ways, and if the reading
changes the result the case is UNDETERMINED rather than guessed.
"""

from __future__ import annotations

import dataclasses
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
    # "both" is a possible side: a letter that omits the side may be rating a
    # single evaluation of both extremities (plantar fasciitis, DC 5269, is
    # rated "unilateral or bilateral").
    sides = SIDES + ("both",)
    if group == "unknown":
        if side == "unknown":
            return [("none", "unknown")] + [(g, s) for g in GROUPS for s in sides]
        return [("none", side)] + [(g, side) for g in GROUPS]
    if group in GROUPS and side == "unknown":
        return [(group, s) for s in sides]
    return [(group, side)]


def paired_disabilities(
    facts: Sequence[tuple[int, str, str]], *, both_in_factor: bool
) -> list[Paired]:
    """Arm and leg disabilities whose side is established, as the engine takes them.

    A single evaluation covering both extremities ("both") is included when
    M21-1 V.iv.1.C.4.b settles it - another compensable disability of the same
    pair is rated - and otherwise only under the `both_in_factor` reading.
    """
    compensable = [(p, g, s) for p, g, s in facts if g in GROUPS and p >= 10]
    out: list[Paired] = []
    for percent, group, side in facts:
        if group not in GROUPS:
            continue
        if side in SIDES:
            out.append(Paired(percent, group, side))
        elif side == "both":
            others_in_pair = [f for f in compensable if f[1] == group and f != (percent, group, side)]
            if others_in_pair or both_in_factor:
                out.append(Paired(percent, group, side))
    return out


def both_reading_is_open(facts: Sequence[tuple[int, str, str]]) -> bool:
    """True when a single both-sides evaluation is in M21-1's unsettled case.

    That is: nothing else compensable is rated in its own pair of
    extremities, but both of the OTHER pair's extremities are. M21-1 says the
    factor applies then, without saying how the calculation runs.
    """
    compensable = [(p, g, s) for p, g, s in facts if g in GROUPS and p >= 10]
    for percent, group, side in facts:
        if side != "both" or group not in GROUPS or percent < 10:
            continue
        own = [f for f in compensable if f[1] == group and f != (percent, group, side)]
        other = "lower" if group == "upper" else "upper"
        other_sides = {s for _, g, s in compensable if g == other}
        if not own and ({"left", "right"} <= other_sides or "both" in other_sides):
            return True
    return False


def evaluate_established(decisions: Sequence[Decision], *, both_in_factor: bool = False) -> Evaluation:
    """The rating from established facts only. Unknown facts contribute no pairing."""
    facts = [(d.percent, d.extremity_group, d.laterality) for d in decisions]
    return evaluate(
        [d.percent for d in decisions],
        paired=paired_disabilities(facts, both_in_factor=both_in_factor),
    )


def evaluate_for_report(decisions: Sequence[Decision]) -> tuple[Evaluation, dict[int, tuple[str, str]]]:
    """The evaluation to report, and any unknown facts its derivation had to fill in.

    Normally this is the evaluation from established facts. It is not always
    the right one. A condition whose group is known but whose side is not
    must be on SOME side, and "no side" is not among the possibilities: with
    a left knee 10%, a right knee 10% and a knee of unstated side 60%, the
    60% joins the bilateral group whichever side it is on, and every
    possibility gives 80%. Leaving it out of the factor - which is what the
    established facts alone say - gave 70%, reported as a potential
    discrepancy against a letter that stated the correct 80%, underneath a
    trace saying no answer could change the result.

    So when the unknown facts are immaterial but the established-facts figure
    is not the settled one, the derivation is shown for the first possible
    completion, and the facts it assumed are returned so the trace can say
    so. Any completion gives the same final degree; that is what settled means.
    """
    established = evaluate_established(decisions)
    m = assess(decisions)
    if not m.unknown or not m.settled or established.final_degree == m.possible[0]:
        return established, {}
    answers, _ = m.by_answers[0]
    assumed = dict(zip(m.unknown, answers))
    completed = list(decisions)
    for index, (group, side) in assumed.items():
        completed[index] = dataclasses.replace(decisions[index], extremity_group=group, laterality=side)
    return evaluate_established(completed), assumed


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
    # Enumerate both readings only where M21-1 leaves the answer open; a
    # completion that is not in that case gives the same degree either way.
    readings = (False, True) if any(d.laterality == "both" or d.laterality == "unknown"
                                    for d in decisions if d.extremity_group != "none") else (False,)

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
