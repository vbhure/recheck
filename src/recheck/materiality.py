"""Does an unknown fact matter? Answer it by trying every possibility.

A letter that does not say which side a knee is on leaves a fact unknown. It
does not follow that a person has to be asked. If every possible answer gives
the same final degree - there is no other leg disability to pair with, the
rating is 0%, or 4.26(d) would exclude the pair anyway - the question is
immaterial, and asking it would waste the one resource the product exists to
protect: the reviewer's attention.

So before any interrupt, Recheck enumerates every combination of possible
answers to the unknown facts (up to MAX_COMPLETIONS, within MAX_SEARCH_WORK;
past either the case is UNDETERMINED, never sampled), runs each through the same verified 4.25/4.26
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
10, left knee 10 -> 27, plus 2.7). When neither case holds, M21-1 settles it
the other way - no factor (the Board called adding one to a lone bilateral
evaluation pyramiding, Citation Nr 0003457) - and nothing is enumerated. The
second case - the rest of the calculation is not spelled out - is enumerated
both ways, and so is the 4.26(d) search leaving the evaluation ALONE in the
factor after removing the disability that let it in. If the reading changes
the result the case is UNDETERMINED rather than guessed.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from dataclasses import dataclass
from typing import Sequence

from recheck.cfr.rating import (
    COMPENSABLE_MINIMUM,
    MAX_BILATERAL_MEMBERS,
    Evaluation,
    Paired,
    bilateral_group,
    evaluate,
)
from recheck.classify import Decision

# Enumeration is exhaustive or it is not used. A letter with so many unknowns
# that this is exceeded is reported UNDETERMINED rather than silently sampled.
#
# It counts completions: combinations of answers to the unknown facts. The
# two readings of a single both-sides evaluation are not counted here. They
# used to be, doubling every letter, so four ordinary 10% terms outside the
# lexicon (7^4 x 2 = 4802) went over the cap - and the over-cap path then
# crashed the assess node with IndexError.
MAX_COMPLETIONS = 4096

# The completion count does not bound time: each distinct evaluation runs a
# 4.26(d) search over up to 2^n subsets of its bilateral group. Twelve arm
# and leg ratings with six unstated sides took 8.4 minutes of CPU, and one
# such letter stalls a whole sweep. The arithmetic of the search is now
# cached (recheck.cfr.rating), and a letter can only ever need as many
# distinct results as its percentages have sub-multisets, so what is left
# grows with the subsets visited - budgeted here, summed over DISTINCT
# evaluations (completions that produce the same facts are evaluated once).
# A count, not a clock, so the same letter gets the same answer on every
# machine. That 12-rating letter needs about 750,000 subsets and completes in
# about 3.5 seconds. The budget does not make every letter under it that
# quick: the slowest one found (a 50% rating, eight sided arm and leg ratings
# of mixed percentages, three terms outside the lexicon: 938,240 subsets)
# took 6 to 10.5 seconds for one assess() on the development machines, and
# 12.7 seconds for the whole audit. That is the cost to expect at the
# budget. Letters over it were refused in under 0.4 seconds. Past the budget
# the result is UNDETERMINED, never sampled.
MAX_SEARCH_WORK = 1 << 20

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


def _enumerated_options(decision: Decision) -> list[tuple[str, str]]:
    """The options assess() has to try for one unknown condition.

    A non-compensable (0%) evaluation can never be a member of the bilateral
    factor (4.26(c); the engine drops it before forming the group), so every
    option gives the same degree and one stands for all of them. Trying all
    seven multiplied the completions by seven for nothing and pushed ordinary
    letters over the cap.
    """
    options = options_for(decision)
    return options if decision.percent >= COMPENSABLE_MINIMUM else options[:1]


def _both_case(facts: Sequence[tuple[int, str, str]], index: int) -> str:
    """Which M21-1 V.iv.1.C.4.b case the both-sides evaluation at `index` is in.

      "joins"     another compensable disability of its own pair is rated
      "open"      nothing else in its pair, but both extremities of the other
                  pair are rated: the factor applies, the calculation is not
                  spelled out
      "excluded"  neither: M21-1 applies no factor to it

    Other members are told apart by position, not by value. Bilateral pes
    planus 30 and bilateral plantar fasciitis 30 are each other's "other"
    disability; comparing value tuples made each disappear from the other's
    view, so neither joined and a 70% letter came out UNDETERMINED.
    """
    _, group, _ = facts[index]
    compensable = [(i, g, s) for i, (p, g, s) in enumerate(facts) if g in GROUPS and p >= COMPENSABLE_MINIMUM]
    if any(i != index and g == group for i, g, _ in compensable):
        return "joins"
    other = "lower" if group == "upper" else "upper"
    other_sides = {s for _, g, s in compensable if g == other}
    if {"left", "right"} <= other_sides or "both" in other_sides:
        return "open"
    return "excluded"


def paired_disabilities(
    facts: Sequence[tuple[int, str, str]], *, both_in_factor: bool
) -> list[Paired]:
    """Arm and leg disabilities whose side is established, as the engine takes them.

    A single evaluation covering both extremities ("both") is included when
    M21-1 V.iv.1.C.4.b settles it - another compensable disability of the same
    pair is rated - never when M21-1 settles it the other way, and in M21-1's
    open case only under the `both_in_factor` reading.
    """
    out: list[Paired] = []
    for index, (percent, group, side) in enumerate(facts):
        if group not in GROUPS:
            continue
        if side in SIDES:
            out.append(Paired(percent, group, side))
        elif side == "both":
            case = _both_case(facts, index)
            if case == "joins" or (case == "open" and both_in_factor):
                out.append(Paired(percent, group, side))
    return out


def both_reading_is_open(facts: Sequence[tuple[int, str, str]]) -> bool:
    """True when the result for these facts may depend on how M21-1 is read.

    That is when a compensable single both-sides evaluation is in M21-1's
    open case, or joins the factor under its first case - because the 4.26(d)
    search can then leave out the disability that let it in and leave it in
    the factor alone. For any other facts the two readings are the same
    calculation.
    """
    return any(
        side == "both" and group in GROUPS and percent >= COMPENSABLE_MINIMUM
        and _both_case(facts, index) != "excluded"
        for index, (percent, group, side) in enumerate(facts)
    )


def readings_for(facts: Sequence[tuple[int, str, str]]) -> tuple[bool, ...]:
    """The readings of a single both-sides evaluation worth running, strict first."""
    return (False, True) if both_reading_is_open(facts) else (False,)


def evaluate_established(decisions: Sequence[Decision], *, both_in_factor: bool = False) -> Evaluation:
    """The rating from established facts only. Unknown facts contribute no pairing."""
    facts = [(d.percent, d.extremity_group, d.laterality) for d in decisions]
    return evaluate(
        [d.percent for d in decisions],
        paired=paired_disabilities(facts, both_in_factor=both_in_factor),
        lone_both_in_factor=both_in_factor,
    )


@functools.lru_cache(maxsize=1 << 16)
def _final_degree(ratings: tuple[int, ...], paired: tuple[tuple[int, str, str], ...], reading: bool) -> int:
    """One engine run, keyed by the multisets that decide it.

    The final degree depends on which ratings there are and which of them
    are paired, not on their order, so completions that differ only in which
    of two identical evaluations is on which side share one run. The cache
    also serves the repeat assessments of one letter in one process (the
    assess node, then compute, then the printed question).
    """
    return evaluate(list(ratings), paired=[Paired(*p) for p in paired], lone_both_in_factor=reading).final_degree


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
    #: engine evaluations the enumeration stands for: completions times readings
    combinations: int = 0
    #: why the enumeration is not exhaustive, when it is not - a whole reason,
    #: in lower case, fit to be the case's undetermined_reason
    limit: str | None = None

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
    choices = [_enumerated_options(decisions[i]) for i in unknown]

    completions = math.prod(len(c) for c in choices)
    if completions > MAX_COMPLETIONS:
        return Materiality(
            possible=(), by_answers=(), unknown=unknown, exhaustive=False,
            limit=(f"too many facts are unknown to try every possibility (the unknown facts can be "
                   f"completed {completions} ways, over the limit of {MAX_COMPLETIONS})"),
        )

    # First every completion's engine inputs, and what running them would
    # cost - so a letter over budget is refused in milliseconds, not after
    # spending the budget.
    base = [(d.percent, d.extremity_group, d.laterality) for d in decisions]
    planned = []
    work = runs = 0
    seen: set[tuple[tuple[tuple[int, str, str], ...], bool]] = set()
    for answers in itertools.product(*choices):
        facts = list(base)
        for i, (group, side) in zip(unknown, answers):
            facts[i] = (facts[i][0], group, side)
        keys = []
        # Both readings only where M21-1 leaves the answer open for THIS
        # completion. Choosing once for the whole letter ran the open reading
        # wherever any side was unknown, so a lone "bilateral" rating that
        # M21-1 settles (no factor) came out UNDETERMINED, or raised a
        # question whose every answer led to the same rating.
        for reading in readings_for(facts):
            paired = paired_disabilities(facts, both_in_factor=reading)
            key = (tuple(sorted((p.percent, p.extremity, p.side) for p in paired)), reading)
            if key not in seen:
                seen.add(key)
                members = len(bilateral_group(paired, lone_both_in_factor=reading))
                if members > MAX_BILATERAL_MEMBERS:
                    # The engine refuses this rather than truncate 4.26(d),
                    # and its ValueError escaped the assess node, leaving the
                    # case stuck in 'classified'. One completion the engine
                    # will not evaluate means the possible degrees are not
                    # known, so none are reported - as over the cap.
                    return Materiality(
                        possible=(), by_answers=(), unknown=unknown, exhaustive=False,
                        limit=(f"{members} arm and leg disabilities would enter the bilateral factor, "
                               f"more than the {MAX_BILATERAL_MEMBERS} its 4.26(d) search is verified for"),
                    )
                work += 2 ** members
            keys.append(key)
        runs += len(keys)
        planned.append((tuple(answers), keys))
    if work > MAX_SEARCH_WORK:
        return Materiality(
            possible=(), by_answers=(), unknown=unknown, exhaustive=False,
            limit=(f"too many facts are unknown to try every possibility (trying every completion of "
                   f"the unknown facts needs {work} 4.26(d) combinations, over the limit of {MAX_SEARCH_WORK})"),
        )

    ratings = tuple(sorted(d.percent for d in decisions))
    by_answers = []
    possible: set[int] = set()
    for answers, keys in planned:
        degrees = {_final_degree(ratings, *key) for key in keys}
        possible |= degrees
        by_answers.append((answers, tuple(sorted(degrees))))
    return Materiality(
        possible=tuple(sorted(possible)),
        by_answers=tuple(by_answers),
        unknown=unknown,
        exhaustive=True,
        combinations=runs,
    )
