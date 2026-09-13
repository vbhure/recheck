"""Rating determination: orchestrates 4.25/4.26 and records provenance.

combine.py holds pure Table I arithmetic. This module applies the rules in
the order the regulation requires and records a step-by-step trace, so every
number in a Recheck report can be traced back to the rule that produced it.

Nothing here calls a model. Given the same inputs it always produces the
same output and the same trace.

38 CFR 4.26, as applied here (every quoted clause is the regulation's text,
from the eCFR source cited in combine.py):

  which disabilities   "the ratings for the disabilities of the right and
                       left sides" of both arms, or of both legs - ALL of
                       them, not one pair. "Arms" and "legs" mean the upper
                       and lower extremities "as a whole" (4.26(a)).
  four extremities     both arms AND both legs: "combine the ratings of the
                       disabilities affecting the 4 extremities" into one
                       subtotal (4.26(b)).
  compensable only     "not applicable unless there is partial disability of
                       compensable degree in each of 2 paired extremities"
                       (4.26(c)).
  most favourable      when leaving "one or more bilateral disabilities" out
                       of the factor gives a higher combined evaluation, they
                       are removed and combined separately (4.26(d)). Every
                       such removal is tried; the regulation does not limit
                       it to all-or-nothing.

Paired skeletal muscles are not modelled: Recheck classifies conditions into
upper and lower extremities only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

from recheck.cfr.combine import bilateral_subtotal, combine_step, final_degree

# 4.26(c): the factor needs a compensable disability in EACH of two paired
# extremities. A 0% rating is not compensable.
COMPENSABLE_MINIMUM = 10

EXTREMITIES = ("upper", "lower")
SIDES = ("left", "right")

# 4.26(d) is an exhaustive search over which bilateral disabilities to leave
# out. A rating decision with more than this many arm and leg disabilities is
# not something Recheck has been verified against, so it refuses rather than
# silently truncating the search.
MAX_BILATERAL_MEMBERS = 12


@dataclass(frozen=True)
class Paired:
    """A disability of an arm or a leg whose side has been established.

    side "both" is ONE evaluation covering both extremities of a pair (a
    letter's "bilateral pes planus, 30%"). Whether 4.26 treats such an
    evaluation as bilateral is not something Recheck decides: it is used only
    to test whether that question can change a result (recheck.materiality),
    and no reported figure ever depends on it.
    """

    percent: int
    extremity: str  # "upper" | "lower"
    side: str  # "left" | "right" | "both"

    def sides(self) -> set[str]:
        return set(SIDES) if self.side == "both" else {self.side}

    def label(self) -> str:
        return f"{self.percent}% {self.side} {self.extremity}"


@dataclass(frozen=True)
class Step:
    """One auditable arithmetic step."""

    rule: str
    detail: str
    running_before: int | None
    rating_applied: int | None
    running_after: int


@dataclass(frozen=True)
class Evaluation:
    """The result of a rating determination, with its full derivation."""

    final_degree: int
    combined_value: int
    steps: tuple[Step, ...] = field(default_factory=tuple)
    bilateral_applied: bool = False
    bilateral_subtotal_value: int | None = None
    bilateral_members: tuple[Paired, ...] = field(default_factory=tuple)
    excluded_under_426d: tuple[Paired, ...] = field(default_factory=tuple)
    alternative_final_degree: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)



def bilateral_group(disabilities: Sequence[Paired]) -> list[Paired]:
    """The disabilities 4.26 puts into the bilateral factor by default.

    An extremity pair qualifies when there is a compensable disability on
    BOTH its left and right side (4.26(c)). Every compensable disability of a
    qualifying pair enters the factor. If both pairs qualify, all of them form
    one group (4.26(b)).
    """
    compensable = [d for d in disabilities if d.percent >= COMPENSABLE_MINIMUM]
    qualifying = {
        extremity
        for extremity in EXTREMITIES
        if _sides_covered([d for d in compensable if d.extremity == extremity]) == set(SIDES)
    }
    return [d for d in compensable if d.extremity in qualifying]


def _sides_covered(members: Sequence[Paired]) -> set[str]:
    covered: set[str] = set()
    for m in members:
        covered |= m.sides()
    return covered


def _is_valid_group(members: Sequence[Paired]) -> bool:
    """A non-empty set the factor may be applied to: both sides of each pair used."""
    if not members:
        return False
    for extremity in {m.extremity for m in members}:
        if _sides_covered([m for m in members if m.extremity == extremity]) != set(SIDES):
            return False
    return True


def _fold(ratings: Sequence[int], label: str) -> tuple[int, list[Step]]:
    """Fold ratings through Table I in descending order, recording each step."""
    ordered = sorted(ratings, reverse=True)
    steps: list[Step] = []
    if not ordered:
        return 0, steps
    running = ordered[0]
    steps.append(
        Step(
            rule="38 CFR 4.25(a)",
            detail=f"{label}: arrange in descending severity; start at the greatest disability",
            running_before=None,
            rating_applied=running,
            running_after=running,
        )
    )
    for rating in ordered[1:]:
        after = combine_step(running, rating)
        steps.append(
            Step(
                rule="38 CFR 4.25 Table I",
                detail=f"combine {running} with {rating}: {running} + ({100 - running} x {rating}%) rounded",
                running_before=running,
                rating_applied=rating,
                running_after=after,
            )
        )
        running = after
    return running, steps


def _remove_all(ratings: Sequence[int], members: Sequence[int]) -> list[int]:
    """`ratings` minus `members` as multisets. Raises if members are not a subset."""
    remaining = list(ratings)
    for r in members:
        if r not in remaining:
            raise ValueError(f"bilateral disability {r} is not among the ratings {list(ratings)}")
        remaining.remove(r)
    return remaining


def _with_factor(others: Sequence[int], members: Sequence[Paired]) -> tuple[int, int]:
    """(combined value, subtotal) with the factor applied to `members`."""
    subtotal = bilateral_subtotal([m.percent for m in members])
    value, _ = _fold([*others, subtotal], "")
    return value, subtotal


def evaluate(
    ratings: Sequence[int],
    bilateral_pair: Sequence[int] | None = None,
    *,
    paired: Sequence[Paired] | None = None,
) -> Evaluation:
    """Determine the final degree of disability.

    Args:
        ratings: All individual ratings, INCLUDING any arm and leg disabilities.
        paired: Every arm or leg disability whose side is established. Which
            of them enter the bilateral factor is decided here, by 4.26 -
            never by the caller.
        bilateral_pair: Shorthand for exactly two disabilities of one pair of
            extremities, on opposite sides. Kept because the regulation's own
            worked examples are stated that way.

    Returns:
        An Evaluation carrying the final degree, the combined value, the full
        derivation, and - where the factor was considered - the alternative
        result, so a reader can see what 4.26 changed.

    Raises:
        ValueError: If the paired disabilities are not a sub-multiset of
            `ratings`, if both arguments are given, or if there are more arm
            and leg disabilities than the 4.26(d) search is verified for.
    """
    ratings = [int(r) for r in ratings]
    if paired is not None and bilateral_pair is not None:
        raise ValueError("pass either `paired` or `bilateral_pair`, not both")
    if bilateral_pair is not None:
        if len(bilateral_pair) != 2:
            raise ValueError(f"a bilateral pair has exactly 2 ratings, got {len(bilateral_pair)}")
        # The extremity label is immaterial to the arithmetic; one pair of
        # extremities with a disability on each side is what matters.
        paired = [Paired(int(bilateral_pair[0]), "upper", "left"),
                  Paired(int(bilateral_pair[1]), "upper", "right")]
    paired = list(paired or [])
    _remove_all(ratings, [p.percent for p in paired])  # validate membership

    plain_value, plain_steps = _fold(ratings, "all ratings, no bilateral factor")
    plain_degree = final_degree(plain_value)
    group = bilateral_group(paired)

    if not group:
        if not paired:
            note = "No arm or leg disabilities with an established side; 4.26 not applied."
        else:
            note = (
                f"4.26(c): the bilateral factor requires a compensable (>={COMPENSABLE_MINIMUM}%) "
                f"disability on both the left and right side of the same pair of extremities; "
                f"got {', '.join(p.label() for p in paired)}. Factor not applied."
            )
        return Evaluation(
            final_degree=plain_degree,
            combined_value=plain_value,
            steps=tuple(plain_steps),
            notes=(note,),
        )

    if len(group) > MAX_BILATERAL_MEMBERS:
        raise ValueError(
            f"{len(group)} bilateral disabilities exceeds the {MAX_BILATERAL_MEMBERS} this "
            f"engine is verified for; refusing rather than truncating the 4.26(d) search"
        )

    others = _remove_all(ratings, [m.percent for m in group])

    # The default procedure: every qualifying disability in the factor.
    default_value, default_subtotal = _with_factor(others, group)
    default_degree = final_degree(default_value)

    # 4.26(d): try leaving out each combination of one or more members. The
    # empty kept-set is "no factor at all". Ties keep the larger group, so
    # the regulation's default procedure is only departed from when doing so
    # is strictly more favourable.
    best_members: tuple[Paired, ...] = tuple(group)
    best_degree = default_degree
    for size in range(len(group) - 1, -1, -1):
        for kept_idx in itertools.combinations(range(len(group)), size):
            kept = [group[i] for i in kept_idx]
            if size == 0:
                degree = plain_degree
            elif not _is_valid_group(kept):
                continue
            else:
                left_out = [group[i].percent for i in range(len(group)) if i not in kept_idx]
                value, _ = _with_factor([*others, *left_out], kept)
                degree = final_degree(value)
            if degree > best_degree:
                best_degree, best_members = degree, tuple(kept)

    notes: list[str] = []
    four = {m.extremity for m in group} == set(EXTREMITIES)
    rule = "38 CFR 4.26(b)" if four else "38 CFR 4.26"

    if not best_members:
        notes.append(
            f"4.26(d) exception applied: the bilateral factor yields {default_degree}%, but "
            f"combining without it yields {plain_degree}%. The regulation requires the result "
            f"most favorable to the veteran."
        )
        return Evaluation(
            final_degree=plain_degree,
            combined_value=plain_value,
            steps=tuple(plain_steps),
            bilateral_applied=False,
            bilateral_subtotal_value=default_subtotal,
            excluded_under_426d=tuple(group),
            alternative_final_degree=default_degree,
            notes=tuple(notes),
        )

    members = list(best_members)
    # Multiset difference, so two identical disabilities are counted correctly.
    excluded = list(group)
    for m in members:
        excluded.remove(m)
    left_out = [m.percent for m in excluded]
    subtotal = bilateral_subtotal([m.percent for m in members])
    base = _fold([m.percent for m in members], "")[0]
    steps = [
        Step(
            rule=rule,
            detail=(
                f"bilateral disabilities {', '.join(m.label() for m in members)}: combine as "
                f"usual ({base}), then ADD (not combine) 10% of that value; the result is "
                f"treated as one disability"
                + ("; all four extremities form one group" if four and len(members) == len(group) else "")
            ),
            running_before=None,
            rating_applied=None,
            running_after=subtotal,
        )
    ]
    value, fold_steps = _fold([*others, *left_out, subtotal], "bilateral subtotal plus remaining ratings")
    steps.extend(fold_steps)
    degree = final_degree(value)

    if excluded:
        notes.append(
            f"4.26(d) exception applied: with every bilateral disability in the factor the result "
            f"is {default_degree}%; leaving out {', '.join(m.label() for m in excluded)} gives "
            f"{degree}%, which is more favorable to the veteran."
        )
        alternative = default_degree
    elif plain_degree < degree:
        notes.append(f"Bilateral factor is favorable: {degree}% with it, {plain_degree}% without.")
        alternative = plain_degree
    else:
        notes.append(f"Bilateral factor does not change the final degree here (both {degree}%).")
        alternative = plain_degree

    return Evaluation(
        final_degree=degree,
        combined_value=value,
        steps=tuple(steps),
        bilateral_applied=True,
        bilateral_subtotal_value=subtotal,
        bilateral_members=tuple(members),
        excluded_under_426d=tuple(excluded),
        alternative_final_degree=alternative,
        notes=tuple(notes),
    )
