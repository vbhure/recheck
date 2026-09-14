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
                       such removal is tried (the regulation does not limit
                       it to all-or-nothing), for up to MAX_BILATERAL_MEMBERS
                       disabilities in the factor; more is refused, and the
                       case is reported UNDETERMINED.

Paired skeletal muscles are not modelled: Recheck classifies conditions into
upper and lower extremities only.
"""

from __future__ import annotations

import functools
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
    evaluation as bilateral is not something Recheck decides beyond what
    M21-1 settles: recheck.materiality decides which of them are passed in,
    evaluates the unsettled treatment both ways (see `lone_both_in_factor`
    on `evaluate`), and no reported figure ever depends on it.
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



def bilateral_group(disabilities: Sequence[Paired], *, lone_both_in_factor: bool = True) -> list[Paired]:
    """The disabilities 4.26 puts into the bilateral factor by default.

    An extremity pair qualifies when there is a compensable disability on
    BOTH its left and right side (4.26(c)). Every compensable disability of a
    qualifying pair enters the factor. If both pairs qualify, all of them form
    one group (4.26(b)). See `evaluate` for `lone_both_in_factor`.
    """
    compensable = [d for d in disabilities if d.percent >= COMPENSABLE_MINIMUM]
    qualifying = {
        extremity
        for extremity in EXTREMITIES
        if _pair_may_take_factor([d for d in compensable if d.extremity == extremity], lone_both_in_factor)
    }
    return [d for d in compensable if d.extremity in qualifying]


def _sides_covered(members: Sequence[Paired]) -> set[str]:
    covered: set[str] = set()
    for m in members:
        covered |= m.sides()
    return covered


def _pair_may_take_factor(members: Sequence[Paired], lone_both_in_factor: bool) -> bool:
    """Whether these members of ONE pair of extremities may carry the factor.

    Both sides must be covered. A single evaluation that names both sides
    covers both on its own, but M21-1 V.iv.1.C.4.b applies the factor to it
    only alongside "an independently ratable condition in one of the involved
    extremities". Under the strict reading that other disability must be in
    the factor with it; `lone_both_in_factor` is the reading that lets it
    stand there alone.
    """
    if _sides_covered(members) != set(SIDES):
        return False
    if not lone_both_in_factor and len(members) == 1 and members[0].side == "both":
        return False
    return True


def _is_valid_group(members: Sequence[Paired], lone_both_in_factor: bool = True) -> bool:
    """A non-empty set the factor may be applied to: both sides of each pair used."""
    if not members:
        return False
    for extremity in {m.extremity for m in members}:
        if not _pair_may_take_factor([m for m in members if m.extremity == extremity], lone_both_in_factor):
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


def _percentage(value: object) -> int:
    """A rating as this engine takes it: a whole percentage from 0 to 100.

    Coercing with int() silently turned 24.9 into 24 (40 and 24.9 gave 50%,
    not 60%) and True into 1; an unknown extremity or side dropped a
    disability from the factor without a word. Refused instead.
    """
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError(f"a rating must be a whole percentage from 0 to 100, got {value!r}")
    return value


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


@functools.lru_cache(maxsize=1 << 16)
def _degree_with_factor(others: tuple[int, ...], members: tuple[int, ...]) -> int:
    """Final degree with the factor applied to `members`: one 4.26(d) candidate.

    The same arithmetic as `_with_factor`, on sorted percentages, cached.
    The search tries up to 2^12 subsets per evaluation, and recheck.materiality
    runs it once per completion of a letter's unknown facts - completions that
    mostly differ in which side a rating is on, not in the percentages being
    combined. Uncached, twelve arm and leg ratings with six unstated sides took
    8.4 minutes of CPU.
    """
    # _with_factor reads only the percentages; the extremity and side of these
    # placeholders were checked by the caller (_is_valid_group) and play no part.
    return final_degree(_with_factor(list(others), [Paired(p, "upper", "left") for p in members])[0])


def evaluate(
    ratings: Sequence[int],
    bilateral_pair: Sequence[int] | None = None,
    *,
    paired: Sequence[Paired] | None = None,
    lone_both_in_factor: bool = True,
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
        lone_both_in_factor: Whether a single evaluation naming both sides
            may be the ONLY member of its pair in the factor. False is the
            strict reading of M21-1 V.iv.1.C.4.b. It matters most in the
            4.26(d) search: bilateral pes planus 50 and a right knee 10 enter
            the factor together, but leaving the knee out left the pes planus
            alone in the factor - the case M21-1 says gets no factor - and
            that reading alone turned 90% into a reported 100%. The caller
            (recheck.materiality) evaluates both readings wherever this can
            arise and withholds the figure when they differ.

    Returns:
        An Evaluation carrying the final degree, the combined value, the full
        derivation, and - where the factor was considered - the alternative
        result, so a reader can see what 4.26 changed.

    Raises:
        ValueError: If the paired disabilities are not a sub-multiset of
            `ratings`, if both arguments are given, or if there are more arm
            and leg disabilities than the 4.26(d) search is verified for.
    """
    ratings = [_percentage(r) for r in ratings]
    if paired is not None and bilateral_pair is not None:
        raise ValueError("pass either `paired` or `bilateral_pair`, not both")
    if bilateral_pair is not None:
        if len(bilateral_pair) != 2:
            raise ValueError(f"a bilateral pair has exactly 2 ratings, got {len(bilateral_pair)}")
        # The extremity label is immaterial to the arithmetic; one pair of
        # extremities with a disability on each side is what matters.
        paired = [Paired(_percentage(bilateral_pair[0]), "upper", "left"),
                  Paired(_percentage(bilateral_pair[1]), "upper", "right")]
    paired = list(paired or [])
    for p in paired:
        _percentage(p.percent)
        if p.extremity not in EXTREMITIES or p.side not in (*SIDES, "both"):
            raise ValueError(f"not an arm or leg disability with a side: {p!r}")
    _remove_all(ratings, [p.percent for p in paired])  # validate membership

    plain_value, plain_steps = _fold(ratings, "all ratings, no bilateral factor")
    plain_degree = final_degree(plain_value)
    group = bilateral_group(paired, lone_both_in_factor=lone_both_in_factor)

    if not group:
        if not paired:
            note = "No arm or leg disabilities with an established side; 4.26 not applied."
        elif any(_sides_covered([p for p in paired if p.extremity == e and p.percent >= COMPENSABLE_MINIMUM])
                 == set(SIDES) for e in EXTREMITIES):
            # Both sides ARE covered - by one evaluation naming both, which the
            # strict reading keeps out of the factor on its own. Citing 4.26(c)'s
            # left-and-right requirement here contradicted the "both" it listed.
            note = (
                f"M21-1 V.iv.1.C.4.b: a single evaluation of both extremities takes the bilateral factor "
                f"only with another compensable disability of the same pair in the factor; got "
                f"{', '.join(p.label() for p in paired)}. Factor not applied."
            )
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
            elif not _is_valid_group(kept, lone_both_in_factor):
                continue
            else:
                left_out = [group[i].percent for i in range(len(group)) if i not in kept_idx]
                degree = _degree_with_factor(tuple(sorted([*others, *left_out])),
                                             tuple(sorted(m.percent for m in kept)))
            if degree > best_degree:
                best_degree, best_members = degree, tuple(kept)

    notes: list[str] = []
    four = {m.extremity for m in group} == set(EXTREMITIES)
    # Cite 4.26(b) only for a factor that still spans both pairs. When 4.26(d)
    # leaves one pair out, the subtotal is of that one pair: plain 4.26.
    rule = "38 CFR 4.26(b)" if {m.extremity for m in best_members} == set(EXTREMITIES) else "38 CFR 4.26"

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
