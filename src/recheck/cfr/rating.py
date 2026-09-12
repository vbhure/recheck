"""Rating determination: orchestrates 4.25/4.26 and records provenance.

combine.py holds pure Table I arithmetic. This module applies the rules in
the order the regulation requires and records a step-by-step trace, so every
number in a Recheck report can be traced back to the rule that produced it.

Nothing here calls a model. Given the same inputs it always produces the
same output and the same trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from recheck.cfr.combine import bilateral_subtotal, combine_step, final_degree

# 4.26(c): the factor needs a compensable disability in EACH of two paired
# extremities. A 0% rating is not compensable.
COMPENSABLE_MINIMUM = 10


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
    alternative_final_degree: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def favorable_path(self) -> str:
        if self.alternative_final_degree is None:
            return "single path"
        return "with bilateral factor" if self.bilateral_applied else "without bilateral factor (4.26(d))"


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


def _is_compensable_pair(pair: Sequence[int]) -> bool:
    return len(pair) == 2 and all(r >= COMPENSABLE_MINIMUM for r in pair)


def evaluate(
    ratings: Sequence[int],
    bilateral_pair: Sequence[int] | None = None,
) -> Evaluation:
    """Determine the final degree of disability.

    Args:
        ratings: All individual ratings, INCLUDING any bilateral pair members.
        bilateral_pair: The two ratings affecting paired extremities, if any.
            Must be a subset of `ratings`.

    Returns:
        An Evaluation carrying the final degree, the combined value, the full
        derivation, and - where 4.26(d) applies - the alternative it beat.

    Raises:
        ValueError: If `bilateral_pair` is not a sub-multiset of `ratings`.
    """
    ratings = [int(r) for r in ratings]
    notes: list[str] = []

    if not bilateral_pair:
        value, steps = _fold(ratings, "all ratings")
        return Evaluation(
            final_degree=final_degree(value),
            combined_value=value,
            steps=tuple(steps),
            notes=("No bilateral pair identified; 4.26 not applied.",),
        )

    pair = [int(r) for r in bilateral_pair]
    remaining = list(ratings)
    for r in pair:
        if r not in remaining:
            raise ValueError(f"bilateral pair member {r} is not among the ratings {ratings}")
        remaining.remove(r)

    # Path B: ignore the bilateral factor entirely. Needed for 4.26(d).
    plain_value, plain_steps = _fold(ratings, "all ratings, no bilateral factor")
    plain_degree = final_degree(plain_value)

    if not _is_compensable_pair(pair):
        notes.append(
            f"4.26(c): the bilateral factor requires a compensable (>={COMPENSABLE_MINIMUM}%) "
            f"disability in each paired extremity; got {pair}. Factor not applied."
        )
        return Evaluation(
            final_degree=plain_degree,
            combined_value=plain_value,
            steps=tuple(plain_steps),
            notes=tuple(notes),
        )

    # Path A: apply the bilateral factor.
    subtotal = bilateral_subtotal(pair)
    bi_steps = [
        Step(
            rule="38 CFR 4.26",
            detail=(
                f"bilateral pair {pair[0]} and {pair[1]}: combine as usual, then ADD "
                f"(not combine) 10% of that value; result treated as one disability"
            ),
            running_before=None,
            rating_applied=None,
            running_after=subtotal,
        )
    ]
    bi_value, fold_steps = _fold([*remaining, subtotal], "bilateral subtotal plus remaining ratings")
    bi_steps.extend(fold_steps)
    bi_degree = final_degree(bi_value)

    # 4.26(d): choose whichever is most favorable to the veteran.
    if plain_degree > bi_degree:
        notes.append(
            f"4.26(d) exception applied: including the pair in the bilateral calculation "
            f"yields {bi_degree}%, but excluding it yields {plain_degree}%. The regulation "
            f"requires the result most favorable to the veteran."
        )
        return Evaluation(
            final_degree=plain_degree,
            combined_value=plain_value,
            steps=tuple(plain_steps),
            bilateral_applied=False,
            bilateral_subtotal_value=subtotal,
            alternative_final_degree=bi_degree,
            notes=tuple(notes),
        )

    if plain_degree < bi_degree:
        notes.append(
            f"Bilateral factor is favorable: {bi_degree}% with it, {plain_degree}% without."
        )
    else:
        notes.append(
            f"Bilateral factor does not change the final degree here (both {bi_degree}%)."
        )

    return Evaluation(
        final_degree=bi_degree,
        combined_value=bi_value,
        steps=tuple(bi_steps),
        bilateral_applied=True,
        bilateral_subtotal_value=subtotal,
        alternative_final_degree=plain_degree,
        notes=tuple(notes),
    )
