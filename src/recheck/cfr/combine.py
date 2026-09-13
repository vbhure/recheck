"""Combined disability rating arithmetic, 38 CFR 4.25 and 4.26.

This module is the correctness spine of Recheck. No model calls, no I/O,
no network. Pure functions only, exact Decimal arithmetic, never float.

WHY INTEGER STEPS (the non-obvious part):
38 CFR 4.25(a) says the combined value "exactly as found in table I" is
combined with the next disability. Table I cells are INTEGERS, so the
regulation rounds to a whole number at EVERY pairwise step - it does not
carry unrounded decimals through the chain. Getting this wrong shifts
results by a point or two, which is enough to flip the final 10-point band.

The rounding mode at each step is ROUND_HALF_UP. This was not assumed: it
was fitted against all 684 published cells of Table I, extracted from the
official eCFR API and committed to fixtures/cfr425_table1_points.json.
ROUND_HALF_UP matches 684/684. ROUND_HALF_EVEN fails 33. Truncation fails 310.
See tests/test_table1_exhaustive.py.

PRIMARY SOURCES (fetched via the official eCFR API, not scraped):
  38 CFR 4.25 - Combined ratings table
  38 CFR 4.26 - Bilateral factor
  https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-38.xml?subtitle=A&part=4

Worked examples in the regulations themselves, all covered by tests:
  4.25  60 and 30 -> 72
  4.25  50 and 30 -> 65 -> converts to 70
  4.25  40 and 20 -> 52 -> converts to 50
  4.26  60, 20, 10, 10 (the two 10s bilateral) -> 21, 68, 74 -> converts to 70
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Sequence

ONE_HUNDRED = Decimal(100)
BILATERAL_FACTOR_PCT = Decimal(10)


def _round_half_up(value: Decimal) -> int:
    """Round a Decimal to the nearest integer, halves away from zero."""
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _validate(ratings: Sequence[Decimal]) -> None:
    for v in ratings:
        if v < 0 or v > ONE_HUNDRED:
            raise ValueError(f"rating out of range 0-100: {v}")


def combine_step(running: int | Decimal, rating: int | Decimal) -> int:
    """One Table I lookup: combine a running value with the next rating.

    Implements the efficiency logic of 4.25 - the new disability consumes a
    share of the efficiency that REMAINS, not of the original 100.

    Returns an integer, because Table I cells are integers.
    """
    a = Decimal(running)
    b = Decimal(rating)
    _validate([a, b])
    return _round_half_up(a + (ONE_HUNDRED - a) * b / ONE_HUNDRED)


def combine(ratings: Iterable[int | Decimal]) -> int:
    """Combine ratings per 4.25, returning the integer combined value.

    Disabilities are arranged in descending order of severity as the
    regulation requires, then folded pairwise through Table I.

    This is the combined VALUE (e.g. 74), not the final degree of
    disability (e.g. 70). Call `final_degree` for that.
    """
    values = [Decimal(r) for r in ratings]
    _validate(values)
    if not values:
        return 0
    ordered = sorted(values, reverse=True)
    running: int | Decimal = ordered[0]
    for rating in ordered[1:]:
        running = combine_step(running, rating)
    return int(running)


def bilateral_subtotal(ratings: Sequence[int | Decimal]) -> int:
    """Combine the bilateral disabilities and add the 4.26 factor.

    4.26: "the ratings for the disabilities of the right and left sides will
    be combined as usual, and 10 percent of this value will be added (i.e.,
    not combined)". That is every such disability, not one pair of them: a
    left knee, a left ankle and a right knee all enter the calculation, and
    4.26(b) folds all four extremities into one subtotal when both arms and
    both legs are affected. Which disabilities qualify is decided in
    recheck.cfr.rating; this function only does the arithmetic.

    "Added, not combined" is arithmetic addition - treating it as another
    4.25 combination understates the result and is the most common way this
    calculation is gotten wrong.

    The result is rounded to an integer. This is not an assumption: 4.26's
    own worked example combines 10 and 10 to 19, adds 10% (1.9) and states
    the order of severity as "60, 21 and 20" - i.e. 20.9 becomes 21.

    Capped at 100. Adding 10% can carry a large subtotal past 100 (80 and 60
    combine to 92; adding 9.2 gives 101), which no percentage of disability
    can be. Anything combined with 100 stays 100, so the cap cannot change a
    final degree - without it, a legitimate letter crashed Table I validation.
    """
    if not ratings:
        raise ValueError("the bilateral factor needs at least one disability")
    base = Decimal(combine(ratings))
    return min(100, _round_half_up(base + base * BILATERAL_FACTOR_PCT / ONE_HUNDRED))


def final_degree(combined_value: int | Decimal) -> int:
    """Convert a combined value to the final degree of disability.

    4.25(a): "converted to the nearest number divisible by 10, and combined
    values ending in 5 will be adjusted upward."

    This is the ONLY rounding-to-ten site in Recheck. Intermediate values
    are never converted; only a finished combined value is.

    A combined value above 100 cannot exist (the bilateral subtotal is
    capped), so one arriving here is a defect upstream and is refused loudly
    rather than converted into an impossible "110%".
    """
    if not Decimal(0) <= Decimal(combined_value) <= ONE_HUNDRED:
        raise ValueError(f"combined value out of range 0-100: {combined_value}")
    scaled = (Decimal(combined_value) / Decimal(10)) + Decimal("0.5")
    return int(scaled.to_integral_value(rounding="ROUND_FLOOR")) * 10
