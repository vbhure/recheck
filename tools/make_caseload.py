"""Generate a synthetic caseload of VA rating decision letters.

Every letter is SYNTHETIC. No real veteran data is used anywhere in this
project; names, file numbers and dates are invented. The MIX is synthetic too:
it is a configured distribution, printed by this script, not a sample of real
decisions, and nothing about real error rates can be read from it.

Why a caseload exists as a fixture. The real user is a County Veterans Service
Officer or an accredited representative with a stack of letters, not one. A
sweep has to finish what it can establish alone and bring back only the
questions whose answers change a rating, so the stack mixes letters that agree,
letters that differ, and letters with facts missing.

Three defects in the previous generator shaped this one:

  EVALUATIONS THE SCHEDULE CANNOT PRODUCE. Values were drawn from one list for
  every condition, so letters carried "neuritis of the musculospiral nerve,
  10 percent" (the schedule's lowest radial-nerve value is 20) and knee
  flexion at 50 percent (DC 5260 stops at 30). A VA-literate reader stops
  trusting everything else. Every condition below now carries the value set
  its diagnostic code allows, checked against the eCFR text of 38 CFR Part 4,
  and paired arm-nerve ratings use a major value on the right and a minor
  value on the left.

  STATED VALUES FROM THE ENGINE UNDER TEST. The stated combined evaluations
  were computed by recheck.cfr - so the caseload could only ever agree with
  the code it was meant to test. Ordinary letters now use this script's own
  Table I arithmetic (written here, not imported), and the letters that pin
  behaviour - SPECIAL below - carry values derived by hand, with the steps in
  fixtures/caseload/EXPECTED.md, asserted by tests/test_caseload_expectations.py
  without reference to this script.

  AI THAT WAS DECORATIVE. The old long-tail terms were rating-schedule names
  the lexicon now covers. Long-tail names here are clinical names outside the
  schedule's vocabulary, and only some letters use them, so the classifier
  fixture does real work on some letters and none on others.

Generation is seeded, so the caseload is byte-reproducible.

    python tools/make_caseload.py
"""

from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "caseload"
SEED = 20260914
COUNT = 24

BANNER = "*** SYNTHETIC DOCUMENT - NOT A REAL VA DECISION - NO REAL VETERAN DATA ***"

INITIALS = list("ABCDEFGHJKLMNPRSTVW")
SURNAMES = ["SYNTHETIC", "TESTCASE", "EXAMPLE", "SPECIMEN", "PLACEHOLDER"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

LEXICON, LONG_TAIL = "lexicon", "long-tail"

# Conditions that are not of an arm or leg. The deterministic lexicon
# recognises every one, so none of them costs a classifier call.
NON_EXTREMITY = [
    # name, values the diagnostic code allows (a subset), citation
    ("post-traumatic stress disorder", (30, 50, 70), "DC 9411, General Rating Formula for Mental Disorders"),
    ("major depressive disorder", (30, 50, 70), "DC 9434, General Rating Formula for Mental Disorders"),
    ("tinnitus", (10,), "DC 6260, recurrent: 10"),
    ("lumbosacral strain", (10, 20, 40), "DC 5237, General Rating Formula for the Spine"),
    ("migraine headaches", (10, 30, 50), "DC 8100"),
    ("obstructive sleep apnea", (30, 50), "DC 6847"),
    ("allergic rhinitis", (10, 30), "DC 6522"),
]


@dataclass(frozen=True)
class Limb:
    """A condition of one arm or leg, with the values its code allows per side.

    For arms the schedule distinguishes the major (dominant) and minor
    extremity. These letters never state handedness - Recheck does not need
    it - so the right side takes major values and the left minor ones.
    """

    template: str  # "{side}" is replaced with left / right
    group: str
    source: str  # LEXICON or LONG_TAIL: who is expected to establish the group
    right: tuple[int, ...]
    left: tuple[int, ...]
    citation: str


PAIRED = [
    Limb("limitation of flexion of the {side} knee", "lower", LEXICON, (10, 20, 30), (10, 20, 30),
         "DC 5260: flexion limited to 45 / 30 / 15 degrees"),
    Limb("limited motion of the {side} ankle", "lower", LEXICON, (10, 20), (10, 20),
         "DC 5271: moderate 10, marked 20"),
    Limb("plantar fasciitis of the {side} foot", "lower", LEXICON, (10, 20), (10, 20),
         "DC 5269: 10, or 20 unilateral without relief (30 is bilateral only)"),
    Limb("radiculopathy of the {side} lower extremity", "lower", LEXICON, (10, 20, 40), (10, 20, 40),
         "DC 8520 sciatic nerve, incomplete: mild 10, moderate 20, moderately severe 40"),
    Limb("carpal tunnel syndrome of the {side} wrist", "upper", LEXICON, (10, 30), (10, 20),
         "DC 8515 median nerve, incomplete: mild 10/10, moderate 30 major / 20 minor"),
    Limb("limitation of motion of the {side} arm", "upper", LEXICON, (20, 30), (20,),
         "DC 5201: shoulder level 20/20, midway 30 major / 20 minor"),
    Limb("{side} cubital tunnel syndrome", "upper", LONG_TAIL, (10, 30), (10, 20),
         "by analogy to DC 8516 ulnar nerve, incomplete: mild 10/10, moderate 30 major / 20 minor"),
    Limb("{side} lateral epicondylitis", "upper", LONG_TAIL, (10, 20), (10, 20),
         "DC 5024 rated on DC 5206 forearm flexion: 100 degrees 10/10, 90 degrees 20/20"),
    Limb("{side} De Quervain's tenosynovitis", "upper", LONG_TAIL, (10,), (10,),
         "DC 5024 rated on DC 5215 wrist motion: 10/10"),
    Limb("{side} Achilles tendinopathy", "lower", LONG_TAIL, (10, 20), (10, 20),
         "DC 5024 rated on DC 5271 ankle motion: moderate 10, marked 20"),
    Limb("{side} patellofemoral pain syndrome", "lower", LONG_TAIL, (10, 20), (10, 20),
         "by analogy to DC 5260 knee flexion: 10, 20"),
    Limb("{side} iliotibial band syndrome", "lower", LONG_TAIL, (10,), (10,),
         "DC 5024 rated on DC 5260 knee flexion: 10"),
]


# ---------------------------------------------------------------------------
# The generator's own arithmetic. Deliberately NOT imported from recheck.cfr:
# a caseload whose stated values come from the engine under test cannot find
# anything wrong with that engine. Used only for the ordinary letters, which
# have at most one left/right pair, so 4.26(d) is "with or without the factor,
# whichever is higher".
# ---------------------------------------------------------------------------

def _half_up(value: Fraction) -> int:
    return int((value + Fraction(1, 2)) // 1)


def _table_i(ratings: list[int]) -> int:
    ordered = sorted(ratings, reverse=True)
    running = ordered[0]
    for rating in ordered[1:]:
        running = _half_up(running + Fraction((100 - running) * rating, 100))
    return running


def _degree(combined: int) -> int:
    return (combined + 5) // 10 * 10


def _with_pair(others: list[int], pair: tuple[int, int]) -> tuple[int, int]:
    """(final degree without 4.26, final degree with the most favourable 4.26 result)."""
    without = _degree(_table_i(others + list(pair)))
    base = _table_i(list(pair))
    subtotal = min(100, _half_up(base + Fraction(base, 10)))
    with_factor = _degree(_table_i(others + [subtotal]))
    return without, max(without, with_factor)


# ---------------------------------------------------------------------------
# SPECIAL letters: each pins one behaviour. The stated value and the expected
# outcome are derived by hand in fixtures/caseload/EXPECTED.md; nothing here
# computes them. Format is fixed so the evidence line numbers are stable.
# ---------------------------------------------------------------------------

SPECIAL: list[tuple[str, str, list[tuple[str, int]], int]] = [
    # (key, format, conditions as the letter writes them, stated combined)
    ("i_three_legs", "tabular", [
        ("post-traumatic stress disorder", 30),
        ("limitation of flexion of the left knee", 20),        # DC 5260, 30 degrees
        ("limited motion of the left ankle", 20),              # DC 5271, marked
        ("limitation of flexion of the right knee", 30),       # DC 5260, 15 degrees
        ("tinnitus", 10),
    ], 80),
    ("ii_four_extremities", "prose", [
        ("lumbosacral strain", 20),                            # DC 5237
        ("right cubital tunnel syndrome", 10),                 # DC 8516 by analogy, mild
        ("left cubital tunnel syndrome", 10),                  # DC 8516 by analogy, mild
        ("radiculopathy of the right lower extremity", 20),    # DC 8520, moderate
        ("radiculopathy of the left lower extremity", 10),     # DC 8520, mild
    ], 50),
    ("iii_linked_clause", "tabular", [
        ("Lumbosacral strain", 20),                                                   # DC 5237
        ("Radiculopathy, left lower extremity, associated with lumbosacral strain", 20),   # DC 8520
        ("Radiculopathy, right lower extremity, associated with lumbosacral strain", 10),  # DC 8520
    ], 50),
    ("iv_sideless_immaterial", "prose", [
        ("post-traumatic stress disorder", 70),
        ("limitation of flexion of the knee", 10),             # DC 5260, no side stated
        ("tinnitus", 10),
    ], 80),
    ("v_sideless_material", "prose", [
        ("post-traumatic stress disorder", 50),
        ("limitation of flexion of the knee", 20),             # DC 5260, no side stated
        ("plantar fasciitis", 10),                             # DC 5269, no side stated
    ], 60),
    ("vi_unlisted_long_tail", "prose", [
        ("post-traumatic stress disorder", 50),
        ("limitation of flexion of the right knee", 20),       # DC 5260
        ("left Lisfranc injury", 10),                          # DC 5284 foot injuries, moderate
    ], 60),
    ("vii_stated_higher", "tabular", [
        ("post-traumatic stress disorder", 50),
        ("tinnitus", 10),
        ("limitation of flexion of the right knee", 10),       # DC 5260, 45 degrees
        ("limited motion of the right ankle", 10),             # DC 5271, moderate
    ], 70),
    ("ix_sideless_long_tail_pair", "prose", [
        ("post-traumatic stress disorder", 50),
        ("cubital tunnel syndrome", 20),                       # DC 8516 by analogy
        ("De Quervain's tenosynovitis", 10),                   # DC 5024 on DC 5215
    ], 60),
]

# Ordinary letters fill the rest: AGREE states the value 4.25/4.26 give;
# DISCREPANT states the value you get by leaving the bilateral factor out.
ORDINARY_MIX = {"agree": 10, "discrepant": 6}
# Of the ordinary AGREE letters, this many have no arm or leg condition at all.
NO_LIMB_AGREE = 2
# Chance that an ordinary letter's limb pair uses clinical (long-tail) names.
LONG_TAIL_SHARE = 0.3


def _header(rng: random.Random, index: int) -> str:
    name = f"{rng.choice(INITIALS)}. {rng.choice(SURNAMES)}"
    return (
        f"{BANNER}\n"
        f"DEPARTMENT OF VETERANS AFFAIRS\n"
        f"Regional Office\n\n"
        f"Name: {name}\n"
        f"File Number: 00-{index:03d}-{rng.randint(100, 999)}\n"
        f"Date of Notification: {rng.choice(MONTHS)} {rng.randint(1, 28)}, 2026\n"
    )


def _prose(items: list[tuple[str, int]], stated: int, header: str) -> str:
    body = [header, "\nDECISION\n"]
    for condition, percent in items:
        body.append(
            f"\nService connection for {condition} is granted with an\n"
            f"evaluation of {percent} percent effective January 9, 2026.\n"
        )
    body.append(f"\nYour combined evaluation for compensation is {stated} percent.\n")
    return "".join(body)


def _tabular(items: list[tuple[str, int]], stated: int, header: str) -> str:
    body = [header, "\n                         RATING DECISION\n\n"]
    for number, (condition, percent) in enumerate(items, start=1):
        dots = "." * max(3, 60 - len(condition))
        body.append(f"  {number}. {condition} {dots} {percent}%\n")
    body.append(f"\nCOMBINED EVALUATION FOR COMPENSATION: {stated}%\n")
    return "".join(body)


def _ordinary(rng: random.Random, kind: str, no_limb: bool) -> tuple[list[tuple[str, int]], int, str]:
    """One ordinary letter. Returns (items, stated, description)."""
    for _ in range(500):
        others = [(name, rng.choice(values)) for name, values, _ in rng.sample(NON_EXTREMITY, rng.randint(1, 2))]
        if no_limb:
            others += [(name, rng.choice(values)) for name, values, _ in
                       rng.sample([n for n in NON_EXTREMITY if n[0] not in {o for o, _ in others}], 1)]
            return others, _degree(_table_i([p for _, p in others])), "no arm or leg condition"
        # Most letters name limbs the way the schedule does; a minority use
        # clinical names, so the classifier works on some letters, not all.
        source = LONG_TAIL if rng.random() < LONG_TAIL_SHARE else LEXICON
        right = rng.choice([limb for limb in PAIRED if limb.source == source])
        siblings = [limb for limb in PAIRED if limb.group == right.group and limb.source == source]
        left = right if rng.random() < 0.5 else rng.choice(siblings)
        r_value, l_value = rng.choice(right.right), rng.choice(left.left)
        without, correct = _with_pair([p for _, p in others], (r_value, l_value))
        if kind == "discrepant" and correct == without:
            continue  # the factor is absorbed by rounding here; not a discrepancy
        items = others + [(right.template.format(side="right"), r_value),
                          (left.template.format(side="left"), l_value)]
        rng.shuffle(items)
        stated = without if kind == "discrepant" else correct
        return items, stated, f"{right.group} pair, {source} names"
    raise RuntimeError(f"no {kind} combination found in 500 attempts")


def build() -> tuple[dict[str, str], list[str]]:
    rng = random.Random(SEED)
    letters: dict[str, str] = {}
    log: list[str] = []

    slots = list(range(1, COUNT + 1))
    special_slots = sorted(rng.sample(slots, len(SPECIAL)))
    rng.shuffle(special_slots)
    special_at = dict(zip(special_slots, SPECIAL))

    ordinary = ["agree"] * ORDINARY_MIX["agree"] + ["discrepant"] * ORDINARY_MIX["discrepant"]
    rng.shuffle(ordinary)
    no_limb_left = NO_LIMB_AGREE

    for index in slots:
        header = _header(rng, index)
        case = f"case_{index:03d}"
        if index in special_at:
            key, fmt, items, stated = special_at[index]
            render = _tabular if fmt == "tabular" else _prose
            letters[case] = render(items, stated, header)
            log.append(f"{case}  SPECIAL {key:<28} stated {stated}%  (hand-derived, see EXPECTED.md)")
            continue
        kind = ordinary.pop()
        no_limb = kind == "agree" and no_limb_left > 0
        no_limb_left -= no_limb
        items, stated, description = _ordinary(rng, kind, no_limb)
        render = _tabular if rng.random() < 0.4 else _prose
        letters[case] = render(items, stated, header)
        log.append(f"{case}  {kind:<10} {description:<36} stated {stated}%")
    return letters, log


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("case_*.txt"):
        stale.unlink()

    letters, log = build()
    for name, body in letters.items():
        (OUT / f"{name}.txt").write_text(body, encoding="utf-8")

    print(f"wrote {len(letters)} synthetic letters to {OUT.relative_to(ROOT).as_posix()}")
    print("configured mix (synthetic - not a sample of real decisions):")
    print(f"  special letters pinning one behaviour each: {len(SPECIAL)}")
    print(f"  ordinary agree: {ORDINARY_MIX['agree']} ({NO_LIMB_AGREE} with no arm or leg condition)")
    print(f"  ordinary discrepant (4.26 left out of the stated value): {ORDINARY_MIX['discrepant']}")
    print()
    for line in log:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
