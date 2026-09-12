"""Generate a synthetic caseload of VA rating decision letters.

Every letter is SYNTHETIC. No real veteran data is used anywhere in this
project; names, file numbers and dates are invented.

Why a caseload exists as a fixture. The real user is not a veteran holding one
letter - it is a County Veterans Service Officer or an accredited
representative with a stack of them. A tool that audits one document
interactively is a demo; a tool that works through a stack unattended and
surfaces only the decisions a human must actually make is the product. The
distribution below is chosen to make that honest:

  MAJORITY AGREE. Most decisions are arithmetically correct, and the sweep
  must show that. A tool that finds a problem in every case is a tool nobody
  should trust, and a caseload where everything is wrong would be a lie.

  SOME DISCREPANT. A minority where applying 4.26 to a pair the letter's own
  stated total appears to have ignored moves the final band.

  SOME AMBIGUOUS. A few where the letter never states which side a condition
  is on, so 4.26 eligibility cannot be established from the document and a
  human has to be asked.

Generation is seeded, so the caseload is byte-reproducible.

    python tools/make_caseload.py            # 24 letters
    python tools/make_caseload.py --count 40
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "caseload"
SEED = 20260914

sys.path.insert(0, str(ROOT / "src"))

BANNER = "*** SYNTHETIC DOCUMENT - NOT A REAL VA DECISION - NO REAL VETERAN DATA ***"

INITIALS = list("ABCDEFGHJKLMNPRSTVW")
SURNAMES = ["SYNTHETIC", "TESTCASE", "EXAMPLE", "SPECIMEN", "PLACEHOLDER"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Non-extremity conditions: the deterministic lexicon recognises these.
NON_EXTREMITY = [
    ("post-traumatic stress disorder", [30, 50, 70]),
    ("tinnitus", [10]),
    ("lumbosacral strain", [10, 20, 40]),
    ("migraine headaches", [30]),
    ("obstructive sleep apnea", [50]),
]

# Paired-extremity conditions, as (template, extremity group). Some use real
# rating-schedule nerve terminology the lexicon abstains on, so the classifier
# is genuinely exercised across the caseload rather than in one showcase case.
PAIRED = [
    ("limitation of flexion of the {side} knee", "lower", False),
    ("degenerative arthritis of the {side} ankle", "lower", False),
    ("plantar fasciitis of the {side} foot", "lower", False),
    ("carpal tunnel syndrome of the {side} wrist", "upper", False),
    ("incomplete paralysis of the {side} median nerve", "upper", True),
    ("neuritis of the {side} musculospiral nerve", "upper", True),
    ("incomplete paralysis of the {side} sciatic nerve", "lower", True),
    ("radiculopathy of the {side} lower extremity", "lower", False),
]

PAIR_VALUES = [(20, 10), (30, 20), (10, 10), (40, 20), (20, 20), (30, 10), (50, 20)]


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
        dots = "." * max(3, 52 - len(condition))
        body.append(f"  {number}. {condition} {dots} {percent}%\n")
    body.append(f"\nCOMBINED EVALUATION FOR COMPENSATION: {stated}%\n")
    return "".join(body)


def build(count: int) -> dict[str, str]:
    from recheck.cfr.combine import combine, final_degree
    from recheck.cfr.rating import evaluate

    rng = random.Random(SEED)
    letters: dict[str, str] = {}

    # 60% agree, 25% discrepant, 15% ambiguous - rounded to whole letters.
    n_ambiguous = max(2, round(count * 0.15))
    n_discrepant = max(2, round(count * 0.25))
    n_agree = count - n_ambiguous - n_discrepant
    kinds = ["agree"] * n_agree + ["discrepant"] * n_discrepant + ["ambiguous"] * n_ambiguous
    rng.shuffle(kinds)

    for index, kind in enumerate(kinds, start=1):
        header = _header(rng, index)
        template, group, _ = rng.choice(PAIRED)
        high, low = rng.choice(PAIR_VALUES)
        others = []
        for condition, options in rng.sample(NON_EXTREMITY, rng.randint(1, 2)):
            others.append((condition, rng.choice(options)))

        if kind == "ambiguous":
            # Two DIFFERENT conditions in the same extremity group, neither of
            # which states a side. 4.26 cannot be established from the letter.
            #
            # An earlier version reused one template twice and derived the
            # second name by string substitution, which for some templates
            # produced two IDENTICAL condition names. Extraction dedups on
            # (condition, percent), so one of the pair silently vanished and
            # the case had a rating missing. Use two distinct templates.
            siblings = [t for t, g, _ in PAIRED if g == group and t != template]
            if not siblings:
                kinds.append("agree")  # cannot build an ambiguous case here
                continue
            other_template = rng.choice(siblings)
            strip = lambda t: t.replace(" of the {side}", "").replace("{side} ", "")
            items = others + [(strip(template), high), (strip(other_template), low)]
            stated = final_degree(combine([p for _, p in items]))
        else:
            # For a DISCREPANT case the letter must state the value you get by
            # ignoring 4.26, and that must actually differ from applying it.
            # For most rating combinations it does NOT differ - the factor gets
            # absorbed by the rounding to the nearest ten - so the combination
            # has to be searched for rather than picked and hoped over.
            #
            # An earlier version picked randomly and fell back to agreeing when
            # the values matched. That silently turned the whole discrepant
            # bucket into agreeing cases, so the caseload misrepresented its
            # own mix. Search, and fail loudly if nothing is found.
            combos = [(h, l) for h in (10, 20, 30, 40, 50) for l in (10, 20, 30, 40, 50) if h >= l]
            rng.shuffle(combos)
            chosen = None
            for h, l in combos:
                percents = [p for _, p in others] + [h, l]
                with_factor = evaluate(percents, bilateral_pair=[h, l]).final_degree
                without = final_degree(combine(percents))
                differs = with_factor != without
                if (kind == "discrepant") == differs:
                    chosen = (h, l, without if kind == "discrepant" else with_factor)
                    break
            if chosen is None:
                raise RuntimeError(
                    f"case {index}: no {kind} rating combination exists for others={others}"
                )
            high, low, stated = chosen
            items = others + [
                (template.format(side="right"), high),
                (template.format(side="left"), low),
            ]

        rng.shuffle(items)
        render = _tabular if rng.random() < 0.4 else _prose
        letters[f"case_{index:03d}"] = render(items, stated, header)

    return letters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=24)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("case_*.txt"):
        stale.unlink()

    letters = build(args.count)
    for name, body in letters.items():
        (OUT / f"{name}.txt").write_text(body, encoding="utf-8")
    print(f"wrote {len(letters)} synthetic letters to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
