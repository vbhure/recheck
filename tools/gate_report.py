"""AI JUSTIFICATION GATE: measure what deterministic parsing actually achieves.

Runs the deterministic parser against every fixture letter and compares the
result to hand-written ground truth. The purpose is to find the honest
boundary of deterministic parsing BEFORE any model is integrated.

If this reports full coverage, the correct conclusion is that a model is not
needed for extraction within the supported input scope, and the product
architecture should be reassessed accordingly.

    python tools/gate_report.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from recheck.cfr.rating import evaluate  # noqa: E402
from recheck.extract.deterministic import (  # noqa: E402
    candidate_bilateral_pairs,
    parse,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
LETTERS = ROOT / "fixtures" / "letters"

# Hand-written ground truth. Each entry: the ratings a human reader would
# extract, the stated combined value, and the expected bilateral outcome.
TRUTH = {
    "01_tabular": dict(ratings=[60, 20, 10, 10], stated=70, pairs=1, ambiguous=False),
    "02_prose": dict(ratings=[60, 20, 10, 10], stated=70, pairs=1, ambiguous=False),
    "03_history_trap": dict(ratings=[60, 20, 10, 10], stated=70, pairs=1, ambiguous=False),
    "04_cross_bodypart": dict(ratings=[60, 20, 10, 10], stated=70, pairs=1, ambiguous=False),
    "05_missing_side": dict(ratings=[60, 20, 10, 10], stated=70, pairs=0, ambiguous=True),
    "06_agrees": dict(ratings=[50, 30], stated=70, pairs=0, ambiguous=False),
}


def main() -> int:
    passes = 0
    failures: list[str] = []

    for path in sorted(LETTERS.glob("*.txt")):
        name = path.stem
        truth = TRUTH.get(name)
        extraction = parse(path.read_text(encoding="utf-8"))
        got = sorted((r.percent for r in extraction.ratings), reverse=True)
        want = sorted(truth["ratings"], reverse=True) if truth else []
        pairs = candidate_bilateral_pairs(extraction.ratings)

        problems = []
        if got != want:
            problems.append(f"ratings {got} != expected {want}")
        if extraction.stated_combined != truth["stated"]:
            problems.append(f"stated combined {extraction.stated_combined} != {truth['stated']}")
        if len(pairs) != truth["pairs"]:
            problems.append(f"{len(pairs)} bilateral pair(s) detected, expected {truth['pairs']}")
        if bool(extraction.ambiguities) != truth["ambiguous"]:
            problems.append(
                f"ambiguity detected={bool(extraction.ambiguities)}, expected {truth['ambiguous']}"
            )

        status = "PASS" if not problems else "FAIL"
        if problems:
            failures.append(name)
        else:
            passes += 1

        print(f"\n{status}  {name}")
        for rating in extraction.ratings:
            print(
                f"      {rating.percent:>3}%  {rating.condition[:52]:<52} "
                f"[{rating.extremity_group}/{rating.laterality}]"
            )
        print(f"      stated combined: {extraction.stated_combined}")
        if pairs:
            for i, j in pairs:
                a, b = extraction.ratings[i], extraction.ratings[j]
                print(f"      bilateral candidate: {a.condition[:28]} + {b.condition[:28]}")
        for note in extraction.ambiguities:
            print(f"      AMBIGUOUS: {note}")
        if extraction.unparsed_reason:
            print(f"      UNPARSED: {extraction.unparsed_reason}")
        for problem in problems:
            print(f"      >>> {problem}")

        # What the arithmetic says, where extraction succeeded.
        if extraction.ok and len(pairs) == 1 and not extraction.ambiguities:
            i, j = pairs[0]
            pair = [extraction.ratings[i].percent, extraction.ratings[j].percent]
            ev = evaluate(got, bilateral_pair=pair)
            verdict = "DISCREPANCY" if ev.final_degree != extraction.stated_combined else "agrees"
            print(f"      -> recomputed {ev.final_degree}% vs stated {extraction.stated_combined}% ({verdict})")
        elif extraction.ok and not pairs and not extraction.ambiguities:
            ev = evaluate(got)
            verdict = "DISCREPANCY" if ev.final_degree != extraction.stated_combined else "agrees"
            print(f"      -> recomputed {ev.final_degree}% vs stated {extraction.stated_combined}% ({verdict})")

    total = len(TRUTH)
    print("\n" + "=" * 68)
    print(f"DETERMINISTIC COVERAGE: {passes}/{total} letters fully handled")
    if failures:
        print(f"deterministic parsing FAILED on: {', '.join(failures)}")
        print("-> these are the cases a model must earn its place on")
    else:
        print("-> deterministic parsing is SUFFICIENT for the supported input scope")
        print("-> a model is NOT justified for extraction; reassess the architecture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
