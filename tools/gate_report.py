"""AI JUSTIFICATION GATE: where the deterministic lexicon stops, measured.

Recheck uses a model for exactly one judgment - which extremity group a
condition name belongs to - and only for names the deterministic lexicon
abstains on. This report measures the lexicon on two sets and prints both,
because each answers a different question and neither is enough alone:

  (a) RATING-SCHEDULE VOCABULARY  fixtures/va_condition_names.json
      138 diagnostic-code titles from 38 CFR Part 4, labelled by the
      diagnostic-code range they sit in - labels this project does not
      control. The schedule's vocabulary is finite, so the lexicon is
      expected to cover most of it. Where it does, a model call would be
      waste, and none is made.

  (b) LETTER PHRASINGS  fixtures/va_letter_phrasings.json
      Clinical, eponymous and colloquial names in the style decision letters
      use. Labels are INTERNAL - assigned by this project from anatomy, not an
      external oracle - and entries marked ambiguous are not scored. The
      lexicon is expected to abstain on most of the arm and leg names here.
      That residue is the model's job.

On both sets the lexicon must never assert a wrong group. It abstains instead,
and an abstention costs at most a model call or one question.

No model is run by this report, and nothing here says how accurate a model
would be on set (b).

    python tools/gate_report.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recheck.extract.deterministic import LEXICON_SIZE, _classify_extremity  # noqa: E402

ABSTAIN = "unrecognised"
SCHEDULE = ROOT / "fixtures" / "va_condition_names.json"
PHRASINGS = ROOT / "fixtures" / "va_letter_phrasings.json"


def measure(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["group"] in ("upper", "lower", "none")]
    out = {"total": len(rows), "scored": len(scored), "excluded": len(rows) - len(scored),
           "correct": [], "abstained": [], "wrong": []}
    for row in scored:
        said = _classify_extremity(row["name"])
        bucket = "abstained" if said == ABSTAIN else "correct" if said == row["group"] else "wrong"
        out[bucket].append((row["name"], row["group"], said))
    return out


def _split(entries, limb: bool):
    return [e for e in entries if (e[1] != "none") == limb]


def report(title: str, m: dict) -> None:
    print("=" * 74)
    print(title)
    print("=" * 74)
    limbs = _split([*m["correct"], *m["abstained"], *m["wrong"]], True)
    others = _split([*m["correct"], *m["abstained"], *m["wrong"]], False)
    print(f"names scored: {m['scored']}   (excluded as ambiguous: {m['excluded']})")
    print(f"  arm or leg:  {len(limbs):>3}   resolved {len(_split(m['correct'], True)):>3}   "
          f"abstained {len(_split(m['abstained'], True)):>3}")
    print(f"  neither:     {len(others):>3}   resolved {len(_split(m['correct'], False)):>3}   "
          f"abstained {len(_split(m['abstained'], False)):>3}")
    print(f"  all:         {m['scored']:>3}   resolved {len(m['correct']):>3}   "
          f"abstained {len(m['abstained']):>3}   WRONG {len(m['wrong'])}")
    for name, truth, said in m["wrong"]:
        print(f"  >>> WRONG: {name!r} is {truth}, lexicon said {said}")


def main() -> int:
    schedule = measure(json.loads(SCHEDULE.read_text(encoding="utf-8")))
    phrasings = measure(json.loads(PHRASINGS.read_text(encoding="utf-8"))["conditions"])

    print(f"deterministic lexicon: {LEXICON_SIZE} entries (words, phrases, non-extremity hints)")
    print()
    report("(a) RATING-SCHEDULE VOCABULARY - labels: diagnostic-code range (external)", schedule)
    print("abstained:")
    for name, truth, _ in schedule["abstained"]:
        print(f"    {truth:<6} {name}")
    print("-> the schedule's own vocabulary is resolved deterministically; no model call is spent on it")
    print()
    report("(b) LETTER PHRASINGS - labels: internal, authored for this project", phrasings)
    print("arm and leg names the lexicon abstains on (the classifier's work):")
    for name, truth, _ in _split(phrasings["abstained"], True):
        print(f"    {truth:<6} {name}")
    print("-> clinical, eponymous and colloquial names are outside the schedule's vocabulary;")
    print("   this residue is where a semantic classifier is used. No live model has been")
    print("   run against it, so nothing here measures a model's accuracy.")
    print()
    wrong = len(schedule["wrong"]) + len(phrasings["wrong"])
    print(f"wrong-group assertions across both sets: {wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
