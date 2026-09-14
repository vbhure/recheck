"""The Recheck demonstration. Zero model, zero network, zero cost.

Every command below runs as its own process, exactly as a person would type
it, so the process boundaries on screen are real.

    python tools/demo.py              # the whole story
    python tools/demo.py --part 2     # one part: 1 caseload, 2 one letter, 3 guards

AI decisions in this demo are REPLAYED from committed fixtures through
Strands' real structured-output path; no model is called. The reports say so.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "runs" / "demo"
LETTERS = "fixtures/letters"
FIXTURES = "fixtures/classifications"
WIDE = "#" * 78


def banner(title: str) -> None:
    print(f"\n{WIDE}\n# {title}\n{WIDE}")


def say(text: str) -> None:
    print()
    for line in text.strip("\n").splitlines():
        print(f"  {line}")
    print()


def run(*args: str, expect: int) -> str:
    print(f"  $ recheck {' '.join(args)}")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(STORE), *args],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ROOT), timeout=300,
    )
    def shorten(text: str) -> str:
        # Printed commands use forward slashes, so the store appears in both spellings.
        return text.replace(str(STORE), "runs/demo").replace(STORE.as_posix(), "runs/demo")

    output = shorten(result.stdout)
    sys.stdout.write(output)
    if result.stderr.strip():
        sys.stdout.write(shorten(result.stderr))
    print(f"  [exit {result.returncode}]")
    if result.returncode != expect:
        raise SystemExit(f"demo stopped: expected exit {expect}, got {result.returncode}")
    return output


def part_caseload() -> None:
    banner("1. A CASELOAD, WORKED THROUGH UNATTENDED")
    say("""
A County Veterans Service Officer has a folder of VA decision letters (24
synthetic ones here). Recheck audits every letter on its own, finishes what the
letters settle, and brings back only questions whose answer changes a rating.
""")
    run("sweep", "fixtures/caseload", "--scripted", "--classifications", f"{FIXTURES}/caseload.json", expect=2)
    say("""
Three questions out of 24 letters. One more letter left a side unstated, but no
answer could change its rating, so nobody was asked.

case_019: the classifier fixture does not list "left Lisfranc injury", so its
answer came back at confidence 0 and the floor refused it. Recheck did not
guess whether it is an arm or a leg. That process has exited; the question
lives in the case's Strands session. A different process answers it:
""")
    run("resume", "--case", "case_019", "--answer", "2=lower", "--brief", expect=0)
    say("One word from the reviewer. Deterministic code did the rest. Sweep again:")
    out = run("sweep", "fixtures/caseload", "--scripted", "--classifications", f"{FIXTURES}/caseload.json",
              expect=2)
    assert "case_019" in out


def part_letter() -> None:
    banner("2. ONE LETTER, EVERY DECISION TRACED")
    say("""
Four evaluations. Two use clinical names outside the rating schedule's own
vocabulary ("cubital tunnel syndrome", "De Quervain's tenosynovitis"), so the
deterministic lexicon abstains and the classifier decides the extremity group -
replayed here from a fixture. The SIDE is read from the letter by code; the
model has no field for it. Every number is deterministic.
""")
    run("audit", f"{LETTERS}/07_clinical_terms.txt", "--case", "letter07", "--scripted",
        "--classifications", f"{FIXTURES}/07_clinical_terms.json", expect=0)


def part_guards() -> None:
    banner("3. WHAT RECHECK REFUSES TO DO")
    say("The same letter, but it never says which side. The question carries its stakes:")
    run("audit", f"{LETTERS}/08_clinical_terms_no_side.txt", "--case", "letter08", "--scripted",
        "--classifications", f"{FIXTURES}/08_clinical_terms_no_side.json", expect=2)
    say("A number is not a fact. The answer is refused and the question stays open:")
    run("resume", "--case", "letter08", "--answer", "1=100", expect=3)
    say("The real answer, from yet another process:")
    run("resume", "--case", "letter08", "--answer", "1=left,2=right", "--brief", expect=0)

    say("""
No classifier at all. The terms outside the lexicon stay UNKNOWN - Recheck asks
for the missing fact instead of treating "unknown" as "not an arm":
""")
    run("audit", f"{LETTERS}/07_clinical_terms.txt", "--case", "noclassifier", expect=2)
    say("And if the reviewer does not know either, Recheck does not pick a number:")
    run("resume", "--case", "noclassifier", "--answer", "1=unknown,2=unknown", "--brief", expect=3)

    say("The control: a letter whose arithmetic is right is reported as right.")
    run("audit", f"{LETTERS}/06_agrees.txt", "--case", "control", "--brief", expect=0)


def closing() -> None:
    banner("WHAT JUST HAPPENED")
    say("""
DETERMINISTIC  read every evaluation and every side a letter states,
               recognised the rating schedule's own anatomy, decided which
               unknown facts could change a rating, and did all 38 CFR 4.25 /
               4.26 arithmetic (Table I verified against all 684 published
               cells).
AI             named the extremity group only for names outside that
               vocabulary - replayed from fixtures in this demo - and could not
               express a side, a number, or free text.
HUMAN          supplied facts only when an answer changed a rating.

No network, no credential, no inference cost. A live provider replaces the
fixture with --model once its SDK and credentials are set up; recheck preflight
--model <provider> checks that without an inference call.
""")


PARTS = {"1": part_caseload, "2": part_letter, "3": part_guards}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=sorted(PARTS), default=None)
    args = ap.parse_args()
    if STORE.exists():
        shutil.rmtree(STORE)
    banner("RECHECK - re-checks the combined rating in VA decision letters (synthetic data)")
    say("""
VA does not add disability percentages; 38 CFR 4.25 combines them, and 38 CFR
4.26 adds a "bilateral factor" when both arms or both legs are affected. Which
disabilities belong in that factor can move a rating by 10 points, and VA's own
quality reviewers check for "misapplication of the bilateral factor".
""")
    for key in ([args.part] if args.part else sorted(PARTS)):
        PARTS[key]()
    if not args.part:
        closing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
