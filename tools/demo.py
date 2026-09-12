"""The Recheck demonstration. Zero model, zero network, zero cost.

Runs the three golden cases in sequence. Golden B uses REAL subprocesses, so
the process death and resume are genuine rather than narrated.

    python tools/demo.py            # all three cases
    python tools/demo.py --case b   # one case

The point the demo has to land in its first thirty seconds:

    AI decides what needs human judgment.
    Human judgment resolves ambiguity.
    Deterministic code decides the arithmetic.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LETTERS = ROOT / "fixtures" / "letters"
CLASSIFICATIONS = ROOT / "fixtures" / "classifications"
STORE = ROOT / "runs" / "demo"

WIDE = "#" * 78


def banner(title: str, subtitle: str = "") -> None:
    print()
    print(WIDE)
    print(f"# {title}")
    if subtitle:
        print(f"# {subtitle}")
    print(WIDE)


def step(text: str) -> None:
    print(f"\n  >>> {text}\n")


def cli(*args: str, expect: int | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    def rel(a):
        text = str(a)
        try:
            return pathlib.Path(text).relative_to(ROOT).as_posix()
        except (ValueError, OSError):
            return text
    printable = " ".join(rel(a) for a in args)
    print(f"  $ python -m recheck.cli --store runs/demo {printable}")
    print(f"    (pid of this shell: {os.getpid()} - the command below gets its own)")
    result = subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(STORE), *[str(a) for a in args]],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=180,
    )
    sys.stdout.write(result.stdout)
    if result.stderr.strip():
        sys.stderr.write(result.stderr)
    print(f"  [exit code {result.returncode}]")
    if expect is not None and result.returncode != expect:
        raise SystemExit(f"demo aborted: expected exit {expect}, got {result.returncode}")
    return result


def golden_sweep() -> None:
    banner(
        "THE PRODUCT - a caseload, worked through unattended",
        "24 synthetic decision letters; only the decisions come back to you",
    )
    print("""
  The user is not a veteran holding one letter. It is a County Veterans
  Service Officer, or an accredited representative, with a stack of them.

  So this is the product: point it at the stack, walk away, and read the
  triage. Everything the system can establish from the documents themselves
  it finishes on its own. What it cannot establish - which side a condition
  is on, when the letter never says - comes back as a question.

  Watch the last line of the triage. That is the whole argument.""")
    step("sweep the caseload")
    cli("sweep", LETTERS.parent / "caseload", "--scripted",
        "--classifications", CLASSIFICATIONS / "caseload.json", expect=2)
    print("""
  Each of those 24 documents now has its own durable case on disk. The ones
  needing a decision can be answered later, in any order, by a different
  process - which is the next case.""")


def golden_a() -> None:
    banner(
        "GOLDEN A - a discrepancy the deterministic layer could not have found alone",
        "letter 07: real rating-schedule nerve terminology, sides stated",
    )
    print("""
  The letter lists four evaluations and states a combined 70%.

  Two of the four conditions are named with real VA rating-schedule
  terminology - "median nerve" (DC 8515) and "musculospiral nerve"
  (DC 8514). The deterministic lexicon ABSTAINS on both: it has no opinion,
  rather than a wrong one. Classifying them as upper-extremity conditions is
  what makes 38 CFR 4.26 applicable, and that is the model's entire job here.

  Watch the actor tags. Percentages are DETERMINISTIC. The extremity groups
  are AI. The arithmetic is DETERMINISTIC.""")
    step("audit the letter")
    cli("audit", LETTERS / "07_nerve_terminology.txt", "--case", "golden-a",
        "--scripted", "--classifications", CLASSIFICATIONS / "07_nerve_terminology.json",
        expect=0)


def golden_b() -> None:
    banner(
        "GOLDEN B - the agent stops and asks, then survives being closed",
        "letter 08: same conditions, but the letter never says which side",
    )
    print("""
  Identical medicine, one word missing. The letter says "the median nerve"
  and "the musculospiral nerve" without stating left or right.

  The model can classify both as upper-extremity conditions. It cannot know
  which side they are on, and neither can any amount of reasoning - only the
  veteran knows. 4.26 requires a compensable disability in EACH of two paired
  extremities, so this is unanswerable from the document.

  So the agent stops. Note what it asks for: a side, and nothing else. It
  never asks a human for a percentage or a combined evaluation.""")
    step("PROCESS A: audit, hit the ambiguity, persist, exit")
    cli("audit", LETTERS / "08_nerve_no_side.txt", "--case", "golden-b",
        "--scripted", "--classifications", CLASSIFICATIONS / "08_nerve_no_side.json",
        expect=2)

    print("""
  That process has now exited. Everything needed to continue is on disk.""")
    for path in sorted((STORE / "golden-b").rglob("*.json")):
        print(f"      {path.relative_to(ROOT)}")
    time.sleep(0.4)

    step("PROCESS B: a different interpreter, hours later, answers the question")
    cli("resume", "--case", "golden-b", "--answer", "2=left,3=right",
        "--scripted", "--classifications", CLASSIFICATIONS / "08_nerve_no_side.json",
        expect=0)
    print("""
  The human supplied two words. Deterministic code did the rest: identified
  the pair under 4.26(a) and (c), added the bilateral factor, folded the
  values through Table I, and converted to the final degree.

  The human never touched a number.""")


def golden_c() -> None:
    banner(
        "GOLDEN C - the control: Recheck is not built to manufacture errors",
        "letter 06: the VA's arithmetic is correct",
    )
    print("""
  A tool that always finds a problem is a tool nobody should trust. Here the
  stated evaluation and the recomputation agree, no human is asked anything,
  and the report says so plainly.""")
    step("audit a letter with correct arithmetic")
    cli("audit", LETTERS / "06_agrees.txt", "--case", "golden-c", "--scripted", expect=0)


def closing() -> None:
    banner("WHAT JUST HAPPENED")
    print("""
  Across the caseload: 24 documents in, 4 questions out.

  DETERMINISTIC   parsed the percentages and the stated combined value;
                  recognised the anatomy it actually knows; identified the
                  4.26 pair; did every piece of arithmetic. Verified against
                  all 684 published cells of the Table I combined ratings
                  table and against the worked examples written into 4.25
                  and 4.26 themselves.

  AI              classified condition names the lexicon cannot reach.
                  Measured: that lexicon gets 22 of 138 real rating-schedule
                  condition names right, and abstains on the rest. The model
                  cannot express a percentage - the field does not exist in
                  its schema - and a side it asserts is rejected unless the
                  word appears in the source document.

  HUMAN           supplied laterality, and only laterality, for the one
                  question that is genuinely unanswerable from the paper.

  Nothing in this demonstration used a network, a credential, or a paid
  inference call. The same graph runs against a live provider by swapping one
  adapter at the edge.""")
    print(WIDE)


CASES = {"sweep": golden_sweep, "a": golden_a, "b": golden_b, "c": golden_c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default=None)
    ap.add_argument("--keep", action="store_true", help="keep the demo case store")
    args = ap.parse_args()

    if STORE.exists() and not args.keep:
        shutil.rmtree(STORE)

    banner("RECHECK", "an audit aid for VA combined disability ratings - synthetic data only")
    print("""
  A veteran's combined rating is not the sum of their evaluations. Under
  38 CFR 4.25, 60% and 20% combine to 68%, not 80%, because each disability
  applies to the efficiency that REMAINS. And under 4.26, disabilities of
  paired extremities get an extra 10% of their combined value added - not
  combined - before anything else happens.

  Whether that factor applies can move the final rating by a full 10-point
  band. It depends on a judgment about which conditions affect paired
  extremities, and letters do not always say.""")

    selected = [CASES[args.case]] if args.case else [golden_sweep, golden_a, golden_b, golden_c]
    for case in selected:
        case()
    if not args.case:
        closing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
