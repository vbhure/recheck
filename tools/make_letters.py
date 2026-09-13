"""Generate heterogeneous synthetic VA rating decision letters.

Every letter here is SYNTHETIC. No real veteran data is used anywhere in
this project. Names, file numbers and dates are invented.

The set is deliberately varied, because the whole question this project has
to answer honestly is: "where does deterministic parsing actually stop being
enough?" A fixture set that is all one format cannot answer that.

Formats included:
  01  tabular          - clean itemised list with diagnostic codes
  02  prose            - narrative DECISION section, no table
  03  history_trap     - prose containing PRIOR evaluations as well as current
  04  cross_bodypart   - bilateral pair spanning different body parts
  05  missing_side     - laterality genuinely absent from the text
  06  agrees           - control case where the VA's arithmetic is correct
  07  clinical_terms   - clinical names outside the schedule's vocabulary, sides stated
  08  clinical_terms_no_side - the same letter with no side stated
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "letters"

BANNER = (
    "*** SYNTHETIC DOCUMENT - NOT A REAL VA DECISION - NO REAL VETERAN DATA ***"
)

LETTERS: dict[str, str] = {}

LETTERS["01_tabular"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: J. SYNTHETIC
File Number: 00-000-001
Date of Notification: March 14, 2026

                         RATING DECISION

  1. Asthma, bronchial (DC 6602) ............................ 60%
  2. Limitation of flexion, right knee (DC 5260) ............. 20%
  3. Limitation of flexion, left knee (DC 5260) .............. 10%
  4. Tinnitus (DC 6260) ...................................... 10%

COMBINED EVALUATION FOR COMPENSATION: 70%
"""

LETTERS["02_prose"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: R. SYNTHETIC
File Number: 00-000-002
Date of Notification: April 2, 2026

DECISION

Service connection for bronchial asthma is granted with an
evaluation of 60 percent effective January 9, 2026.

Service connection for limitation of flexion of the right knee is granted
with an evaluation of 20 percent effective January 9, 2026.

Service connection for limitation of flexion of the left knee is granted
with an evaluation of 10 percent effective January 9, 2026.

Service connection for tinnitus is granted with an evaluation of 10 percent
effective January 9, 2026.

Your combined evaluation for compensation is 70 percent.
"""

LETTERS["03_history_trap"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: M. SYNTHETIC
File Number: 00-000-003
Date of Notification: May 20, 2026

DECISION

Evaluation of bronchial asthma, currently evaluated as
30 percent disabling, is increased to 60 percent effective February 1, 2026.

Evaluation of limitation of flexion of the right knee, currently evaluated
as 10 percent disabling, is increased to 20 percent effective February 1, 2026.

Service connection for limitation of flexion of the left knee is granted
with an evaluation of 10 percent effective February 1, 2026.

Evaluation of tinnitus is continued as 10 percent disabling.

REASONS FOR DECISION

An evaluation of 100 percent is assigned for bronchial asthma only where
FEV-1 is less than 40 percent of predicted. A 60 percent evaluation requires
FEV-1 of 40 to 55 percent of predicted. The evidence shows FEV-1 of 48
percent of predicted.

Your combined evaluation for compensation is 70 percent.
"""

LETTERS["04_cross_bodypart"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: T. SYNTHETIC
File Number: 00-000-004
Date of Notification: June 11, 2026

DECISION

Service connection for bronchial asthma is granted with an
evaluation of 60 percent effective March 3, 2026.

Service connection for carpal tunnel syndrome of the left wrist is granted
with an evaluation of 20 percent effective March 3, 2026.

Service connection for painful motion of the right forearm due to residuals
of a healed radial fracture is granted with an evaluation of 10 percent
effective March 3, 2026.

Service connection for tinnitus is granted with an evaluation of 10 percent
effective March 3, 2026.

Your combined evaluation for compensation is 70 percent.
"""

LETTERS["05_missing_side"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: A. SYNTHETIC
File Number: 00-000-005
Date of Notification: July 8, 2026

DECISION

Service connection for bronchial asthma is granted with an
evaluation of 60 percent effective April 15, 2026.

Service connection for degenerative arthritis of the knee is granted with an
evaluation of 20 percent effective April 15, 2026.

Service connection for limitation of motion of the knee is granted with an
evaluation of 10 percent effective April 15, 2026.

Service connection for tinnitus is granted with an evaluation of 10 percent
effective April 15, 2026.

Your combined evaluation for compensation is 70 percent.
"""

LETTERS["06_agrees"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: K. SYNTHETIC
File Number: 00-000-006
Date of Notification: August 1, 2026

                         RATING DECISION

  1. Post-traumatic stress disorder (DC 9411) ................ 50%
  2. Cervical strain (DC 5237) ............................... 30%

COMBINED EVALUATION FOR COMPENSATION: 70%
"""


# The two letters below use condition names the way decision letters actually
# write them - clinical names that are NOT the rating schedule's own
# vocabulary. The deterministic lexicon covers the schedule's anatomy (median
# nerve, sciatic nerve, humerus, ...), so a schedule name would never reach the
# classifier. These two names contain no lexicon word, so the semantic
# classifier is genuinely load-bearing here rather than decorative.
#
# Evaluations are ones the schedule can produce for how these conditions are
# rated (38 CFR Part 4, eCFR text):
#   cubital tunnel syndrome      ulnar nerve entrapment at the elbow, rated by
#                                analogy to DC 8516 ulnar nerve: incomplete
#                                paralysis mild 10/10, moderate 30 major /
#                                20 minor. 20% on the left is moderate, minor
#                                (dominant) extremity.
#   De Quervain's tenosynovitis  DC 5024 tenosynovitis, rated on limitation of
#                                motion of the wrist, DC 5215: 10% major or minor.
#
# Letter 07 states the sides INSIDE the condition names, where the
# deterministic side reader looks. Recomputed by hand, 38 CFR 4.25 / 4.26:
#   bilateral factor, upper extremities (left 20, right 10):
#     20 combined with 10 = 20 + 80 x 0.10 = 28; add 10% of 28 = 2.8 -> 30.8 -> 31
#   60, 31, 10 in order of severity:
#     60 + 40 x 0.31 = 72.4 -> 72; 72 + 28 x 0.10 = 74.8 -> 75
#   combined value 75 -> final degree 80 (values ending in 5 round up)
#   without 4.26: 60, 20, 10, 10 -> 68 -> 71.2 -> 71 -> 73.9 -> 74 -> 70,
#   which is what the letter states.
# Letter 08 is the same letter with no side stated: the classifier can say
# both conditions are of an arm, but only a reviewer can say which arm, and
# the answer moves the result between 70 and 80.

LETTERS["07_clinical_terms"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: D. SYNTHETIC
File Number: 00-000-007
Date of Notification: August 19, 2026

DECISION

Service connection for bronchial asthma is granted with an
evaluation of 60 percent effective May 4, 2026.

Service connection for left cubital tunnel syndrome is granted with an
evaluation of 20 percent effective May 4, 2026.

Service connection for right De Quervain's tenosynovitis is granted with an
evaluation of 10 percent effective May 4, 2026.

Service connection for tinnitus is granted with an evaluation of 10 percent
effective May 4, 2026.

Your combined evaluation for compensation is 70 percent.
"""

LETTERS["08_clinical_terms_no_side"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: P. SYNTHETIC
File Number: 00-000-008
Date of Notification: September 2, 2026

DECISION

Service connection for bronchial asthma is granted with an
evaluation of 60 percent effective June 1, 2026.

Service connection for cubital tunnel syndrome is granted with an
evaluation of 20 percent effective June 1, 2026.

Service connection for De Quervain's tenosynovitis is granted with an
evaluation of 10 percent effective June 1, 2026.

Service connection for tinnitus is granted with an evaluation of 10 percent
effective June 1, 2026.

Your combined evaluation for compensation is 70 percent.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in LETTERS.items():
        path = OUT / f"{name}.txt"
        path.write_text(body, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(body)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
