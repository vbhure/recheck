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

  1. Post-traumatic stress disorder (DC 9411) ................ 60%
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

Service connection for post-traumatic stress disorder is granted with an
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

Evaluation of post-traumatic stress disorder, currently evaluated as
30 percent disabling, is increased to 60 percent effective February 1, 2026.

Evaluation of limitation of flexion of the right knee, currently evaluated
as 10 percent disabling, is increased to 20 percent effective February 1, 2026.

Service connection for limitation of flexion of the left knee is granted
with an evaluation of 10 percent effective February 1, 2026.

Evaluation of tinnitus is continued as 10 percent disabling.

REASONS FOR DECISION

An evaluation of 70 percent is assigned for post-traumatic stress disorder
only where there is occupational and social impairment with deficiencies in
most areas. A 100 percent evaluation requires total occupational and social
impairment. The evidence does not show this level of impairment.

Your combined evaluation for compensation is 70 percent.
"""

LETTERS["04_cross_bodypart"] = f"""{BANNER}
DEPARTMENT OF VETERANS AFFAIRS
Regional Office

Name: T. SYNTHETIC
File Number: 00-000-004
Date of Notification: June 11, 2026

DECISION

Service connection for post-traumatic stress disorder is granted with an
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

Service connection for post-traumatic stress disorder is granted with an
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
  2. Lumbosacral strain (DC 5237) ............................ 30%

COMBINED EVALUATION FOR COMPENSATION: 70%
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
