# Caseload: expected outcomes derived by hand

All letters are synthetic. The caseload's mix is configured in
`tools/make_caseload.py` and is not a sample of real decisions.

The eight letters below each pin one behaviour. Their stated values and
expected outcomes were worked out by hand from 38 CFR 4.25 (Table I: each
step is `running + (100 - running) x rating%`, rounded half up; the final
combined value is converted to the nearest 10, with 5 rounding up) and
38 CFR 4.26 (combine every compensable disability of a qualifying pair, then
add 10% of that value, rounded; the result is one disability). Nothing here
was produced by `recheck.cfr`. `tests/test_caseload_expectations.py` sweeps
the committed caseload with the committed classification fixture and asserts
these outcomes.

Every evaluation is one the rating schedule can produce; the diagnostic code
is noted beside it.

The other 16 letters are ordinary agree / discrepant letters whose stated
values come from the generator's own Table I arithmetic (not imported from
the engine). They are not pinned here.

---

## (i) case_020 - three leg disabilities: the whole group, not one pair

| evaluation | % | code |
|---|---|---|
| post-traumatic stress disorder | 30 | DC 9411 |
| limitation of flexion of the left knee | 20 | DC 5260, flexion limited to 30 degrees |
| limited motion of the left ankle | 20 | DC 5271, marked |
| limitation of flexion of the right knee | 30 | DC 5260, flexion limited to 15 degrees |
| tinnitus | 10 | DC 6260 |

Stated: 80%.

Both legs have a compensable disability, so all three leg disabilities enter
the factor (4.26(a), (c)).

    30, 20, 20:   30 + 70 x 0.20 = 44;   44 + 56 x 0.20 = 55.2 -> 55
    factor:       55 + 5.5 = 60.5 -> 61
    61, 30, 10:   61 + 39 x 0.30 = 72.7 -> 73;   73 + 27 x 0.10 = 75.7 -> 76
    final degree: 76 -> 80

4.26(d) alternatives, none more favourable:

    factor on left knee 20 + right knee 30 only (left ankle combined separately):
      30, 20 -> 44; 44 + 4.4 = 48.4 -> 48
      48, 30, 20, 10 -> 63.6 -> 64 -> 71.2 -> 71 -> 73.9 -> 74 -> 70
    factor on left ankle 20 + right knee 30 only: the same numbers -> 70
    no factor: 30, 30, 20, 20, 10 -> 51 -> 60.8 -> 61 -> 68.8 -> 69 -> 72.1 -> 72 -> 70

**Expected: complete, 80%, no discrepancy.** An engine that paired only one
left/right pair would compute 70% and flag a correct letter.

## (ii) case_021 - four extremities, one group (4.26(b))

| evaluation | % | code |
|---|---|---|
| lumbosacral strain | 20 | DC 5237 |
| right cubital tunnel syndrome | 10 | DC 8516 by analogy, mild |
| left cubital tunnel syndrome | 10 | DC 8516 by analogy, mild |
| radiculopathy of the right lower extremity | 20 | DC 8520, moderate |
| radiculopathy of the left lower extremity | 10 | DC 8520, mild |

Stated: 50%. The arm conditions are clinical names the lexicon abstains on;
the classifier fixture establishes "upper".

    20, 10, 10, 10: 20 + 80 x 0.10 = 28;  28 + 72 x 0.10 = 35.2 -> 35;  35 + 65 x 0.10 = 41.5 -> 42
    factor:         42 + 4.2 = 46.2 -> 46
    46, 20:         46 + 54 x 0.20 = 56.8 -> 57
    final degree:   57 -> 60

4.26(d) alternatives (a kept set must still cover both sides of each pair it
contains):

    legs only:  20, 10 -> 28 + 2.8 = 30.8 -> 31;  31, 20, 10, 10 -> 44.8 -> 45 -> 50.5 -> 51 -> 55.9 -> 56 -> 60
    arms only:  10, 10 -> 19 + 1.9 = 20.9 -> 21;  21, 20, 20, 10 -> 36.8 -> 37 -> 49.6 -> 50 -> 55 -> 60
    no factor:  20, 20, 10, 10, 10 -> 36 -> 42.4 -> 42 -> 47.8 -> 48 -> 53.2 -> 53 -> 50

**Expected: complete, 60%, potential discrepancy (higher than stated).**

## (iii) case_010 - linked clause: the rated condition comes first

| evaluation | % | code |
|---|---|---|
| Lumbosacral strain | 20 | DC 5237 |
| Radiculopathy, left lower extremity, associated with lumbosacral strain | 20 | DC 8520, moderate |
| Radiculopathy, right lower extremity, associated with lumbosacral strain | 10 | DC 8520, mild |

Stated: 50%.

    factor:       20, 10 -> 28; 28 + 2.8 = 30.8 -> 31
    31, 20:       31 + 69 x 0.20 = 44.8 -> 45
    final degree: 45 -> 50
    no factor:    20, 20, 10 -> 36 -> 42.4 -> 42 -> 40

**Expected: complete, 50%, no discrepancy.** Reading the linked clause
("lumbosacral strain") as the rated condition would call both radiculopathies
"not an arm or leg", drop the pair, and flag a correct letter at 40%.

## (iv) case_001 - side not stated, and it cannot matter

| evaluation | % | code |
|---|---|---|
| post-traumatic stress disorder | 70 | DC 9411 |
| limitation of flexion of the knee | 10 | DC 5260 |
| tinnitus | 10 | DC 6260 |

Stated: 80%.

    70, 10, 10:   70 + 30 x 0.10 = 73;  73 + 27 x 0.10 = 75.7 -> 76
    final degree: 76 -> 80

There is no other arm or leg disability for the knee to pair with, so left
and right both give 80%.

**Expected: complete, 80%, no question asked; the knee (index 1) is recorded
as an immaterial unknown.**

## (v) case_012 - sides not stated, and they matter

| evaluation | % | code |
|---|---|---|
| post-traumatic stress disorder | 50 | DC 9411 |
| limitation of flexion of the knee | 20 | DC 5260 |
| plantar fasciitis | 10 | DC 5269 |

Stated: 60%.

    opposite legs: 20, 10 -> 28 + 2.8 = 30.8 -> 31;  50 + 50 x 0.31 = 65.5 -> 66 -> 70
    same leg:      50, 20, 10 -> 60 -> 64 -> 60

**Expected: awaiting an answer; possible results 60% or 70%; the question
asks for the side of indices 1 and 2 only.**

## (vi) case_019 - a clinical term the classifier fixture does not list

| evaluation | % | code |
|---|---|---|
| post-traumatic stress disorder | 50 | DC 9411 |
| limitation of flexion of the right knee | 20 | DC 5260 |
| left Lisfranc injury | 10 | DC 5284, moderate |

Stated: 60%. The lexicon abstains on "Lisfranc injury"; the fixture does not
list it, answers at confidence 0.0, and the confidence floor refuses that.
Its extremity group stays unknown. Its side is stated.

    if a leg:           20, 10 -> 31;  50, 31 -> 65.5 -> 66 -> 70
    if an arm, or none: 50, 20, 10 -> 64 -> 60

**Expected: awaiting an answer; possible results 60% or 70%; the question
asks for the extremity group of index 2 only.**

## (vii) case_015 - stated HIGHER than the regulation gives

| evaluation | % | code |
|---|---|---|
| post-traumatic stress disorder | 50 | DC 9411 |
| tinnitus | 10 | DC 6260 |
| limitation of flexion of the right knee | 10 | DC 5260, 45 degrees |
| limited motion of the right ankle | 10 | DC 5271, moderate |

Stated: 70%. Both leg disabilities are on the right; with nothing on the
left, 4.26 does not apply (4.26(c)).

    50, 10, 10, 10: 50 + 50 x 0.10 = 55;  55 + 4.5 = 59.5 -> 60;  60 + 4 = 64
    final degree:   64 -> 60

The stated 70% is what a factor wrongly applied to the two right-leg ratings
gives (10, 10 -> 19 + 1.9 = 20.9 -> 21; 50, 21, 10 -> 60.5 -> 61 -> 64.9 ->
65 -> 70).

**Expected: complete, 60%, potential discrepancy (lower than stated).**

## (ix) case_014 - clinical names with no side stated

| evaluation | % | code |
|---|---|---|
| post-traumatic stress disorder | 50 | DC 9411 |
| cubital tunnel syndrome | 20 | DC 8516 by analogy |
| De Quervain's tenosynovitis | 10 | DC 5024 on DC 5215 |

Stated: 60%. The classifier fixture establishes "upper" for both; only a
reviewer can say which arm.

    opposite arms: 20, 10 -> 31;  50, 31 -> 65.5 -> 66 -> 70
    same arm:      50, 20, 10 -> 64 -> 60

**Expected: awaiting an answer; possible results 60% or 70%; the question
asks for the side of indices 1 and 2 only.**
