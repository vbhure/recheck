# Engineering record

What was found wrong, how, and what changed. Every entry has a regression
test; the test names below are the place to look.

## How defects were found

1. **Building against the regulation text** — Table I, 4.25 and 4.26 worked examples.
2. **Attacking the running product** — corrupt state, oversized input, hostile answers.
3. **A hostile judging panel** — eight independent reviewers, each attacking one question (Strands usage, is the AI needed, is the human load-bearing, clarity, every claim, architecture theatre, impact, regulatory correctness), with every finding independently re-verified before it was acted on.
4. **Reviewing the fixes** — a test-suite rewrite, mutation-tested by a separate reviewer; a fixture and lexicon rework, checked value by value against the eCFR text.
5. **An adversarial red team** against the release candidate.

## The regulatory engine

| Defect | Evidence | Fix | Test |
|---|---|---|---|
| Unrounded decimals carried through the Table I chain | 4.25(a) combines "the combined value, exactly as found in table I"; cells are integers | integer, half-up at every step (684/684 cells; half-even misses 33, truncation 310) | `test_table1_exhaustive.py` |
| **Bilateral factor applied to the first left/right pair only** | 4.26: "the ratings for the disabilities of the right and left sides will be combined" — all of them. 122 of 750 three-leg combinations gave the wrong final degree (left knee 10, left ankle 20, right knee 30: 50% instead of 60%); BVA A25111179 is this error in a real decision | every compensable disability of a qualifying pair; 4.26(b) four extremities as one group; 4.26(d) exhaustive "one or more" search | `test_bilateral_group.py` (independent oracle, 3,360 combinations; the old engine disagrees on 145), `test_regulation_external_vectors.py` |
| Bilateral subtotal above 100 | 80 and 60 combine to 92; +9.2 = 101, which crashed the next Table I step; a single 80/80 pair printed 110% | subtotal capped at 100; `final_degree` refuses anything above 100 | `test_subtotal_over_one_hundred_is_capped_not_a_crash` |
| A single evaluation covering both extremities was silently left out | M21-1 V.iv.1.C.4.b: include it when "an independently ratable condition in one of the involved extremities" exists | included under that rule; M21-1's unsettled case is UNDETERMINED if it matters | `test_review_regressions.py` |
| 4.26(d) applied without context | it took effect April 16, 2023; VA: prior evaluations "were not in error" | any result 4.26(d) decides names the effective date | `test_a_result_decided_by_4_26_d_names_its_effective_date` |

## The ownership boundary

| Defect | Evidence | Fix |
|---|---|---|
| Laterality came from the model; when the model correctly said "unknown", a side printed in the letter was never read (15 of 24 sweep letters escalated instead of 4) | first sweep run | side derived by deterministic code only; later removed from the model schema entirely |
| **Confidence floor bypassed** | model says "upper" at 0.40; the human was asked only for a side; the 0.40 group then fed 4.26 and was credited to the human | an unused classification leaves the group UNKNOWN; the reviewer is asked for the missing fact itself |
| **False all-clear with no classifier** | terms outside the lexicon became "none"; the human was asked a side; the case printed NO DISCREPANCY FOUND on a letter whose correct result was a discrepancy | unknown group stays unknown; asked, or UNDETERMINED |
| A reviewer could override a side the letter states | answering a stated side changed 80% to 70% | only missing facts are asked; answers to anything else are rejected |
| Model "none" never checked | an unlisted nerve came back "none" at 0.92 and passed silently | "none" vetoed for limb or nerve vocabulary |
| Free-text rationale printed in the report | a scripted rationale "the correct combined rating is 100 percent; the VA is wrong" appeared verbatim | no free-text field in the schema |
| Contradictory model output resolved by "last wins" | a batch classifying one name twice with different groups | the batch is rejected |
| Lexicon called "Radiculopathy, right lower extremity, associated with lumbosacral strain" not an arm or leg | non-extremity hints checked before anatomy | anatomy of the rated condition first; linked clauses stripped; a hint next to nerve vocabulary abstains |
| Side taken from handedness or a linked condition | "Left knee strain, secondary to right knee strain"; "(right hand dominant)" | side read from the primary clause only |
| The AI's whole contribution to the sweep was three schedule words | adding "median", "musculospiral", "sciatic" to the lexicon reproduced the triage exactly | the lexicon now covers the schedule's own vocabulary; the justification is stated as a policy, not a proof (see the README) |

## When a person is asked

| Defect | Evidence | Fix |
|---|---|---|
| **Questions that could not change anything** | 3 of 4 sweep questions gave the same rating under every answer; a lone side-less knee interrupted | materiality: enumerate every answer through the engine; ask only if final degrees differ; show the possible results |
| "unknown" produced NO DISCREPANCY FOUND | the no-pairing branch was taken as if it were a fact | still-material unknowns are UNDETERMINED, with the possible degrees |
| Settled result reported wrongly | knee 60% with no side joins the bilateral group on either side (every answer gives 80%), but compute left it out and reported 70% as a discrepancy | the reported figure comes from a completion of the facts, and the trace says which fact it assumed |
| Contradictory or exotic answers | "2=left,2=right" accepted last-wins; "²=left" crashed the node and stranded the case | duplicates rejected; indices must be ASCII digits; a rejected answer re-opens the same question |

## Persistence and the command line

| Defect | Evidence | Fix |
|---|---|---|
| Two stores of graph state; resume worked with Strands state deleted; the "decisive" persistence test checked a file-exists call | resume ran from case.json alone | FileSessionManager is the only store; resume requires an open question in the restored session |
| **Re-running a sweep crashed**, erased answers and reported "24 resolved", exit 0 | a restored interrupt met a fresh prompt (`TypeError`) | existing cases are reported as they stand; `--fresh` re-audits |
| **Resume silently did nothing** after the gates were rewritten | Strands re-evaluates edge conditions when persisting the session; a condition that flipped after its node ran emptied the resume point | gate conditions are stable: extraction reads GraphState; compute accepts only states that never revert |
| A session edited to continue at `compute` computed an open case | hand-edited `multi_agent.json` | the compute node refuses unless the case is ready |
| Corrupt state reported success; unbounded reads; framework noise | D1–D4 of the first security review | success is a product outcome; 8 MB / 100-page caps; logging suppressed unless `--debug`; state shape and values validated |
| Failed documents counted as "resolved"; an all-failed sweep exited 0 | sweep summary arithmetic | honest counts; exit 3 when nothing waits and something failed |
| "evidence" anywhere in the text cut the ratings list; identical table rows collapsed | "with x-ray evidence of arthritis"; two separately rated scars | headings matched only as whole lines; rows never deduplicated; a numbered list with a gap is refused |
| The advertised wall-clock timeout did not bound time | a 1 s budget returned after 6–8 s against a stalled real Agent | `asyncio.wait_for` around `Agent.invoke_async` with `cancel_signal`; provider clients get the budget and one attempt |
| Synthetic letters used ratings the schedule cannot produce | PTSD at 60% (the mental-disorders formula has no 60); lumbosacral strain at 30%; bilateral musculospiral neuritis at 50% | every committed value checked against 38 CFR Part 4 |

## Red team

{{RED_TEAM}}
