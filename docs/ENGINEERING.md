# Engineering record

What was found wrong, how, and what changed. Every entry has a regression
test. The regulatory-engine and final-pass tables name them, and the red
team's are in `tests/test_rt_*.py`.

## How defects were found

1. **Building against the regulation text** — Table I, 4.25 and 4.26 worked examples.
2. **Attacking the running product** — corrupt state, oversized input, hostile answers.
3. **A hostile judging panel** — eight independent reviewers, each attacking one question (Strands usage, is the AI needed, is the human load-bearing, clarity, every claim, architecture theatre, impact, regulatory correctness), with every finding independently re-verified before it was acted on.
4. **Reviewing the fixes** — a test-suite rewrite, mutation-tested by a separate reviewer; a fixture and lexicon rework, checked value by value against the eCFR text.
5. **An adversarial red team** against the release candidate, then a final regression pass over the merged fixes.

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
| Two stores of graph state; resume worked with Strands state deleted; the "decisive" persistence test checked a file-exists call | resume ran from case.json alone | FileSessionManager is the only store of graph state. Resume requires an open question in the restored session; a case whose answers were already accepted is finished from case.json with `resume --case ID` |
| **Re-running a sweep crashed**, erased answers and reported "24 resolved", exit 0 | a restored interrupt met a fresh prompt (`TypeError`) | existing cases are reported as they stand, and a case that cannot be loaded is reported under COULD NOT PROCEED, not deleted; `--fresh` re-audits |
| **Resume silently did nothing** after the gates were rewritten | Strands re-evaluates edge conditions when persisting the session; a condition that flipped after its node ran emptied the resume point | gate conditions are stable: extraction reads GraphState; compute accepts only states that never revert |
| A session edited to continue at `compute` computed an open case | hand-edited `multi_agent.json` | the compute node refuses unless the case is ready |
| Corrupt state reported success; unbounded reads; framework noise | D1–D4 of the first security review | success is a product outcome. Documents are capped at 8 MB, 100 PDF pages, 500,000 characters of text, 2 MB per inflated PDF stream and 4 MB of page content in total; PDF text is read in a child process stopped after 20 s. Framework logging is suppressed unless `--debug`; even then the HTTP and signing loggers stay at INFO and credential-shaped text is masked. State shape and values are validated |
| Failed documents counted as "resolved"; an all-failed sweep exited 0 | sweep summary arithmetic | honest counts; exit 3 when nothing waits and something failed |
| "evidence" anywhere in the text cut the ratings list; identical table rows collapsed | "with x-ray evidence of arthritis"; two separately rated scars | headings matched only as whole lines; rows never deduplicated; a numbered list with a gap is refused |
| The advertised wall-clock timeout did not bound time | a 1 s budget returned after 6–8 s against a stalled real Agent | the call runs through `Agent.stream_async` on its own thread and event loop. The caller waits with `asyncio.wait_for`, sets Strands' `cancel_signal` at the budget, and discards any later answer, even from a model that swallows cancellation (red team MODEL-RT-P2P6-08). Provider clients get the budget and one attempt |
| Synthetic letters used ratings the schedule cannot produce | PTSD at 60% (the mental-disorders formula has no 60); lumbosacral strain at 30%; bilateral musculospiral neuritis at 50% | every committed value checked against 38 CFR Part 4 |

## Red team

Five attackers went at the release candidate (af714cb), one each on arithmetic, the model boundary, the human loop, files and state, and secrets and network. Each finding was reproduced independently before it was assigned. There were **53 findings**: 6 P0, 17 P1, 18 P2 and 12 P3. Some are one defect seen from two angles.

Four agents fixed them, each in a separate part of the code. A separate reviewer then attacked each fix again. The review found one new P1 and one data-loss regression introduced by the first fixes, plus leftover gaps. A second round fixed those. A final regression pass then attacked the merged code where the fixes met each other; its findings are listed below and were fixed too. There are **391 regression tests** in `tests/test_rt_*.py`. Each one fails on the code before its fix, except the controls, which say so.

### What was wrong, and what changed

| Defect class | As found | Fix |
|---|---|---|
| **A partial reading reported as a result** (ARITH-F1, F2, F5, F7, F9; MODEL-RT-P2P6-01) | The parser silently skipped a rating sentence in a wording it did not read ("An evaluation of 10 percent is assigned"), the second stage of a staged rating, a wrapped last table row, and rows after a mid-list "Evidence" heading. It then computed a confident figure from the rest, exit 0. | Every percentage in the decision section must be accounted for, in any spelling ("10-percent", "ten (10) percent", "pct", "٪"), or the letter is refused. Also refused: a condition rated twice, two different current combined figures, a rating after a stop heading that repeats nothing above it, and a count of numbered lines that differs from the rows read. Round 2: the first fix turned a hard-wrapped "(DC 5260)" line into a condition named "is granted". That is fixed. |
| **Letter text inside a condition name** (MODEL-RT-P2P6-02, SECRETS-F2) | The veteran's name, file number and date were joined to a prose condition name and sent to the model. | Sentences no longer join across a blank line, a heading or a "Label:" line. A name with a colon, a percentage, a long number or non-Latin letters is refused. classify.py also withholds any name that carries a percentage, a label, a date or a long number. Final pass: an unfinished line that does not lead into the next ("Dear Mr. Right,", a name and address block) no longer joins a rating sentence with no lead-in; such a rating is refused. |
| **Disguised words** (MODEL-RT-P2P6-04) | A ligature, soft hyphen or zero-width space inside "knee" hid it from the lexicon and from the "none" veto. The result was a wrong figure, exit 0. | Text is normalised before matching (NFKD, with combining marks and format characters dropped). Words mixing alphabets are refused. The veto normalises first, and also vetoes letters outside Latin-1. |
| **Out-of-range or hostile documents** (FILES-P4-01, SECRETS-F3, FILES-P4-05) | "101%" crashed `audit` with a traceback. A 68 KB PDF inflated to 5 GB over 11 minutes. A 2.7 KB PDF that re-invoked one form kept pypdf busy for 95 s. The prose patterns ran in quadratic time. | Values over 100% are refused at extraction. New caps: 500,000 characters of text, 2 MB per inflated stream, 4 MB of page content. PDF text is read in a child process that is killed after 20 s. The patterns now run in linear time. |
| **Model output accepted without complaint** (MODEL-RT-P2P6-06, -07, -09; SECRETS-F5) | With two structured answers, the first won. With a repeated JSON key, the last won, lifting a confidence of 0.10 to 0.95. `true` and `"0.9"` were accepted as confidences. Provider error text ("VA is wrong…") was copied into the reviewer's question and case.json. | The raw tool-use JSON is re-parsed, and a repeated key or a second answer discards the output. The schema is strict. Notes use Recheck's own wording, with only the exception class name and a known stop reason, and never quote the name the model echoed back. |
| **Time and size of the call** (MODEL-RT-P2P6-08, -10) | A blocking model held a 1 s budget for 6 s, and its late answer was used. More than 40 names, or one name over 200 characters, guaranteed a wasted call. | The call runs through `Agent.stream_async` on its own thread and event loop, and a late answer is discarded. Names the answer could not hold are not sent. max_tokens is now 4096, which fits the largest valid answer. |
| **Credentials and endpoints** (SECRETS-F1, F6, F8, F9, F10) | `--debug` printed the AWS access key id and session token. Bedrock preflight probed EC2 instance metadata even with no credentials. A password in RECHECK_OLLAMA_HOST was printed. ANTHROPIC_BASE_URL, or a profile `endpoint_url`, sent the credential elsewhere while preflight said "provider default". | Under `--debug`, signing and header loggers stay at INFO, and a filter masks anything shaped like a credential. Instance metadata is used only if the operator opts in. URLs are shown through one strict parse with user:password masked, and ambiguous URLs are refused. An Anthropic override must be https or loopback. For Bedrock, the endpoint botocore resolves must be the regional https AWS endpoint. RECHECK_PROVIDER is documented as the default for preflight only. |
| **Case files trusted** (FILES-P4-02, -03, -04; HUMAN-F7) | A hand-edited "complete" 90% was printed as the engine's result. One field of the wrong type stopped the whole sweep. A forged case_id overwrote a different case. | Loading a case checks the type of every field. It re-derives a complete result, and every arithmetic line of its trace, from the stored facts. The file must name its own case. One bad case is listed under COULD NOT PROCEED and the sweep goes on. |
| **Concurrent and interrupted runs** (HUMAN-F1, F2, F3, F4, F5, F14) | Two resumes at once corrupted case.json. `--fresh` on a file that could not be deleted left a half-deleted case. One failed resume lost its question for good. A killed resume threw away an accepted answer. A letter edited after its audit was still reported. Round 2: a sweep deleted an answered case whose id matched another file's id apart from letter case. | Each case has an OS lock, and each save uses its own temporary file. Discard renames the case aside before deleting it. Resume snapshots case.json and the session and restores both if the run fails (Ctrl+C included). `resume --case ID` with no answer finishes a case whose answers are on file. `show` and `sweep` never offer a question the session does not hold. The letter's SHA-256 is checked. A sweep never deletes a case it cannot load unless given `--fresh`, and case ids are compared ignoring letter case. |
| **Enumeration crashes and run time** (ARITH-F3, F8; HUMAN-F6; MODEL-RT-P2P6-03) | Four condition names outside the lexicon crashed the assess node (IndexError) and left the case stuck. A 12-row letter took 8.4 minutes of CPU. Round 2: 13 arm and leg ratings crashed it the same way. | A letter ends UNDETERMINED, before any arithmetic, when it has more than 4,096 answer combinations, needs more than 2^20 subsets of 4.26(d) work, or has more than 12 arm and leg ratings. The 4.26(d) arithmetic is cached: the 8.4-minute letter now takes about 3.5 s. |
| **Single evaluations covering both sides** (ARITH-F4, F6; HUMAN-F12) | The 4.26(d) search left a lone "bilateral" evaluation in the factor and reported 100% on a reading VA's manual leaves open. Where M21-1 does settle the reading, the case still came back UNDETERMINED, or asked a question no answer could change. | The reading is chosen for each combination of answers, as M21-1 V.iv.1.C.4.b settles it, with evaluations compared by position. The strict reading is also run, and a result that depends on the choice is UNDETERMINED. |
| **Answers and printed commands** (HUMAN-F8 to F11, F13, F15) | Answering "lower" without a side ended the case for good. A null answer counted as "none". Usage errors exited 2, the code for "waiting on an answer". A letter named `a$(touch PWNED).txt` printed a command that ran `touch` when pasted. Case ids starting with "-", and Windows backslashes, broke pasted commands. | A partial answer brings a follow-up question (exit 2). Only text is accepted as an answer. Usage errors exit 3. A path goes into a printed command only if no shell would interpret it; otherwise a placeholder stands in and a note names the path. Paths print with forward slashes. |
| **Reporting** (SECRETS-F4, F7, F11, F12, F13; FILES-P4-06; MODEL-RT-P2P6-05) | The 4.26(d) caveat was missing from `--brief` and from the triage. A verdict resting on AI groups did not say so. The session path repeated the case id, went past Windows' 260-character path limit, and the failure was reported as "could not establish enough facts". The scripted MAP fixture classified a foot injury as an arm from its linked clause. | Every verdict and triage row that 4.26(d) decides carries the caveat, with the figure under the prior rule. The verdict names "the AI classification of [i]". The session id is short, and a path that is too long is refused before anything is written. The fixture reads only the rated condition and does not classify names that join two conditions. |

### What held

- **Arithmetic.** 3,060 generated letters were run through the CLI, and 58,325 engine fact sets were checked against an independent oracle. No complete case had a wrong degree, and no rating list was misread. 192 cases stopped for an answer. Resumed with the true facts, 187 gave the oracle's degree. Four ended UNDETERMINED because of ARITH-F6 (fixed above), and one was a genuinely open case.
- **Model boundary.** Every malformed payload was rejected after exactly one call. That covered extra fields, NaN or 1e308 confidences, lookalike group names, tool input that was not an object, 41 items, and prompt-injection names. The one-turn limit held.
- **Network and credentials.** The zero-model path made no connection or DNS attempts. Planted credentials never appeared in any output or file, apart from SECRETS-F1. Re-checked on the merged code with `--debug` and planted AWS and Anthropic keys: none appeared.
- **Human loop.** All 31 hostile `--answer` strings were rejected, and each question stayed open. A fact the letter states could not be overridden. No figure was ever computed while a question was open. Tampered Strands sessions were refused without producing a figure.
- **Wording.** Outside the disclaimer, no report says "wrong", "error", "owed" or "appeal".

### Final regression pass

| Finding | As found | Fix | Test |
|---|---|---|---|
| RG-01 salutation joined to a condition name | "Dear Mr. Right," above "Knee strain is continued as 20 percent" gave the knee a right side: 50% POTENTIAL DISCREPANCY, exit 0. Name and address lines were sent to the model | an unfinished line that does not end on a joining word ("for", "of", "is" …) does not join an upper-case line below it; a rating there with no lead-in is refused | `test_rt_final.py::test_an_unfinished_line_above_a_rating_sentence_is_not_its_condition_name` |
| RG-02 quadratic restatement key | a 256 KB name of repeated "(AB)" took 438 s | the acronym check reads only the words before it | `test_the_restatement_key_is_linear_in_the_number_of_acronyms` |
| RG-03 a second evaluation dropped as a restatement | a rating under DC 7805 after a stop heading was taken as the DC 7804 row above it: 30% NO DISCREPANCY instead of a refusal | a statement under different diagnostic codes is not a restatement | `test_a_statement_under_another_code_after_a_stop_heading_is_not_a_restatement` |
| RG-04, RG-09 wording | a letter over the 12-rating limit, with no fact unknown, read "too many facts are unknown"; triage rows cut the reason mid-word | "No possible ratings were computed" with the reason; rows wrap | `test_an_over_the_member_limit_case_does_not_blame_unknown_facts` |
| RG-05 open reading in a "complete" case | a complete case edited to a single both-sides evaluation M21-1 leaves open still loaded and printed | a complete case must be settled on its facts, whatever makes it open | `test_a_complete_case_whose_m21_reading_is_open_is_corrupt` |
| RG-06 forged trace entries | "Final degree: 90%" or "Result: VA erred" added to a trace printed at exit 0 | a trace entry must use an action a node writes; a test re-derives that set from the source | `test_a_trace_entry_under_an_unknown_action_is_refused` |
| RG-10 PDF child | Ctrl+C waited for the whole 20 s budget; a parent killed outright left pypdf running | short waits with a kill in `finally`; the child ends itself at its budget | `test_the_pdf_child_ends_itself_at_its_budget` |
| RG-11 a question no answer could change | a 0% condition with no side was listed in the question, and an answer leaving it out was refused | 0% conditions are neither asked nor required (4.26(c)); they may still be answered | `test_a_non_compensable_unknown_is_not_asked_and_not_required` |
| RG-12 preflight | a passing preflight did not say `audit` and `sweep` ignore the provider without `--model` | it says so | `test_a_ready_preflight_says_audit_uses_the_provider_only_with_model` |
| RG-13 code lists | "(DCs 5260-5261)", "(DC 5260, 5261)" refused as a long number | read as diagnostic code references | `test_every_spelling_of_a_code_list_is_a_code_reference` |

### Known residuals

- **A confident wrong classification is used.** The floor and the veto stop uncertain answers and a "none" for a limb or nerve word. A wrong "upper" at confidence 0.9 still gets through. The verdict then names the AI classification it rests on.
- **Text inside a table row's name is part of the name.** A row "Cubital tunnel syndrome Jane Q Veteran ...... 10%" is read with the whole name and sent to the model with it, unless it carries a label, a date or a long number.
- **A staged rating under two different names reads as two ratings.** For example, "left knee strain, 20%" followed later by "limitation of flexion of the left knee, increased to 30%". Text alone cannot tell this apart from two separately rated conditions of the same knee.
- **Some letters cannot be finished by answering.** Five compensable names outside the lexicon, with no classifier, make 16,807 combinations. That letter ends UNDETERMINED without a question.
- **Letters near the work budget are slow.** The slowest letter under the budget took 10 to 13 s to audit.
- **Provider SDK logs.** Under `--debug`, Recheck masks credential-shaped text it can see, but not everything the provider SDKs log, so treat a `--debug` log as sensitive. Preflight refuses a Bedrock VPC endpoint or gateway.
- **Not exercised on POSIX.** The lock's delete-before-release order was worked out on paper and tested with a monkeypatch on Windows, not on a POSIX machine. The letter's path is stored as given, so a resume from another working directory cannot check the letter's digest and goes ahead anyway.
- **The parser refuses some ordinary wording.** Hard-wrapped all-caps prose, a year inside a condition name, or a percentage in an unrelated paragraph refuses the whole letter. That fails closed, but it is a letter a reviewer checks by hand.
- **Minor.** A custom Model's exception class name can appear in the note.
