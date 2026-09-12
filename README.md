# Recheck

**Deterministic verification of VA combined disability ratings, with a narrow AI boundary and human-in-the-loop clarification.**

Recheck is a local-first audit aid for reviewing the arithmetic and the paired-extremity assumptions in a VA disability rating decision. It combines three kinds of reasoning and keeps them strictly separated:

- **Deterministic code** extracts values where the rules are explicit, and performs every calculation.
- **AI** classifies condition terminology that a hand-maintained vocabulary does not cover.
- **A human** resolves ambiguity when the document does not contain enough information to decide safely.

> **AI interprets. The human resolves ambiguity. Deterministic code calculates.**

The language model cannot produce the final percentage. It has no field in which to express one.

---

## Why this exists

A combined rating is not the sum of the individual evaluations. Under **38 CFR 4.25**, each disability applies to the efficiency that *remains*, so 60% and 20% combine to 68%, not 80%. Under **4.26**, disabilities of paired extremities have 10% of their combined value **added — not combined** — before anything else happens.

Whether 4.26 applies can move the final rating by a full 10-point band. It depends on a judgment about which conditions affect paired extremities, and letters do not always state the side.

That creates a review problem with four distinct parts, only one of which is genuinely probabilistic:

| Part | Nature | Owner in Recheck |
|---|---|---|
| Read the individual evaluations | explicit, patterned | deterministic |
| Decide which extremity group a condition affects | open medical vocabulary | AI |
| Decide which side a condition is on | absent from the document | human |
| Combine the values | arithmetic, published table | deterministic |

Recheck separates these rather than asking a model to do all four.

---

## Architecture

See [`docs/architecture.svg`](docs/architecture.svg) for the diagram. In text:

```text
          decision letter (.txt or text-layer .pdf)
                          |
        +-----------------v------------------+
        |  extract                           |   DETERMINISTIC
        |  percentages, stated combined      |   no model, no network
        |  value, recognised anatomy         |
        +-----------------+------------------+
                          |
        +-----------------v------------------+
        |  classify                          |   AI, only for terminology
        |  extremity group + laterality      |   the lexicon abstains on
        |  strict schema, bounded, grounded  |   one batched call per letter
        +-----------------+------------------+
                          |
        +-----------------v------------------+
        |  assess                            |   DETERMINISTIC
        |  safe to compute, or ask a human?  |
        +--------+----------------+----------+
                 |                |
              clear          ambiguous
                 |                |
                 |      +---------v----------+
                 |      |  INTERRUPT         |  state persisted, process exits
                 |      |  human gives a     |  resumed by a NEW process
                 |      |  SIDE, nothing else|  answer validated before use
                 |      +---------+----------+
                 |                |
                 +--------+-------+
                          |
              conditional edge: fail-closed
                          |
        +-----------------v------------------+
        |  compute                           |   DETERMINISTIC
        |  4.26 bilateral factor, then       |   684/684 published cells
        |  4.25 Table I, then final degree   |
        +-----------------+------------------+
                          |
        +-----------------v------------------+
        |  evidence-backed report            |   actor-tagged trace
        +------------------------------------+
```

**Interfaces.** A command-line interface (`recheck audit` / `resume` / `show`) and a plain-JSON case file per run. No web UI.

**AWS.** None is used, and none is required. Bedrock and other providers are reachable through a single adapter at the model boundary; the verified path uses no cloud service at all. See [Model providers](#model-providers).

---

## What the AI does

Exactly one thing: classify a condition name into an extremity group and, where the document supports it, a side.

```text
extremity_group : upper | lower | none
laterality      : left | right | bilateral | unknown
confidence      : 0.0 - 1.0
```

This is needed because the rating schedule's vocabulary is far wider than a small lexicon. Real entries from 38 CFR Part 4 include *genu recurvatum*, *os calcis or astragalus*, *astragalectomy*, *radius and ulna*, *median nerve*, *sciatic nerve*, and *external popliteal nerve (common peroneal)* — terminology that generalises semantically but not lexically.

**Measured, non-circularly.** Against 138 real condition names taken from the rating schedule and labelled by diagnostic-code range — an oracle this project does not control — the 29-term deterministic lexicon classifies **22 correctly (15.9%)**, abstains on 116, and makes **zero wrong-group assertions**.

That last number is what makes the split safe: when the lexicon commits, it is trusted without a model call, because it has never been observed to commit *wrongly*. It abstains instead.

This is stated narrowly on purpose: **the current lexical baseline does not scale to the terminology tested.** It is not a claim that no deterministic approach could. A large curated ontology, or a mapping onto an external terminology such as SNOMED CT, might well achieve high coverage — that is a different system, with an external dependency and a maintenance burden, and it is not what this project has.

Evidence: [`tests/test_ai_justification_gate.py`](tests/test_ai_justification_gate.py), and `python tools/gate_report.py` for the per-letter measurement.

## What the AI does not do

It does not calculate the combined rating, apply the bilateral factor, decide the final percentage, invent laterality, override the document, or reach any legal conclusion.

Three guards sit between the model and the arithmetic:

| Guard | Behaviour |
|---|---|
| **Validation** | Invalid, incomplete or contradictory output is discarded, never repaired. A condition with `extremity_group="none"` may not carry a side. Unexpected fields are rejected rather than silently dropped. |
| **Grounding** | A side the source text does not contain is rejected. Verified: a fabricated `left` at **0.99 confidence** is refused. The document outranks the model; confidence is not evidence. |
| **Floor** | Confidence below 0.75 routes to a human. |

Any of these failing produces `UNKNOWN / HUMAN REVIEW` — never a guessed 4.26 calculation.

---

## Human in the loop

The human is asked one kind of question — which side a condition is on — and supplies one kind of answer. There is no place in the answer grammar for a number, so a human cannot set the outcome directly even deliberately: `2=100` fails as an invalid side.

Answering `unknown` is legitimate. It prevents 4.26 rather than provoking a guess.

The answer is validated before any arithmetic uses it. Rejected: invalid sides, missing answers, indices that do not exist, indices that did not ask for input, and unparseable input. **A rejected answer cannot reach the calculation node** — enforced by a conditional edge in the graph, not by convention.

---

## Cross-process persistence

Interrupt and resume cross a real process boundary.

```text
PROCESS A                          PROCESS B  (a different interpreter)
---------                          ---------
extract                            restore persisted state from disk
classify                           validate the human answer
detect ambiguity                   apply 4.26 where it qualifies
INTERRUPTED                        apply 4.25 Table I
persist state                      compare against the stated value
exit code 2                        report, exit code 0
```

The test suite proves this with real subprocesses rather than two calls inside one interpreter, which would prove nothing. The decisive test **deletes the persisted graph state and shows that resume then fails** — a passing resume alone cannot distinguish reading disk from inheriting memory.

Also covered: unknown case, corrupt JSON, incompatible schema version, and case identifiers that attempt to escape the store directory.

---

## The deterministic engine

`src/recheck/cfr/` contains no model calls, no network access and no I/O. All arithmetic uses `Decimal`.

**The non-obvious detail.** 4.25(a) says the combined value *"exactly as found in table I"* is combined with the next disability, and Table I cells are **integers**. So the regulation rounds to a whole number at every pairwise step; it does not carry unrounded decimals through the chain. The first implementation here got that wrong, and the resulting error was large enough to move a final band.

The rounding mode was not assumed. It was fitted against every published cell:

| Rounding mode | Mismatches against 684 published cells |
|---|---|
| **ROUND_HALF_UP** | **0** |
| ROUND_HALF_EVEN | 33 |
| Truncation | 310 |

The fixture is committed at `fixtures/cfr425_table1_points.json` and regenerated by `python tools/fetch_cfr.py`, which re-extracts from the official eCFR API and diffs against the committed copy. It currently reports byte-identical.

The worked examples written into the regulations are also tests — including 4.26's, which pins every rounding decision in the engine by stating the intermediate values: ratings 60/20/10/10 with the two 10s bilateral give "*the order of severity would be 60, 21 and 20. The 60 and 21 combine to 68 percent and the 68 and 20 combine to 74 percent, converted to 70 percent.*"

4.26 is implemented in full: the factor is **added, not combined**; the subtotal is rounded; 4.26(c) compensability is checked; and **4.26(d)** selects whichever result is more favourable to the veteran when applying the factor would lower the outcome.

---

## The case that motivates the whole design

Ratings `60 / 20 / 10 / 10`.

- Treat the two 10s as the bilateral pair → **70%** (the regulation's own worked example).
- Treat the 20 and a 10 as the pair → **80%**.

Same four numbers. A full band apart. The difference is entirely a judgment about which conditions affect paired extremities.

The point is not that one answer is wrong. It is that **the pairing assumption is consequential and is not always recoverable from the document** — which is why a human is asked, and why the report never says the VA erred.

---

## Report language

The verdict is always framed as:

```text
POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED
```

never as a finding that the decision is incorrect. Recheck cannot see the medical evidence or the rating criteria applied, and a difference may rest on a lawful judgment about paired extremities that Recheck surfaced rather than resolved.

It is not legal advice, not a VA adjudication, and not a claims or appeals service.

---

## Provenance

Provenance is part of the data model, not presentation metadata. Every consequential decision records its actor, and the type system enforces the boundary:

- a confidence may only be attached to an **AI** decision — attaching one to a deterministic or human decision raises
- a regulation citation may only be attached to a **deterministic** decision — the model does not apply regulations
- where no source line was located, the trace prints `evidence: none recorded` rather than inventing one

Real output, from `recheck audit fixtures/letters/07_nerve_terminology.txt`:

```text
[DETERMINISTIC]
  Extracted rating: 20%
      incomplete paralysis of the left median nerve
      (38 CFR 4.25 (individual evaluations as stated); evidence: line 14)
[DETERMINISTIC]
  Extremity group: none
      tinnitus: recognised as a non-extremity condition
      (38 CFR 4.26(c); evidence: line 20)
[AI]
  Extremity group: upper / left
      incomplete paralysis of the left median nerve - the median nerve
      serves the forearm and hand, an upper extremity
      (confidence 0.93; evidence: line 14)
[AI]
  Extremity group: upper / right
      neuritis of the right musculospiral nerve - the musculospiral (radial)
      nerve serves the arm, an upper extremity
      (confidence 0.88; evidence: line 17)
[DETERMINISTIC]
  Bilateral pair identified: 20% + 10%
      same extremity group, opposite sides, both compensable
      (38 CFR 4.26(a), 4.26(c); evidence: none recorded)
[DETERMINISTIC]
  Final degree of disability: 80%
      combined value converted to the nearest degree divisible by 10; values
      ending in 5 are adjusted upward
      (38 CFR 4.25(a); evidence: none recorded)
```

Each run ends with a count, so the division of labour is visible at a glance:

```text
ownership: 14 deterministic, 2 AI, 0 human decision(s)
```

---

## Running it

Python 3.10 or later. No AWS account, no API key, no network.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -e ".[dev]"
```

### The demonstration

```bash
python tools/demo.py
```

Three golden cases, using `ScriptedModel` — no network, no credential, no inference cost. Golden B uses real subprocesses, so the process death and resume are genuine.

| Case | Letter | Outcome | Ownership |
|---|---|---|---|
| A | nerve terminology, sides stated | POTENTIAL DISCREPANCY, 80% vs stated 70% | 14 deterministic, 2 AI, 0 human |
| B | same, side never stated | interrupt → resume in a new process → 80% | 14 deterministic, 2 AI, 2 human |
| C | correct arithmetic | NO DISCREPANCY FOUND | 10 deterministic, 0 AI, 0 human |

Golden C is the one that earns trust. A tool that always finds a problem is a tool nobody should use.

### Individual commands

```bash
# audit a letter
recheck audit fixtures/letters/07_nerve_terminology.txt --case demo1 \
    --scripted --classifications fixtures/classifications/07_nerve_terminology.json

# a letter with missing laterality stops for a human and exits 2
recheck audit fixtures/letters/08_nerve_no_side.txt --case demo2 \
    --scripted --classifications fixtures/classifications/08_nerve_no_side.json

# resume in a fresh process; the human supplies sides only
recheck resume --case demo2 --answer "2=left,3=right" \
    --scripted --classifications fixtures/classifications/08_nerve_no_side.json

# reprint a stored case
recheck show --case demo2
```

`python -m recheck.cli ...` is equivalent if you prefer not to rely on the console script.

Exit codes: `0` completed, `2` awaiting human input, `3` could not proceed.

### Tests

```bash
pytest                                              # everything
pytest --ignore=tests/test_cross_process_resume.py  # skip the subprocess suite
python tools/gate_report.py                         # the AI justification measurement
python tools/fetch_cfr.py                           # re-verify the Table I fixture
recheck preflight --model bedrock                   # check provider config, no inference
```

---

## Model providers

The zero-model path is the architecture. A live provider is an adapter at the edge, and **provider availability never determines whether the core product works.**

`ScriptedModel` (`src/recheck/models/scripted.py`) is **not** a mock that bypasses Strands. It implements `stream()` and emits the same tool-use event sequence a real provider emits, so structured output is parsed and validated by Strands' own machinery. The adversarial tests therefore drive malformed payloads through the real validation path rather than around it.

`src/recheck/models/factory.py` is the live seam. Supported: `bedrock` (bundled), `anthropic` and `ollama` (each needs a pip extra, and the error names the exact install command).

### Credentials are not ours

Recheck never reads, stores, logs or prints a credential. Each provider resolves its own through its own standard mechanism — the AWS credential chain, or `ANTHROPIC_API_KEY`. Preflight reports only whether a credential could be **resolved**, and for AWS it reports the *method* (`via shared-credentials-file`), never any key material. A test asserts that planted credential values cannot appear in preflight output.

### Check configuration without spending anything

```bash
recheck preflight --model bedrock
```

This makes **no inference call**. It verifies the provider is known, the SDK is importable, a model id and region are set, and a credential is resolvable — then states plainly that a live run *will* consume credits.

```text
  [PASS]  provider known              bedrock
  [PASS]  provider SDK importable     strands.models.bedrock.BedrockModel
  [PASS]  region configured           us-west-2
  [FAIL]  credentials resolvable      the AWS credential chain resolved nothing
  [PASS]  single-call contract        one structured classification call per letter
  [PASS]  prompt minimisation         condition names only; no percentages, no letter text
```

Configuration comes from `RECHECK_PROVIDER`, `RECHECK_MODEL_ID`, `RECHECK_REGION`, `RECHECK_OLLAMA_HOST`, `RECHECK_TIMEOUT_S`, `RECHECK_MAX_TOKENS`. See [`.env.example`](.env.example).

### What a live run costs, and what it sends

**One structured call per letter.** No agent loop, no tool use, no re-prompting — `limits={"turns": 1}` bounds it, because Strands otherwise retries a structured output that fails validation.

**Only condition names are transmitted.** The percentages, the stated combined evaluation, the file number and the rest of the letter never leave the machine, because the classification task does not need them. Temperature is 0 for reproducibility.

### Failure is uniform and safe

Timeout, API error, unavailable provider, misconfiguration, malformed output and unsupported content all produce the same outcome: the affected conditions route to `UNKNOWN / HUMAN REVIEW`. Nothing is fabricated or substituted, and no confidence value is invented.

`BoundedAgent` adds a **wall-clock** budget on top of Strands' turn limit, and signals cancellation to the underlying call rather than merely abandoning it — a provider that accepts a connection and then stalls would otherwise hang a terminal tool indefinitely.

---

## Strands usage

| Primitive | Where | Proven by |
|---|---|---|
| `GraphBuilder` / `Graph` | `graph.py:build_graph` | `test_cross_process_resume.py` |
| `add_edge(condition=...)` — the fail-closed safety gate | `graph.py:_safe_to_compute` | `test_safety_gate.py` |
| `set_max_node_executions` | `graph.py:build_graph` | — |
| Custom deterministic `MultiAgentBase` nodes (4) | `graph.py` | `test_cross_process_resume.py` |
| `Interrupt` raised from a node | `graph.py:AssessNode` | `test_cross_process_resume.py` |
| `FileSessionManager` + `serialize_state` / `deserialize_state` | `graph.py`, `cli.py` | `test_resume_requires_the_persisted_state_on_disk` |
| `Agent(structured_output_model=...)` | `classify.py:_ask_model` | `test_boundary_adversarial.py` |
| Custom `Model` implementation | `models/scripted.py` | `test_boundary_adversarial.py` |
| `limits={"turns": 1}` — bounds retry on invalid output | `classify.py:_ask_model` | `test_invalid_payload_routes_every_condition_to_human_review` |
| `callback_handler=None` | `cli.py` | `test_no_framework_chatter_in_product_output` |
| `result.execution_order` | `cli.py:cmd_resume` | — |

Deliberately **not** used: Swarm, A2A, MCP, Cedar, AgentSkills, memory managers, checkpointing and hooks. None of them would earn their place in this workflow, and adding them to lengthen the list would be the opposite of the design principle above. One workflow is enough because the problem is one workflow.

---

## Test suite

**848 tests.** The subprocess suite takes about 40 seconds; everything else runs in under two.

| Area | Coverage |
|---|---|
| Calculation | all 684 published Table I cells · the regulations' worked examples · pairwise rounding mode fitted against the table · final-degree boundaries, monotonicity and range properties · 4.26(c) and 4.26(d) · invalid input |
| Orchestration | graph transitions · ambiguity detection · interrupt · persistence · fresh-process resume · deleted-state resume failure · corrupt and version-mismatched cases · rejected human input · the fail-closed compute gate |
| Model boundary | invalid enums · out-of-range and wrongly-typed confidence · missing fields · extra fields · contradictory output · fabricated laterality at high confidence · provider exceptions · no provider configured · prompt injection in document text |
| Input bounds | oversized text and PDFs refused by name · corrupt and unparseable persisted state · hand-edited case files · framework noise kept out of product output |
| Extraction | tabular and prose formats · hard-wrapped lines · historical-percentage and rating-criteria traps · missing laterality · unparseable documents · PDF text layer parity with plain text · image-only PDF refusal |
| Justification gate | 138 externally-labelled real condition names · coverage, abstention rate, and the zero-wrong-assertion safety property |

---

## Known limitations

Stated as product boundaries, not hidden failures.

- **The fixture letters are synthetic and written by this project.** Extraction is measured as sufficient for the supported formats, which is evidence about those formats and nothing more. Real VA correspondence will contain layouts not represented here.
- **No real VA letters were used**, and no real veteran data appears anywhere in the repository.
- **Classification quality depends on the configured model.** The zero-model path replays committed fixtures and therefore measures the plumbing, not a provider's accuracy.
- **OCR is out of scope.** An image-only PDF is refused by name rather than parsed.
- Laterality often requires a human. That is the design, not a gap.
- Only a single bilateral pair is selected per case. Multi-pair and four-extremity procedures under 4.26(b) are not implemented.
- Recheck makes no legal determination about a decision's correctness.
- Live provider classification quality is unmeasured. The seam is built, configurable and tested against provider-independent doubles, but no live inference has been run, so no accuracy claim is made.

---

## Repository layout

```text
src/recheck/
  cfr/combine.py             38 CFR 4.25 Table I arithmetic, 4.26 factor
  cfr/rating.py              rule ordering, 4.26(c)/(d), derivation trace
  extract/deterministic.py   parser and anatomy lexicon
  classify.py                the ownership boundary: lexicon, model, human
  schema.py                  strict schemas for anything the model produces
  provenance.py              actors, entries, trace rendering
  graph.py                   Strands graph, nodes, interrupt, safety gate
  case.py                    durable domain state, separate from graph state
  report.py                  evidence-backed report and verdict language
  cli.py                     audit / resume / show
  models/scripted.py         zero-model Model implementation
  models/factory.py          live provider adapter, config, preflight, timeout

tools/
  demo.py                    the three golden cases
  fetch_cfr.py               re-extract and diff the Table I fixture
  gate_report.py             the AI justification measurement
  make_letters.py            regenerate the synthetic letters

fixtures/
  letters/                   8 synthetic decision letters
  classifications/           committed classifier responses for the demo
  cfr425_table1_points.json  684 published Table I cells
  va_condition_names.json    138 externally-labelled condition names

spikes/                      verified API probes kept as references
docs/architecture.svg        architecture diagram
```

---

## Regulatory sources

- **38 CFR 4.25** — Combined Ratings Table
- **38 CFR 4.26** — Bilateral factor

Fetched via the official **eCFR API** rather than scraped; `www.ecfr.gov` serves a CAPTCHA to automated clients, and the documented API is the sanctioned route. `tools/fetch_cfr.py` reproduces the committed fixture so a reviewer can diff against the live regulation instead of trusting a blob.

---

## Security

Documents are untrusted input. Document text cannot override agent instructions; the extraction node emits typed records rather than passing raw page content downstream, so fetched prose never re-enters in system-prompt position. Model output is schema-validated with unexpected fields rejected. A side unsupported by the source text is refused regardless of stated confidence. Invalid human input cannot reach the calculation stage. Case identifiers are checked so they cannot escape the store directory. No credentials are stored in the repository, and the zero-model path performs no external transmission.

`src/` contains no `eval`, `exec`, `pickle`, `os.system` or `shell=True`. Five direct dependencies, pinned.

### Findings from the pre-release review

The review attacked the running product rather than reading the code, and found four defects. **None produced a wrong rating** — the fail-closed gate and the answer validation held throughout — but two let a run that had computed nothing report success, which for an audit tool is its own kind of wrong: the operator believes the letter was checked.

| | Finding | Fix |
|---|---|---|
| D1 | A corrupted `graph_state.json` deserialised into a state with no pending work. The graph reported `COMPLETED`, executed no nodes, and the CLI **exited 0**. | Success is now defined by the *product* outcome — the case must be complete and carry a recomputed degree — rather than by the framework's status. |
| D2 | `read_document` had no size limit. A 40 MB file read in 0.09s; a multi-gigabyte one exhausts memory, and PDF decompression bombs are a known vector. | Byte cap (8 MB) and page cap (100). |
| D3 | Strands logs node failures at ERROR, so a *deliberate* refusal printed "node failed / graph execution failed" beneath Recheck's own explanation — correct behaviour looking like a crash. | Framework logging suppressed; `--debug` restores it. |
| D4 | The persisted graph state was passed to the framework unvalidated. A JSON array produced an unhandled `AttributeError` from inside `deserialize_state`. Found by parametrising D1's test over several corrupt payloads instead of one. | Shape validated; any deserialisation failure is treated as a corrupt case. |

**What held up:** hand-editing `case.json` to inject a fabricated laterality and mark it human-resolved does **not** produce arithmetic. That is pinned by a regression test.

Each finding has a regression test in [`tests/test_security_hardening.py`](tests/test_security_hardening.py).

---

## Engineering decisions

Each of these came from a defect found during development, and each is preserved as a regression test.

**Pairwise integer rounding.** The first engine carried unrounded decimals through the chain. Verification against the published table showed 4.25 rounds at every step. The bug had produced a demonstration case that appeared to show a 10-point discrepancy which does not exist.

**Fail-closed graph topology.** Returning `Status.FAILED` from the assess node did not stop downstream nodes, so a *rejected* human answer still reached `compute` and produced a rating. The gate is now a conditional edge that reads committed state and blocks even when the case file cannot be read.

**Bounded model calls.** Strands retries a structured output that fails validation. Against a persistently invalid response this recursed until `RecursionError` — a hang, not an error. Every call is now bounded.

**Strict schemas.** Extra fields in model output were accepted and silently discarded, a smuggling path from document text into our objects.

**Abstention as a distinct state.** "Recognised as a non-extremity condition" and "the lexicon has no opinion" were the same value. That inflated the justification metric by counting 46 abstentions as correct answers, and spent model calls on conditions the lexicon already knew.

**Provenance over presentation.** An early parser produced condition names that had bled into the letterhead. The percentages were right, so tests passed — but the evidence shown to a reviewer was wrong. Wrong provenance is worse than a wrong parse because it is quieter.

---

## License

MIT. See [LICENSE](LICENSE).
