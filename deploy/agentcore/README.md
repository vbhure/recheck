# Recheck on Amazon Bedrock AgentCore Runtime

This directory runs Recheck in your own AWS account. It does two things:

1. **One real Amazon Bedrock call from AWS CloudShell.** It uses Recheck's existing Strands `BedrockModel` provider (`recheck audit --model bedrock`).
2. **A deployment to Bedrock AgentCore Runtime.** Each invocation audits one decision letter through the same Strands Graph the CLI runs.

| File | What it is |
|---|---|
| `app.py` | The AgentCore entrypoint (`BedrockAgentCoreApp`, `@app.entrypoint`). It calls `recheck.cli._run_audit` for an audit and `recheck.cli.cmd_resume` for a resume. It uses the CLI's provider seam (`recheck.models.factory.build_agent_factory`) and `recheck.report.render`. None of that logic is re-implemented here. |
| `requirements.txt` | Pinned runtime dependencies: Recheck's pins plus `bedrock-agentcore`. |
| `cloudshell.sh` | One script to paste into AWS CloudShell: install, preflight, one live audit, configure, deploy, invoke. |

Tests: `tests/test_agentcore_app.py` calls the entrypoint directly. It uses no server, no network and no credentials.

## Run it

Open AWS CloudShell in **us-east-1** or **us-west-2**, then run:

```bash
curl -fsSLO https://raw.githubusercontent.com/<owner>/<repo>/<branch>/deploy/agentcore/cloudshell.sh
bash cloudshell.sh https://github.com/<owner>/<repo>.git <branch-or-tag>
```

Environment variables that change what the script does:

| Variable | Effect |
|---|---|
| `RECHECK_BEDROCK_MODEL_ID` | Model to call. Default `us.amazon.nova-lite-v1:0`. |
| `RECHECK_SKIP_DEPLOY=1` | Stop after the live smoke audit. |
| `RECHECK_CONTINUE_ON_QUESTION=1` | Deploy even if the smoke audit ends with a question. |
| `RECHECK_AGENT_NAME` | Runtime name. |
| `RECHECK_IDLE_TIMEOUT` | Session idle timeout, in seconds. |

The CloudShell identity must be allowed to create IAM roles, ECR repositories, CodeBuild projects and S3 objects, and AgentCore runtimes. An administrator role is simplest.

**Why Nova Lite is the default.** Amazon Nova models need no Anthropic first-use form or Marketplace subscription, so a new account can call it at once. The `us.` inference profile works from both US regions. Recheck's structured output is a Strands structured-output tool. Strands' `BedrockModel` sends it as the only tool in the Converse API's `toolConfig`, with `toolChoice` auto, and Nova supports Converse tool use. Strands would force the tool on a second turn if the model answered in prose, but Recheck caps the call at one turn (`limits={"turns": 1}`). So the model must call the tool on its first reply, or the names stay unknown. The schema Strands generates is flat: no `$ref`, only `type`, `properties`, `required`, `enum`, `items` and bounds. This was checked against strands-agents 1.55.1, `strands/models/bedrock.py` and `event_loop.py`.

**If Nova's answer does not validate,** Recheck fails closed. The two arm conditions stay unknown, the audit ends with a question (exit 2), and the script stops before it deploys. To retry with another model:

```bash
RECHECK_BEDROCK_MODEL_ID=us.amazon.nova-pro-v1:0 bash cloudshell.sh <repo> <ref>
```

You can also use a Claude inference profile, once Anthropic access is enabled in the account.

## Invoke the runtime

The payload is a JSON object with no other keys:

```json
{"letter_text": "...", "case_id": "optional", "answers": "optional, e.g. 1=left,2=right"}
```

**Response fields:**

| Field | Contents |
|---|---|
| `status` | `complete`, `awaiting_human`, `undetermined` or `unparsed`. It is `rejected`, `busy` or `error` for a refused request. |
| `stated_combined`, `recomputed_degree`, `possible_degrees` | The rating the letter states, the recomputed rating, and the ratings still possible. |
| `verdict`, `verdict_detail` | The verdict line and its explanation. |
| `question` | Present when the status is `awaiting_human`: the question text, `answer_format`, the items asked, and any rejected answer. |
| `ratings[]` | Each rating's extremity group and side, with `*_decided_by` set to `lexicon`, `letter`, `AI`, `AI (replayed fixture)` or `reviewer`. |
| `model` | `provider`, `model_id`, `classifier` and `model_call_made`. |
| `report` | Recheck's plain-text brief report. |

Refusals come back as HTTP 200 with `status: "rejected"` and an `error` message. That way `agentcore invoke` shows the reason instead of a generic runtime error.

**Human in the loop across two invocations.** Audit a letter on one runtime session. If the answer is `awaiting_human`, invoke again on the **same session id** with the same `case_id` and `answers`:

```bash
agentcore invoke --session-id "$S" "$(cat payload-05.json)"                  # -> awaiting_human, answer_format 1=<...>,2=<...>
agentcore invoke --session-id "$S" '{"case_id": "side", "answers": "1=left,2=right"}'   # -> complete, 80%
```

## Session lifetime: be honest about it

A case, meaning its `case.json` and its Strands session, lives under `/tmp` inside the AgentCore session's microVM. A resume works **only while that microVM is alive**. That lasts until the idle timeout, `RECHECK_IDLE_TIMEOUT` (900 s by default, up to 28,800), or the maximum lifetime of 8 hours, whichever comes first. It also ends if AgentCore replaces the microVM.

After that, a resume is refused with `no case ... in this runtime session`, and the letter has to be audited again. A different session id cannot see the case either: each session gets its own store.

Recheck's CLI keeps cases on disk for good. This deployment does not, because it has no durable store yet. Adding one, such as S3 or AgentCore Memory, is the next step. It is not built.

## What data goes where

| Data | Where it goes |
|---|---|
| **Letter text** | From whoever invokes the runtime (CloudShell, in the script) to the AgentCore Runtime **in your own AWS account**. It is written to that session's `/tmp` and never logged. The runtime logs only the case id, mode, status, the letter's byte count and whether a model call was made. |
| **What reaches Bedrock** | Only condition names **outside the deterministic lexicon**, in one structured call per letter. A name carrying a percentage, a label, a date or a long number is withheld. Percentages, the stated rating and the file number are never sent. For `07_clinical_terms.txt`, the model sees exactly two names: `left cubital tunnel syndrome` and `right De Quervain's tenosynovitis`. |
| **Response** | The invoker gets back the condition names, the report and the question. |
| **OpenTelemetry** | Turned off (`--disable-otel`). Strands' spans would otherwise record the classifier prompt, which holds condition names, in CloudWatch or X-Ray. |
| **CodeBuild** | Receives the staged source only (`app.py`, `requirements.txt`, `recheck/`), uploaded to an S3 bucket in your account. No letters are included. |
| **Smoke audit (step 5)** | Writes its case to `/tmp/recheck-smoke` in CloudShell. The letter is synthetic. |

Access to the runtime is IAM (SigV4) by default. No OAuth authorizer is configured. Only principals in your account that hold `bedrock-agentcore:InvokeAgentRuntime` can invoke it.

## IAM for the model call

The execution role the toolkit auto-creates (starter toolkit 0.3.12, `execution_role_policy.json.j2`) allows the following:

- `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`
- on `arn:aws:bedrock:*::foundation-model/*`
- and on `arn:aws:bedrock:<region>:<account>:*`, which covers inference profiles in the deploy region

A cross-region inference profile such as `us.amazon.nova-lite-v1:0` needs both: the profile ARN in the calling region, and the foundation model in each destination region. **No extra IAM is needed for Nova.**

Anthropic models also need the account's one-time Anthropic use-case form. The first invocation may also need `aws-marketplace:ViewSubscriptions` and `aws-marketplace:Subscribe`, which that role does not have. Make one Claude call from the console or CloudShell first if you switch to a Claude model.

In the runtime, `app.py` sets `AWS_EC2_METADATA_DISABLED=false` unless it is already set. Recheck's preflight skips instance-metadata credentials unless that opt-in is present, and the runtime's credentials must be allowed to resolve.

## Cost notes

Everything runs in your account at list prices. None of it is free-tier by default.

| Item | Cost |
|---|---|
| **Bedrock** | One call per audited letter, and only when the lexicon abstains. The prompt is about 800 characters of system prompt plus the names; the answer is a short JSON object. The script makes 2 such calls with Nova Lite: a fraction of a cent. |
| **AgentCore Runtime** | Billed on consumption (CPU and memory) while sessions are active, including idle time up to the idle timeout. A short idle timeout keeps that small. |
| **CodeBuild** | One ARM build per deploy, a few minutes. |
| **ECR** | Storage for the image. |
| **S3** | Build artifacts. |
| **CloudWatch Logs** | Runtime logs. |

Check current prices on the AWS pricing pages for Bedrock, AgentCore, CodeBuild and ECR before running at volume.

## Cleanup

```bash
source /tmp/recheck-venv/bin/activate     # recreate with cloudshell.sh steps 2-3 if CloudShell recycled /tmp
cd ~/recheck-aws/agentcore-build
agentcore destroy --agent recheck --force --delete-ecr-repo
```

`destroy` removes the runtime and its endpoint, the ECR images and repository, the CodeBuild project, the S3 build artifacts, and the auto-created execution and CodeBuild IAM roles.

It leaves two things behind, for you to delete by hand if you want:

- the CloudWatch log group `/aws/bedrock-agentcore/runtimes/recheck-*`
- the (emptied) CodeBuild source bucket `bedrock-agentcore-codebuild-sources-<account>-<region>`

## Local verification (no AWS)

```bash
py -3.12 -m venv .venv && .venv/Scripts/pip install -e ".[dev]" bedrock-agentcore bedrock-agentcore-starter-toolkit
.venv/Scripts/python -m pytest tests/test_agentcore_app.py
# serve it: the zero-model path, POST /invocations on :8080
RECHECK_AGENTCORE_SCRIPTED=fixtures/classifications/07_clinical_terms.json RECHECK_AGENTCORE_STORE=/tmp/rc \
  .venv/Scripts/python deploy/agentcore/app.py
curl -s localhost:8080/invocations -H 'Content-Type: application/json' \
  -H 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: local-0000000000000000000000000000001' \
  -d '{"case_id":"side","answers":"1=left,2=right"}'
```

## Known limits

- **Maintenance status.** The starter toolkit prints that it is no longer maintained in favour of the npm `@aws/agentcore` CLI. It still deploys. The script pins toolkit 0.3.12, and the packaging here was checked against that version's Dockerfile template.
- **Payload size.** The 256 KiB letter cap is checked after the runtime's HTTP server has parsed the body. It bounds Recheck's work, not the bytes AgentCore accepts.
- **Nova Lite quality.** Nova Lite's labelling quality for anatomy terms has not been measured here. Recheck's confidence floor and strict schema make a bad answer cost a question, not a wrong rating.
