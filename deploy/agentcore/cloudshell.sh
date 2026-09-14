#!/usr/bin/env bash
# Recheck on AWS, from AWS CloudShell: one live Bedrock call, then AgentCore Runtime.
#
#   bash cloudshell.sh <public GitHub repo URL> [branch-or-tag]
#
# What it does, stopping at the first failure:
#   1  checks the AWS identity and region CloudShell gives you
#   2  installs uv (user-local) and a Python 3.12 virtualenv under /tmp
#   3  clones the repo and installs Recheck + the AgentCore starter toolkit
#   4  `recheck preflight --model bedrock`   (no inference call)
#   5  ONE live audit of fixtures/letters/07_clinical_terms.txt with --model bedrock
#      (one Bedrock call: two condition names the lexicon abstains on)
#   6  stages app.py + requirements.txt + src/recheck into a build directory
#   7  `agentcore configure` non-interactively: container build in CodeBuild (no
#      local Docker), auto-created execution role and ECR repository, memory off,
#      OpenTelemetry off
#   8  `agentcore deploy` with RECHECK_BEDROCK_MODEL_ID set on the runtime
#   9  `agentcore invoke` once with the synthetic letter, then prints the ARN
#      and the cleanup commands
#
# Overridable with environment variables:
#   RECHECK_BEDROCK_MODEL_ID   default us.amazon.nova-lite-v1:0 (no Anthropic use-case form needed);
#                              e.g. us.anthropic.claude-haiku-4-5-20251001-v1:0 once Anthropic access is set up
#   AWS_REGION                 default: CloudShell's region (use us-east-1 or us-west-2)
#   RECHECK_AGENT_NAME         default recheck
#   RECHECK_IDLE_TIMEOUT       runtime session idle timeout in seconds, default 900
#   RECHECK_SKIP_DEPLOY=1      stop after the live smoke audit (steps 1-5 only)
#   RECHECK_CONTINUE_ON_QUESTION=1  deploy even if the smoke audit ended with a question
#
# Costs: one small Bedrock call in step 5, one in step 9; a CodeBuild build; ECR
# storage; AgentCore Runtime consumption while sessions run. See README.md.

set -Eeuo pipefail

REPO_URL="${1:-}"
REF="${2:-}"
if [[ -z "$REPO_URL" ]]; then
  echo "usage: bash cloudshell.sh <public GitHub repo URL> [branch-or-tag]" >&2
  exit 2
fi

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
MODEL_ID="${RECHECK_BEDROCK_MODEL_ID:-us.amazon.nova-lite-v1:0}"
AGENT_NAME="${RECHECK_AGENT_NAME:-recheck}"
IDLE_TIMEOUT="${RECHECK_IDLE_TIMEOUT:-900}"
WORK="${RECHECK_WORK:-$HOME/recheck-aws}"   # small: source + build dir (keeps .bedrock_agentcore.yaml for destroy)
VENV="${RECHECK_VENV:-/tmp/recheck-venv}"   # large: CloudShell's home is 1 GB, /tmp is not persistent
SRC="$WORK/src"
BUILD="$WORK/agentcore-build"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export AGENTCORE_SUPPRESS_RECOMMENDATION=1
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
export PATH="$HOME/.local/bin:$PATH"

STEP="start"
step() { STEP="$1"; printf '\n\033[1m==== [%s] %s\033[0m\n' "$1" "$2"; }
on_error() { echo; echo "!!!! FAILED in step [$STEP] (line $1). Nothing after it ran." >&2; }
trap 'on_error $LINENO' ERR

case "$MODEL_ID" in
  us.*) [[ "$REGION" == us-* ]] || { echo "model $MODEL_ID is a US inference profile; set AWS_REGION to a US region (us-east-1 / us-west-2)" >&2; exit 2; } ;;
  eu.*) [[ "$REGION" == eu-* ]] || { echo "model $MODEL_ID is an EU inference profile; set AWS_REGION to an EU region" >&2; exit 2; } ;;
esac

# ---------------------------------------------------------------------------
step 1 "AWS identity and region"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
echo "account: $ACCOUNT   region: $REGION   model: $MODEL_ID   agent: $AGENT_NAME"

# ---------------------------------------------------------------------------
step 2 "uv and a Python 3.12 virtualenv ($VENV)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

# ---------------------------------------------------------------------------
step 3 "clone $REPO_URL ${REF:+($REF) }and install"
mkdir -p "$WORK"
rm -rf "$SRC"
git clone --quiet --depth 1 ${REF:+--branch "$REF"} "$REPO_URL" "$SRC"
git -C "$SRC" log --oneline -1
uv pip install --quiet "$SRC" -r "$SRC/deploy/agentcore/requirements.txt" "bedrock-agentcore-starter-toolkit==0.3.12"
uv pip show strands-agents bedrock-agentcore bedrock-agentcore-starter-toolkit 2>/dev/null | grep -E '^(Name|Version)'

# ---------------------------------------------------------------------------
step 4 "recheck preflight --model bedrock (no inference call)"
cd "$SRC"
RECHECK_MODEL_ID="$MODEL_ID" RECHECK_REGION="$REGION" recheck preflight --model bedrock

# ---------------------------------------------------------------------------
step 5 "LIVE smoke audit: fixtures/letters/07_clinical_terms.txt --model bedrock (one Bedrock call)"
trap - ERR; set +e
RECHECK_MODEL_ID="$MODEL_ID" RECHECK_REGION="$REGION" \
  recheck --store /tmp/recheck-smoke audit fixtures/letters/07_clinical_terms.txt --case smoke --model bedrock --fresh \
  | tee /tmp/recheck-smoke.txt
SMOKE_RC=${PIPESTATUS[0]}
set -e; trap 'on_error $LINENO' ERR
echo "recheck exit code: $SMOKE_RC   (0 = finished, 2 = waiting on a reviewer's answer, 3 = no result)"
if [[ "$SMOKE_RC" == 3 ]]; then
  echo "The smoke audit produced no result; see the output above." >&2
  false
fi
if [[ "$SMOKE_RC" == 2 ]]; then
  echo "The live call did not settle the letter: the model's answer was missing, invalid, below the"
  echo "confidence floor, or the call failed (the decision trace above says which). Recheck failed closed"
  echo "and asked a question instead of guessing. Expected for this letter: both arm conditions [AI] upper, 80%."
  if grep -q "AccessDenied\|ValidationException\|ResourceNotFound" /tmp/recheck-smoke.txt; then
    echo "The trace names an access or model-id error: check Bedrock model access for $MODEL_ID in $REGION."
  fi
  if [[ "${RECHECK_CONTINUE_ON_QUESTION:-0}" != 1 ]]; then
    echo "Stopping before deploy. Retry with another model, e.g." >&2
    echo "  RECHECK_BEDROCK_MODEL_ID=us.amazon.nova-pro-v1:0 bash cloudshell.sh $REPO_URL $REF" >&2
    echo "or deploy anyway with RECHECK_CONTINUE_ON_QUESTION=1." >&2
    trap - ERR
    exit 1
  fi
fi
if [[ "${RECHECK_SKIP_DEPLOY:-0}" == 1 ]]; then
  echo "RECHECK_SKIP_DEPLOY=1: stopping after the live smoke audit."
  exit 0
fi

# ---------------------------------------------------------------------------
step 6 "stage the runtime package in $BUILD"
mkdir -p "$BUILD"
rm -rf "$BUILD/recheck" "$BUILD/app.py" "$BUILD/requirements.txt"
cp "$SRC/deploy/agentcore/app.py" "$SRC/deploy/agentcore/requirements.txt" "$BUILD/"
cp -r "$SRC/src/recheck" "$BUILD/recheck"
find "$BUILD/recheck" -name __pycache__ -type d -prune -exec rm -rf {} +
ls -la "$BUILD"
cd "$BUILD"

# ---------------------------------------------------------------------------
step 7 "agentcore configure (container via CodeBuild, auto-create role + ECR, memory off, OTEL off)"
agentcore configure \
  --entrypoint app.py \
  --name "$AGENT_NAME" \
  --requirements-file requirements.txt \
  --region "$REGION" \
  --deployment-type container \
  --disable-memory \
  --disable-otel \
  --idle-timeout "$IDLE_TIMEOUT" \
  --non-interactive
echo "--- generated Dockerfile:"
cat ".bedrock_agentcore/$AGENT_NAME/Dockerfile" 2>/dev/null || find . -name Dockerfile -path "*bedrock_agentcore*" -exec cat {} \;

# ---------------------------------------------------------------------------
step 8 "agentcore deploy (CodeBuild ARM64 build, then AgentCore Runtime)"
agentcore deploy --agent "$AGENT_NAME" --auto-update-on-conflict \
  --env "RECHECK_BEDROCK_MODEL_ID=$MODEL_ID"

AGENT_ARN="$(python -c '
import sys, yaml
cfg = yaml.safe_load(open(".bedrock_agentcore.yaml"))
print(cfg["agents"][sys.argv[1]]["bedrock_agentcore"]["agent_arn"] or "")
' "$AGENT_NAME")"
if [[ -z "$AGENT_ARN" ]]; then
  echo "deploy finished without an agent ARN in .bedrock_agentcore.yaml" >&2
  false
fi
echo "runtime ARN: $AGENT_ARN"

# ---------------------------------------------------------------------------
step 9 "agentcore invoke: one synthetic letter"
SESSION_ID="recheck-cloudshell-$(python -c 'import uuid; print(uuid.uuid4().hex)')"
PAYLOAD="$(python -c '
import json, sys
print(json.dumps({"letter_text": open(sys.argv[1], encoding="utf-8").read(), "case_id": "cloud-smoke"}))
' "$SRC/fixtures/letters/07_clinical_terms.txt")"
# The first invocation of a new runtime can take a while (cold start).
agentcore invoke --agent "$AGENT_NAME" --session-id "$SESSION_ID" "$PAYLOAD"

python -c '
import json, sys
json.dump({"letter_text": open(sys.argv[1], encoding="utf-8").read(), "case_id": "side"}, open(sys.argv[2], "w"))
' "$SRC/fixtures/letters/05_missing_side.txt" "$WORK/payload-05.json"

cat <<EOF

==========================================================================
DEPLOYED
  runtime ARN : $AGENT_ARN
  region      : $REGION
  model       : $MODEL_ID
  session used: $SESSION_ID
==========================================================================

Human-in-the-loop across two invocations (same session id; works while that
session's microVM is alive - idle timeout ${IDLE_TIMEOUT}s):

  source $VENV/bin/activate && cd $BUILD
  agentcore invoke --agent $AGENT_NAME --session-id $SESSION_ID "\$(cat $WORK/payload-05.json)"
  agentcore invoke --agent $AGENT_NAME --session-id $SESSION_ID '{"case_id": "side", "answers": "1=left,2=right"}'

Status:
  agentcore status --agent $AGENT_NAME

CLEANUP (removes the runtime, ECR images and repository, CodeBuild project,
S3 build artifacts and the auto-created IAM roles):

  source $VENV/bin/activate 2>/dev/null || { uv venv --python 3.12 $VENV && source $VENV/bin/activate && uv pip install bedrock-agentcore-starter-toolkit==0.3.12; }
  cd $BUILD && agentcore destroy --agent $AGENT_NAME --force --delete-ecr-repo

  Left behind by destroy (delete by hand if wanted): the CloudWatch log group
  /aws/bedrock-agentcore/runtimes/${AGENT_NAME}-*, and the CodeBuild source
  bucket bedrock-agentcore-codebuild-sources-$ACCOUNT-$REGION (emptied).
EOF
