#!/bin/bash
# Run tau2-bench evaluation for EnvFactory-1.7B
#
# Aligned with EnvFactory paper (arXiv 2605.18703):
#   - SGLang serving with reasoning-parser qwen3
#   - Temperature 0.7 (thinking model)
#   - User/Evaluator: DeepSeek-V3.2-Chat
#   - Domains: retail, airline, telecom
#   - seed 300 (tau2-bench default)
#   - max-concurrency 3 (tau2-bench default)
#   - num-trials 1 (tau2-bench default; increase for more stable results)
#
# Three LLM roles, all OpenAI-compatible, differentiated via api_base:
#   - Agent:     local SGLang (EnvFactory-1.7B)
#   - User:      third-party API (DeepSeek-V3.2-Chat user simulator)
#   - Evaluator: third-party API (DeepSeek-V3.2-Chat NL-assertions judge).
#                tau2 has NO --evaluator-llm flag and hard-codes the judge to
#                gpt-4.1 hitting api.openai.com (which 401s once OPENAI_API_KEY
#                is a third-party key). tau2_eval_patch.py redirects the judge to
#                your endpoint via TAU2_EVAL_* env vars -- see that file's header.
#
# Prerequisites:
#   1. bash src/eval/setup_tau2_bench.sh
#   2. bash src/eval/serve_envfactory.sh  (wait until server is ready) -- launch
#      it with the SAME MODEL you pass here: this script never loads weights, so
#      MODEL is only the label sent to the server (sglang ignores the request's
#      model field) plus the bookkeeping key for results/logs. What actually gets
#      evaluated is whatever the server loaded.
#   3. Set environment variables for the user simulator (and, by default, the
#      evaluator -- the paper uses DeepSeek-V3.2-Chat for BOTH):
#        export USER_API_KEY="sk-xxx"
#        export USER_BASE_URL="https://yibuapi.com/v1"
#        export USER_MODEL="deepseek-v3.2"      # (optional, defaults to deepseek-v3.2)
#      The NL judge defaults to these same values. To grade with a different
#      model/endpoint, set EVAL_MODEL / EVAL_BASE_URL / EVAL_API_KEY (below) or
#      pass --eval-model / --eval-base-url / --eval-api-key.
#
# Usage (all knobs are env vars; the script self-logs under logs/<experiment>/,
# so a launch command only needs USER_* , MODEL, and the port -- no LOG= /
# output-path bookkeeping at the call site, which is how stale pre-rename paths
# crept in):
#   USER_BASE_URL=... USER_API_KEY=... \
#     MODEL=$REPO/models/spbench/qwen3-1.7b-sft-ep1 SGLANG_PORT=8002 \
#     nohup bash src/eval/run_tau2_bench.sh >/dev/null 2>&1 &
#   bash src/eval/run_tau2_bench.sh --domain retail    # single domain
#   EXPERIMENT=spbench_v1 MODEL=$REPO/models/spbench/...   # override grouping

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
TAU2_DIR="$EVAL_DIR/tau2-bench"

# === Agent model (local SGLang) ===
# MODEL = local weights dir, used ONLY for bookkeeping (experiment/ckpt name
#         inference + the agent label). AGENT_MODEL = the name sent in requests;
#         defaults to the checkpoint basename.
SGLANG_PORT=${SGLANG_PORT:-8000}
SGLANG_API_KEY=${SGLANG_API_KEY:-"placeholder"}
MODEL=${MODEL:-"$REPO_ROOT/models/envfactory_baseline/EnvFactory-1.7B"}
AGENT_MODEL=${AGENT_MODEL:-"$(basename "$MODEL")"}
AGENT_BASE_URL=${AGENT_BASE_URL:-"http://localhost:${SGLANG_PORT}/v1"}

# === User simulator (third-party OpenAI-compatible) ===
# NOTE: OPENAI_API_KEY is auto-exported below (defaults to $USER_API_KEY) because
# litellm may drop the api_key passed via --user-llm-args under concurrency and
# fall back to that env var (~35% AuthenticationError rate without it).
USER_MODEL=${USER_MODEL:-"deepseek-v3.2"}
USER_BASE_URL=${USER_BASE_URL:?"ERROR: Set USER_BASE_URL (e.g. https://your-provider.com/v1)"}
USER_API_KEY=${USER_API_KEY:?"ERROR: Set USER_API_KEY"}

# === Evaluator / NL-assertions judge (third-party OpenAI-compatible) ===
# Defaults to the user-simulator endpoint (paper uses DeepSeek-V3.2-Chat for
# both). May be overridden by --eval-* flags or EVAL_* env vars. Exports happen
# after arg parsing below.
EVAL_MODEL=${EVAL_MODEL:-"$USER_MODEL"}
EVAL_BASE_URL=${EVAL_BASE_URL:-"$USER_BASE_URL"}
EVAL_API_KEY=${EVAL_API_KEY:-"$USER_API_KEY"}

# === Evaluation settings (aligned with paper) ===
DOMAIN=""
RUN_ALL=""
NUM_TRIALS=${NUM_TRIALS:-1}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-3}
TEMPERATURE=${TEMPERATURE:-0.7}

# === Parse args ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) DOMAIN="$2"; RUN_ALL=false; shift 2;;
        --num-trials) NUM_TRIALS="$2"; shift 2;;
        --port) SGLANG_PORT="$2"; AGENT_BASE_URL="http://localhost:${SGLANG_PORT}/v1"; shift 2;;
        --model) MODEL="$2"; AGENT_MODEL="$(basename "$MODEL")"; shift 2;;
        --agent-model) AGENT_MODEL="$2"; shift 2;;
        --user-model) USER_MODEL="$2"; shift 2;;
        --eval-model) EVAL_MODEL="$2"; shift 2;;
        --eval-base-url) EVAL_BASE_URL="$2"; shift 2;;
        --eval-api-key) EVAL_API_KEY="$2"; shift 2;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift 2;;
        --all) RUN_ALL=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Default: run all three domains (retail, airline, telecom) as per paper
if [ -z "$RUN_ALL" ]; then
    RUN_ALL=true
fi

# Export endpoint config for tau2_eval_patch.py (redirects the NL judge) and the
# litellm env fallbacks. Under concurrency litellm has been observed to DROP the
# api_base/api_key passed via *-llm-args; when that happens it falls back to
# OPENAI_API_BASE / OPENAI_API_KEY. Without these env fallbacks, dropped calls
# go to api.openai.com with the yibuapi key -> "Incorrect API key provided" ->
# 0-message infrastructure_error -> task retries -> hour-long tasks.
# OPENAI_API_BASE is safe for the local agent: its explicit api_base=localhost
# takes precedence (litellm: explicit > env), confirmed by the agent working
# while OPENAI_API_BASE points at yibuapi.
export TAU2_EVAL_LLM="$EVAL_MODEL"
export TAU2_EVAL_API_BASE="$EVAL_BASE_URL"
export TAU2_EVAL_API_KEY="$EVAL_API_KEY"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$EVAL_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$USER_BASE_URL}"

# Redirect tau2's output into our results tree instead of burying it inside the
# vendored tau2-bench checkout (where it'd be lost on re-clone). tau2 uses one
# DATA_DIR for both input (tau2/domains/...) and output (simulations/), so we
# point TAU2_DATA_DIR at a per-checkpoint dir and symlink the input data in
# (relatively, so the tree survives repo relocation -- absolute links died in
# the envfactory_repro -> LIMA move). Unlike BFCL, no per-invocation timestamp
# subdir is needed: tau2 never reuses prior outputs (every run writes a fresh
# timestamped simulations/<domain>_<RUN_TAG>/ dir), so simulations accumulate
# per checkpoint and old runs stay comparable.
#
# Experiment grouping keeps the layout contract (one name threads through
# models/data/results/logs): results land in results/<experiment>/tau2/... and
# logs in logs/<experiment>/. Default experiment = the model's group dir
# (models/<experiment>/<ckpt>); override with EXPERIMENT=... when the checkpoint
# is filed under a coarser dir than its data (e.g. models/spbench/ trained on
# data/sft/spbench_v1/ -> pass EXPERIMENT=spbench_v1).
CKPT_NAME=$(basename "$MODEL")                       # e.g. EnvFactory-1.7B
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT=${EXPERIMENT:-$(basename "$(dirname "$MODEL")")}
TAU2_RUN_ROOT=${TAU2_RUN_ROOT:-"$REPO_ROOT/results/$EXPERIMENT/tau2/$CKPT_NAME"}
export TAU2_DATA_DIR="$TAU2_RUN_ROOT"
mkdir -p "$TAU2_DATA_DIR"
ln -sfn "$(realpath --relative-to="$TAU2_DATA_DIR" "$TAU2_DIR/data/tau2")" \
    "$TAU2_DATA_DIR/tau2"   # input data via (relative) symlink

# Self-log under logs/<experiment>/ with the SAME RUN_TAG as the simulation
# save names, so the log file and its simulations/ dirs always pair up. stdout
# still passes through (tee), so interactive runs and nohup >/dev/null both
# behave. Override with TAU2_LOG=.
LOG_DIR="$REPO_ROOT/logs/$EXPERIMENT"
mkdir -p "$LOG_DIR"
TAU2_LOG=${TAU2_LOG:-"$LOG_DIR/tau2_${CKPT_NAME}_${RUN_TAG}.log"}
exec > >(tee -a "$TAU2_LOG") 2>&1

# === Validate ===
if [ ! -d "$TAU2_DIR" ]; then
    echo "ERROR: tau2-bench not found. Run: bash src/eval/setup_tau2_bench.sh"
    exit 1
fi

if [ ! -d "$MODEL" ]; then
    echo "ERROR: weights dir not found: $MODEL"
    echo "Pass MODEL=/path/to/checkpoint (and serve it with the same path via"
    echo "serve_envfactory.sh)."
    exit 1
fi

if ! curl -s "http://localhost:$SGLANG_PORT/v1/models" > /dev/null 2>&1; then
    echo "ERROR: SGLang server not responding on port $SGLANG_PORT."
    echo "Start it with: bash src/eval/serve_envfactory.sh"
    exit 1
fi

# === Determine domains ===
if [ "$RUN_ALL" = true ]; then
    DOMAINS=("retail" "airline" "telecom")
else
    DOMAINS=("$DOMAIN")
fi

# === Build litellm args JSON ===
AGENT_LLM_ARGS="{\"api_base\": \"$AGENT_BASE_URL\", \"api_key\": \"$SGLANG_API_KEY\", \"temperature\": $TEMPERATURE}"
USER_LLM_ARGS="{\"api_base\": \"$USER_BASE_URL\", \"api_key\": \"$USER_API_KEY\"}"

echo "=============================================="
echo " tau2-bench Evaluation"
echo "=============================================="
echo " Weights:        $MODEL"
echo " Experiment:     $EXPERIMENT"
echo " Domains:        ${DOMAINS[*]}"
echo " Num Trials:     $NUM_TRIALS"
echo " Agent Model:    openai/$AGENT_MODEL ($AGENT_BASE_URL)"
echo " User Model:     openai/$USER_MODEL ($USER_BASE_URL)"
echo " Evaluator:      openai/$EVAL_MODEL ($EVAL_BASE_URL)"
echo " Temperature:    $TEMPERATURE"
echo " Concurrency:    $MAX_CONCURRENCY"
echo " Run tag:        $RUN_TAG"
echo " Results root:   $TAU2_RUN_ROOT"
echo " Log:            $TAU2_LOG"
echo "=============================================="

cd "$TAU2_DIR"

for d in "${DOMAINS[@]}"; do
    echo ""
    echo ">>> Running domain: $d"

    SAVE_NAME="${d}_${RUN_TAG}"

    # Run via tau2_eval_patch.py (not bare `tau2`) so the NL-assertions judge is
    # redirected to $EVAL_BASE_URL; it forwards all args to tau2's CLI main().
    uv run python "$EVAL_DIR/tau2_eval_patch.py" run \
        --domain "$d" \
        --agent-llm "openai/$AGENT_MODEL" \
        --user-llm "openai/$USER_MODEL" \
        --agent-llm-args "$AGENT_LLM_ARGS" \
        --user-llm-args "$USER_LLM_ARGS" \
        --num-trials "$NUM_TRIALS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --seed 300 \
        --save-to "$SAVE_NAME"

    echo ">>> Domain $d complete. Results saved to: $TAU2_DATA_DIR/simulations/$SAVE_NAME"
done

echo ""
echo "=== All evaluations complete ==="
echo "Results root: $TAU2_RUN_ROOT"
echo "View results: TAU2_DATA_DIR=$TAU2_DATA_DIR uv run --project $TAU2_DIR tau2 view"