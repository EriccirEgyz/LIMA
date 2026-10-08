#!/bin/bash
# Run VitaBench evaluation for EnvFactory
#
# VitaBench evaluates LLM agents on interactive tasks across three domains:
#   - delivery: food delivery scenarios
#   - instore: in-store consumption scenarios
#   - ota: online travel agency scenarios
#
# The benchmark includes:
#   - 100 cross-scenario tasks (testing generalization across domains)
#   - 300 single-scenario tasks (domain-specific evaluation)
#   - 66 tools total across all domains
#
# Prerequisites:
#   1. bash src/eval/setup_vitabench.sh
#   2. Start the agent model server (e.g. bash src/eval/serve_31_h100.sh)
#   3. export USER_BASE_URL=... USER_API_KEY=...  (user simulator / evaluator)
#   Model endpoints come from vitabench_models.yaml, filled from env vars below.
#
# Usage:
#   # Default: cross-scenario + delivery + instore + ota, one after another
#   bash src/eval/run_vitabench.sh
#
#   # Custom checkpoint (same path you served) and port
#   MODEL=$REPO/models/spbench/qwen3-1.7b-sft-ep1 SGLANG_PORT=8002 \
#     bash src/eval/run_vitabench.sh
#   # -> results/spbench/vitabench/qwen3-1.7b-sft-ep1/<run_tag>/
#   #      simulations_{cross_domain,delivery,instore,ota}.json + results_*.csv
#   #    logs/spbench/vitabench_qwen3-1.7b-sft-ep1_<run_tag>.log
#
#   # A single task set only
#   DOMAIN=ota bash src/eval/run_vitabench.sh
#   DOMAIN=delivery,instore,ota bash src/eval/run_vitabench.sh   # cross-scenario only
#
#   # With thinking enabled and custom concurrency
#   ENABLE_THINK=true MAX_CONCURRENCY=4 bash src/eval/run_vitabench.sh
#
#   # Run specific tasks
#   TASK_IDS=1,2,3 bash src/eval/run_vitabench.sh
#
#   # Background execution
#   nohup bash src/eval/run_vitabench.sh > logs/vitabench_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
VITABENCH_DIR="$EVAL_DIR/vitabench"

# Redirect caches out of the uid-1001 $HOME (permissions). See cache_env.sh.
if [ -f "$EVAL_DIR/cache_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$EVAL_DIR/cache_env.sh"
fi

# === Configuration ===
# Default: all four task sets reported in the paper, run one after another:
#   delivery,instore,ota (100 cross-scenario tasks), delivery, instore, ota
#   (100 each). Set DOMAIN to run just one, e.g. DOMAIN=ota or
#   DOMAIN=delivery,instore,ota (cross-scenario only).
if [ -n "$DOMAIN" ]; then
    DOMAINS=("$DOMAIN")
else
    DOMAINS=("delivery,instore,ota" "delivery" "instore" "ota")
fi

# Agent under test (local SGLang server, see serve_31_h100.sh).
# MODEL = the weights path being served (same as run_bfcl.sh / run_tau2_bench.sh);
# it only drives naming here. AGENT_MODEL = the name sent in requests; SGLang
# accepts any name for its single served model.
SGLANG_PORT=${SGLANG_PORT:-8000}
MODEL=${MODEL:-"$REPO_ROOT/models/envfactory_baseline/EnvFactory-1.7B"}
AGENT_MODEL=${AGENT_MODEL:-"$(basename "$MODEL")"}
AGENT_BASE_URL=${AGENT_BASE_URL:-"http://localhost:${SGLANG_PORT}/v1"}
AGENT_API_KEY=${AGENT_API_KEY:-"placeholder"}

# Thinking mode for the agent. EnvFactory: temperature 0 for non-thinking,
# 0.7 for thinking models.
ENABLE_THINK=${ENABLE_THINK:-true}
if [ "$ENABLE_THINK" = "true" ]; then
    AGENT_TEMPERATURE=${AGENT_TEMPERATURE:-0.7}
else
    AGENT_TEMPERATURE=${AGENT_TEMPERATURE:-0}
fi

# User simulator + evaluator (EnvFactory: DeepSeek-V3.2-Chat).
# Keep the key in the environment, not in the yaml:
#   export USER_BASE_URL="https://your-provider.com/v1"
#   export USER_API_KEY="sk-xxx"
USER_MODEL=${USER_MODEL:-"deepseek-v3.2"}
USER_BASE_URL=${USER_BASE_URL:?"ERROR: Set USER_BASE_URL (e.g. https://your-provider.com/v1)"}
USER_API_KEY=${USER_API_KEY:?"ERROR: Set USER_API_KEY"}
USER_TEMPERATURE=${USER_TEMPERATURE:-0}
EVALUATOR_MODEL=${EVALUATOR_MODEL:-"$USER_MODEL"}

# Evaluation parameters
NUM_TRIALS=${NUM_TRIALS:4}
NUM_TASKS=${NUM_TASKS:-}  # Empty = all tasks
TASK_IDS=${TASK_IDS:-}    # Empty = all tasks, or comma-separated IDs: 1,2,3
MAX_STEPS=${MAX_STEPS:-300}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
LANGUAGE=${LANGUAGE:-"chinese"}

# === Output layout (same contract as run_bfcl.sh / run_tau2_bench.sh) ===
# results/<experiment>/vitabench/<ckpt>/<run_tag>/ holds the simulation json
# and csv; logs/<experiment>/vitabench_<ckpt>_<run_tag>.log is the self-log.
# Default experiment = the model's group dir (models/<experiment>/<ckpt>).
CKPT_NAME=$(basename "$MODEL")
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT=${EXPERIMENT:-$(basename "$(dirname "$MODEL")")}
VITA_RUN_ROOT=${VITA_RUN_ROOT:-"$REPO_ROOT/results/$EXPERIMENT/vitabench/$CKPT_NAME/$RUN_TAG"}

LOG_DIR="$REPO_ROOT/logs/$EXPERIMENT"
mkdir -p "$VITA_RUN_ROOT" "$LOG_DIR"
VITA_LOG=${VITA_LOG:-"$LOG_DIR/vitabench_${CKPT_NAME}_${RUN_TAG}.log"}
exec > >(tee -a "$VITA_LOG") 2>&1

# Model config template; ${VAR} placeholders are filled from the variables above.
MODELS_TEMPLATE=${MODELS_TEMPLATE:-"$EVAL_DIR/vitabench_models.yaml"}

# === Validation ===
if [ ! -d "$VITABENCH_DIR" ]; then
    echo "ERROR: vitabench not found at $VITABENCH_DIR"
    echo "Run: bash src/eval/setup_vitabench.sh"
    exit 1
fi

if [ ! -f "$MODELS_TEMPLATE" ]; then
    echo "ERROR: Model config template not found at $MODELS_TEMPLATE"
    exit 1
fi

if ! curl -s "${AGENT_BASE_URL}/models" > /dev/null 2>&1; then
    echo "ERROR: Agent server not responding at $AGENT_BASE_URL"
    echo "Start it with: bash src/eval/serve_31_h100.sh"
    exit 1
fi

# === Render model config ===
# Written to a private temp file so API keys never land in the repo.
VITA_MODEL_CONFIG_PATH=$(mktemp --suffix=.yaml)
chmod 600 "$VITA_MODEL_CONFIG_PATH"
trap 'rm -f "$VITA_MODEL_CONFIG_PATH"' EXIT
export VITA_MODEL_CONFIG_PATH
export AGENT_MODEL AGENT_BASE_URL AGENT_API_KEY AGENT_TEMPERATURE ENABLE_THINK \
       USER_MODEL USER_BASE_URL USER_API_KEY USER_TEMPERATURE
python3 - "$MODELS_TEMPLATE" "$VITA_MODEL_CONFIG_PATH" <<'EOF'
import os, re, sys
text = open(sys.argv[1]).read()
def sub(m):
    name = m.group(1)
    if name not in os.environ:
        sys.exit(f"ERROR: {sys.argv[1]} references unset variable ${{{name}}}")
    return os.environ[name]
open(sys.argv[2], "w").write(re.sub(r"\$\{(\w+)\}", sub, text))
EOF

# === Build vita command ===
cd "$VITABENCH_DIR"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Options shared by every domain.
VITA_ARGS="--agent-llm $AGENT_MODEL"
VITA_ARGS="$VITA_ARGS --user-llm $USER_MODEL"
VITA_ARGS="$VITA_ARGS --evaluator-llm $EVALUATOR_MODEL"
VITA_ARGS="$VITA_ARGS --num-trials $NUM_TRIALS"
VITA_ARGS="$VITA_ARGS --max-steps $MAX_STEPS"
VITA_ARGS="$VITA_ARGS --max-concurrency $MAX_CONCURRENCY"
VITA_ARGS="$VITA_ARGS --language $LANGUAGE"

if [ "$ENABLE_THINK" = "true" ]; then
    VITA_ARGS="$VITA_ARGS --enable-think"
fi

if [ -n "$NUM_TASKS" ]; then
    VITA_ARGS="$VITA_ARGS --num-tasks $NUM_TASKS"
fi

if [ -n "$TASK_IDS" ]; then
    VITA_ARGS="$VITA_ARGS --task-ids $TASK_IDS"
fi

# Output file stem per domain: "delivery,instore,ota" -> "cross_domain".
domain_stem() {
    if [[ "$1" == *,* ]]; then echo "cross_domain"; else echo "$1"; fi
}

# === Run evaluation ===
echo "=== VitaBench Evaluation ==="
echo "Weights: $MODEL"
echo "Experiment: $EXPERIMENT"
echo "Run tag: $RUN_TAG"
echo "Domains: ${DOMAINS[*]}"
echo "Agent: $AGENT_MODEL ($AGENT_BASE_URL, temperature=$AGENT_TEMPERATURE)"
echo "User: $USER_MODEL ($USER_BASE_URL, temperature=$USER_TEMPERATURE)"
echo "Evaluator: $EVALUATOR_MODEL"
echo "Thinking: $ENABLE_THINK"
echo "Max steps: $MAX_STEPS"
echo "Concurrency: $MAX_CONCURRENCY"
echo "Language: $LANGUAGE"
echo "Results root: $VITA_RUN_ROOT"
echo "Log: $VITA_LOG"
echo "Model config template: $MODELS_TEMPLATE"

for d in "${DOMAINS[@]}"; do
    stem=$(domain_stem "$d")
    # vita saves to DATA_DIR/simulations/<save_to>; an absolute path overrides that.
    sim_output="$VITA_RUN_ROOT/simulations_${stem}.json"
    csv_output="$VITA_RUN_ROOT/results_${stem}.csv"
    cmd="vita run --domain $d $VITA_ARGS --save-to $sim_output --csv-output $csv_output"

    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Running domain: $d"
    echo "Command: $cmd"
    eval "$cmd"
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished domain: $d"
done

echo ""
echo "=== Results ==="
echo "Results root: $VITA_RUN_ROOT"
echo "Log:          $VITA_LOG"
echo ""
echo "View results with: vita view --file $VITA_RUN_ROOT/simulations_<domain>.json"
