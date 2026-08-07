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
# Both agent (local SGLang) and user (third-party API) use OpenAI-compatible
# endpoints, differentiated via api_base in litellm args.
#
# Prerequisites:
#   1. bash src/eval/setup_tau2_bench.sh
#   2. bash src/eval/serve_envfactory.sh  (wait until server is ready)
#   3. Set environment variables for user simulator:
#        export USER_API_KEY="sk-xxx"           # your API key
#        export USER_BASE_URL="https://xxx/v1"  # (optional, defaults to https://yibuapi.com/v1)
#        export USER_MODEL="deepseek-v3.2"      # (optional, defaults to deepseek-v3.2)
#
# Usage:
#   bash src/eval/run_tau2_bench.sh [--domain retail] [--num-trials 1]
#   bash src/eval/run_tau2_bench.sh --all   # run all 3 domains

set -e

# === Agent model (local SGLang) ===
SGLANG_PORT=${SGLANG_PORT:-8000}
SGLANG_API_KEY=${SGLANG_API_KEY:-"placeholder"}
AGENT_MODEL=${AGENT_MODEL:-"LARK-Lab/EnvFactory-1.7B"}
AGENT_BASE_URL=${AGENT_BASE_URL:-"http://localhost:${SGLANG_PORT}/v1"}

# === User simulator (third-party OpenAI-compatible) ===
USER_MODEL=${USER_MODEL:-"deepseek-v3.2"}
USER_BASE_URL=${USER_BASE_URL:?"ERROR: Set USER_BASE_URL (e.g. https://your-provider.com/v1)"}
USER_API_KEY=${USER_API_KEY:?"ERROR: Set USER_API_KEY"}

# === Evaluation settings (aligned with paper) ===
DOMAIN=""
RUN_ALL=true
NUM_TRIALS=${NUM_TRIALS:-1}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-3}
TEMPERATURE=0.7

# === Parse args ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) DOMAIN="$2"; shift 2;;
        --num-trials) NUM_TRIALS="$2"; shift 2;;
        --port) SGLANG_PORT="$2"; AGENT_BASE_URL="http://localhost:${SGLANG_PORT}/v1"; shift 2;;
        --agent-model) AGENT_MODEL="$2"; shift 2;;
        --user-model) USER_MODEL="$2"; shift 2;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift 2;;
        --all) RUN_ALL=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ "$RUN_ALL" = false ] && [ -z "$DOMAIN" ]; then
    DOMAIN="retail"
fi

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TAU2_DIR="$EVAL_DIR/tau2-bench"
RESULTS_DIR="$EVAL_DIR/results"

mkdir -p "$RESULTS_DIR"

# === Validate ===
if [ ! -d "$TAU2_DIR" ]; then
    echo "ERROR: tau2-bench not found. Run: bash src/eval/setup_tau2_bench.sh"
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# === Build litellm args JSON ===
AGENT_LLM_ARGS="{\"api_base\": \"$AGENT_BASE_URL\", \"api_key\": \"$SGLANG_API_KEY\", \"temperature\": $TEMPERATURE}"
USER_LLM_ARGS="{\"api_base\": \"$USER_BASE_URL\", \"api_key\": \"$USER_API_KEY\"}"

echo "=============================================="
echo " tau2-bench Evaluation: EnvFactory-1.7B"
echo "=============================================="
echo " Domains:        ${DOMAINS[*]}"
echo " Num Trials:     $NUM_TRIALS"
echo " Agent Model:    openai/$AGENT_MODEL ($AGENT_BASE_URL)"
echo " User Model:     openai/$USER_MODEL ($USER_BASE_URL)"
echo " Temperature:    $TEMPERATURE"
echo " Concurrency:    $MAX_CONCURRENCY"
echo "=============================================="

cd "$TAU2_DIR"

for d in "${DOMAINS[@]}"; do
    echo ""
    echo ">>> Running domain: $d"

    SAVE_NAME="envfactory_1.7b_${d}_${TIMESTAMP}"

    uv run tau2 run \
        --domain "$d" \
        --agent-llm "openai/$AGENT_MODEL" \
        --user-llm "openai/$USER_MODEL" \
        --agent-llm-args "$AGENT_LLM_ARGS" \
        --user-llm-args "$USER_LLM_ARGS" \
        --num-trials "$NUM_TRIALS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --seed 300 \
        --save-to "$SAVE_NAME"

    echo ">>> Domain $d complete. Results saved to: data/simulations/$SAVE_NAME"
done

echo ""
echo "=== All evaluations complete ==="
echo "View results: cd $TAU2_DIR && uv run tau2 view"