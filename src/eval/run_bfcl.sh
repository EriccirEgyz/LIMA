#!/bin/bash
# Run BFCL v3 evaluation for EnvFactory-1.7B
#
# Aligned with EnvFactory paper (arXiv 2605.18703):
#   - BFCL v3 (bfcl2025), all categories (single-turn + multi-turn)
#   - SGLang serving (reuses serve_envfactory.sh)
#   - Temperature 0.7 for thinking models (Qwen3-based)
#
# Prerequisites:
#   1. bash src/eval/setup_bfcl.sh
#   2. bash src/eval/serve_envfactory.sh  (wait until server is ready)
#
# Usage:
#   bash src/eval/run_bfcl.sh                          # default: all (paper setting)
#   bash src/eval/run_bfcl.sh --category multi_turn    # only multi-turn

set -e

# === Defaults (aligned with paper: full BFCL v3) ===
MODEL=${MODEL:-"LARK-Lab/EnvFactory-1.7B"}
CATEGORY=${CATEGORY:-"all"}
SGLANG_PORT=${SGLANG_PORT:-8000}

# === Parse args ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --category) CATEGORY="$2"; shift 2;;
        --port) SGLANG_PORT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
GORILLA_DIR="$EVAL_DIR/gorilla"
BFCL_DIR="$GORILLA_DIR/berkeley-function-call-leaderboard"

# === Validate ===
if [ ! -d "$BFCL_DIR/.venv" ]; then
    echo "ERROR: BFCL not installed. Run: bash src/eval/setup_bfcl.sh"
    exit 1
fi

if ! curl -s "http://localhost:$SGLANG_PORT/v1/models" > /dev/null 2>&1; then
    echo "ERROR: SGLang server not responding on port $SGLANG_PORT."
    echo "Start it with: bash src/eval/serve_envfactory.sh"
    exit 1
fi

echo "=============================================="
echo " BFCL v3 Evaluation: EnvFactory-1.7B"
echo "=============================================="
echo " Model:          $MODEL"
echo " Categories:     $CATEGORY"
echo " SGLang Port:    $SGLANG_PORT"
echo "=============================================="

cd "$BFCL_DIR"

# Configure BFCL to use existing SGLang server
export LOCAL_SERVER_ENDPOINT="localhost"
export LOCAL_SERVER_PORT="$SGLANG_PORT"

# === Step 1: Generate responses ===
echo ""
echo ">>> Step 1: Generating model responses..."

uv run bfcl generate \
    --model "$MODEL" \
    --test-category $CATEGORY \
    --backend sglang \
    --skip-server-setup

# === Step 2: Evaluate ===
echo ""
echo ">>> Step 2: Evaluating responses..."

uv run bfcl evaluate \
    --model "$MODEL" \
    --test-category $CATEGORY

echo ""
echo "=== BFCL evaluation complete ==="
echo "Results in: $BFCL_DIR/score/$MODEL/"