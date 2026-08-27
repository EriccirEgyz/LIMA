#!/bin/bash
# Run BFCL v3 evaluation for EnvFactory-1.7B
#
# Aligned with EnvFactory paper (arXiv 2605.18703), Table 2:
#   - Single-turn column = mean of the 13 single_turn subcategory accuracies
#   - Multi-turn  column = mean of the 4  multi_turn  subcategory accuracies
#     (subcategories: base, miss_func, miss_param, long_context)
#   - Both are unweighted means of per-category accuracies; BFCL writes each
#     category's accuracy to score/<model>/BFCL_v3_<cat>_score.json. Average the
#     relevant files to compare against the paper (e.g. Qwen3-1.7B base:
#     single=79.48, multi=16.75).
#   - SGLang serving (reuses serve_envfactory.sh), temp 0.7 for thinking models.
#
# How custom weights plug in (no BFCL source edits needed):
#   EnvFactory-1.7B is a Qwen3-1.7B finetune that keeps Qwen3's tokenizer, chat
#   template, and FC output format, so we reuse BFCL's *local* Qwen3-1.7B handler
#   by passing the registered name "Qwen/Qwen3-1.7B-FC" as --model, and override
#   the weights via --local-model-path. Note: NOT "qwen3-1.7b-FC" (lowercase) —
#   that one uses QwenAPIHandler and hits the DashScope API.
#
# Prerequisites:
#   1. bash src/eval/setup_bfcl.sh
#   2. bash src/eval/serve_envfactory.sh  (wait until server is ready)
#
# Usage:
#   bash src/eval/run_bfcl.sh                                  # single_turn + multi_turn (paper)
#   bash src/eval/run_bfcl.sh --category multi_turn           # multi-turn only
#   bash src/eval/run_bfcl.sh --category all                  # everything (v3-wide)

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"

# === Defaults (aligned with paper: the two reported columns) ===
# MODEL = local weights dir (used for --local-model-path so the tokenizer loads
#         offline and the request's model field matches sglang's served name).
# BFCL_MODEL_NAME = the registered BFCL name whose handler we reuse.
MODEL=${MODEL:-"$REPO_ROOT/models/EnvFactory-1.7B"}
BFCL_MODEL_NAME=${BFCL_MODEL_NAME:-"Qwen/Qwen3-1.7B-FC"}
CATEGORY=${CATEGORY:-"single_turn,multi_turn"}
SGLANG_PORT=${SGLANG_PORT:-8000}
# Paper (Appendix F): temp 0.7 for thinking models. BFCL's default is 0.001
# (greedy), which diverges from the paper — keep 0.7 here.
TEMP=${TEMP:-0.7}

# === Parse args ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --category) CATEGORY="$2"; shift 2;;
        --port) SGLANG_PORT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

GORILLA_DIR="$EVAL_DIR/gorilla"
BFCL_DIR="$GORILLA_DIR/berkeley-function-call-leaderboard"

# Isolate outputs per checkpoint. BFCL keys result/score dirs by the registered
# model name (here the shared "Qwen/Qwen3-1.7B-FC"), so without this every
# Qwen3-1.7B finetune would overwrite the same files. Setting BFCL_PROJECT_ROOT
# relocates BFCL's whole result/ + score/ tree under a per-checkpoint dir; both
# generate and evaluate derive their paths from it, so no per-command flags.
#
# Without a further split, re-running is dangerous: when a result_*.json already
# exists and --allow-overwrite isn't passed, BFCL's collect_test_cases()
# (_llm_response_generation.py) loads the EXISTING results and only generates
# responses for missing test ids -- it does not know or care that you changed
# temperature, dtype, TP, or the sglang version, so a re-run can silently keep
# grading stale responses. Rather than encode every one of those knobs into the
# path (easy to miss one), follow the same fix used in run_tau2_bench.sh: give
# each invocation its own timestamped run dir, so there's never a pre-existing
# result file to accidentally reuse. Override with BFCL_RUN_ROOT=... (e.g. to
# intentionally resume/extend a specific prior run).
CKPT_NAME=$(basename "$MODEL")                       # e.g. EnvFactory-1.7B
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
BFCL_RUN_ROOT=${BFCL_RUN_ROOT:-"$REPO_ROOT/results/bfcl/$CKPT_NAME/$RUN_TAG"}
export BFCL_PROJECT_ROOT="$BFCL_RUN_ROOT"
MODEL_DIR_ESCAPED=${BFCL_MODEL_NAME//\//_}           # Qwen/Qwen3-1.7B-FC → Qwen_Qwen3-1.7B-FC

# === Validate ===
if [ ! -d "$BFCL_DIR/.venv" ]; then
    echo "ERROR: BFCL not installed. Run: bash src/eval/setup_bfcl.sh"
    exit 1
fi

if [ ! -d "$MODEL" ]; then
    echo "ERROR: weights dir not found: $MODEL"
    echo "Download it first, or pass MODEL=/path/to/checkpoint"
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
echo " Weights:        $MODEL"
echo " BFCL name:      $BFCL_MODEL_NAME  (handler reused)"
echo " Categories:     $CATEGORY"
echo " SGLang Port:    $SGLANG_PORT"
echo " Run tag:        $RUN_TAG"
echo " Output root:    $BFCL_RUN_ROOT"
echo "=============================================="

cd "$BFCL_DIR"

# Point BFCL at the existing SGLang server started by serve_envfactory.sh.
# NOTE: BFCL names these VLLM_ENDPOINT/VLLM_PORT for historical reasons, but they
# apply to BOTH backends (vllm and sglang) — there is no SGLANG_ENDPOINT. The
# previous LOCAL_SERVER_ENDPOINT/LOCAL_SERVER_PORT vars are never read by BFCL
# (it would fall back to localhost:1053). Default port is 1053; ours is 8000.
export VLLM_ENDPOINT="localhost"
export VLLM_PORT="$SGLANG_PORT"

# Call the venv's bfcl binary directly instead of `uv run bfcl`. The venv is
# created by setup_bfcl.sh via `uv venv` + `uv pip install` (manual mode). Using
# `uv run` here would put uv into project mode, regenerate uv.lock, and sync the
# venv to the freshly-resolved dependency set — re-downloading torch etc. that
# this HTTP client doesn't even need (--skip-server-setup talks to the external
# SGLang server, no model is loaded here).
BFCL_BIN="$BFCL_DIR/.venv/bin/bfcl"

# === Step 1: Generate responses ===
echo ""
echo ">>> Step 1: Generating model responses..."

"$BFCL_BIN" generate \
    --model "$BFCL_MODEL_NAME" \
    --local-model-path "$MODEL" \
    --test-category $CATEGORY \
    --temperature "$TEMP" \
    --backend sglang \
    --skip-server-setup

# === Step 2: Evaluate ===
echo ""
echo ">>> Step 2: Evaluating responses..."

"$BFCL_BIN" evaluate \
    --model "$BFCL_MODEL_NAME" \
    --test-category $CATEGORY

echo ""
echo "=== BFCL evaluation complete ==="
echo "Per-category accuracies: $BFCL_RUN_ROOT/score/$MODEL_DIR_ESCAPED/"
echo ""
echo "Compare to paper Table 2 (Qwen3-1.7B base: single=79.48, multi=16.75):"
echo "  single-turn = mean of BFCL_v3_{simple,irrelevance,parallel,multiple,"
echo "                  parallel_multiple,java,javascript,live_simple,live_multiple,"
echo "                  live_parallel,live_parallel_multiple,live_irrelevance,live_relevance}_score.json"
echo "  multi-turn  = mean of BFCL_v3_multi_turn_{base,miss_func,miss_param,long_context}_score.json"