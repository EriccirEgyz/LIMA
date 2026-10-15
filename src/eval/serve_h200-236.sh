#!/usr/bin/env bash
# Serve local Qwen3-8B with SGLang on this H200 machine.
# Environment: setup_serve_env_h200-236.sh (SGLang 0.5.9, CUDA 12.8).
#
# Usage (check GPU availability before starting):
#   bash src/eval/serve_h200-236.sh
#   GPU_IDS=4 PORT=8001 bash src/eval/serve_h200-236.sh
#   GPU_IDS=3,4 TP=2 MODEL=models/your-checkpoint bash src/eval/serve_h200-236.sh
#   SERVE_HOST=0.0.0.0 bash src/eval/serve_h200-236.sh
#
# WORK_ROOT defaults to HOME, matching setup_serve_env_h200-236.sh.
# Relative MODEL paths resolve from the repo root; HF repo IDs also work.
# The model's native context length is used unless CONTEXT_LENGTH is set.
# Additional SGLang flags can be passed as script arguments.
# Omit set -u because conda CUDA activation hooks may read unset variables.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$HOME}"
CONDA_BASE="${CONDA_BASE:-$WORK_ROOT/miniconda3}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/gyz_serve}"
MODEL="${MODEL:-$REPO_ROOT/models/Qwen3-8B}"
# Conda's compiler activation hooks overwrite HOST with a toolchain triplet
# (e.g. x86_64-conda-linux-gnu). Keep the listening address independent of HOST,
# including when the calling shell already has a Conda environment activated.
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-3}"
TP="${TP:-1}"
DTYPE="${DTYPE:-bfloat16}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"

if [[ ! "$TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TP must be a positive integer (got '$TP')." >&2
    exit 1
fi
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: GPU_IDS must be comma-separated numeric GPU IDs (e.g. 3 or 3,4)." >&2
    exit 1
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$TP" ]]; then
    echo "ERROR: GPU_IDS=$GPU_IDS contains ${#GPU_ARRAY[@]} GPU(s), but TP=$TP." >&2
    exit 1
fi
if [[ ! "$PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
    echo "ERROR: PORT must be an integer between 1 and 65535." >&2
    exit 1
fi
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" || ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "ERROR: Miniconda or the serving environment is missing." >&2
    echo "Expected Miniconda: $CONDA_BASE; environment: $ENV_PREFIX" >&2
    echo "First run: bash $REPO_ROOT/setup_serve_env_h200-236.sh" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
# Use the environment's CUDA 12.8 compiler and libnuma, not system CUDA 13.0
# or the H100 container's .venv/extra-lib and .venv/cuda-nvcc workarounds.
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-16}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: CUDA compiler not found: $CUDA_HOME/bin/nvcc" >&2
    exit 1
fi

# Share caches with setup_serve_env_h200-236.sh and keep JIT/temp files on /data.
CACHE_ROOT="${CACHE_ROOT:-$WORK_ROOT/.cache/sglang}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export TMPDIR="${BUILD_TMPDIR:-$WORK_ROOT/tmp/sglang-build}"
# shellcheck disable=SC1091
source "$REPO_ROOT/src/eval/cache_env.sh"
mkdir -p "$TMPDIR"

EXTRA_ARGS=()
if [[ -n "${CONTEXT_LENGTH:-}" ]]; then
    EXTRA_ARGS+=(--context-length "$CONTEXT_LENGTH")
fi

cd "$REPO_ROOT"
echo "=== Serving $MODEL at $SERVE_HOST:$PORT (GPU: $GPU_IDS, TP=$TP) ==="
echo "Environment: $CONDA_PREFIX | CUDA_HOME: $CUDA_HOME"
exec "$CONDA_PREFIX/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SERVE_HOST" \
    --port "$PORT" \
    --tp-size "$TP" \
    --dtype "$DTYPE" \
    --mem-fraction-static "$MEM_FRACTION" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --tool-call-parser hermes \
    "${EXTRA_ARGS[@]}" "$@"
