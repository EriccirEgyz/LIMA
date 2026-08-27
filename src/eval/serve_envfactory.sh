#!/bin/bash
# Serve EnvFactory-1.7B with SGLang for tau2-bench evaluation
#
# Aligned with EnvFactory paper:
#   - SGLang framework
#   - reasoning-parser qwen3 (Qwen3 thinking model)
#   - TP=1 sufficient for 1.7B; set TP=2 for larger models
#   - context-length not explicitly set; defaults to the model's native
#     max_position_embeddings (40960 for EnvFactory-1.7B). No YaRN/rope-scaling
#     (config has rope_scaling=None), and none needed — tau2-bench conversations
#     stay well under 40k. Override with --context-length only if you must go
#     beyond 40960 (which would then require rope scaling).
#
# Usage: override any of MODEL / PORT / GPU_IDS / TP on the command line.
#   ./serve_envfactory.sh                                  # defaults: 1.7B on GPU 0,1 with TP=2
#   GPU_IDS=3 TP=1 ./serve_envfactory.sh                   # single GPU
#   MODEL=models/EnvFactory-1.7B ./serve_envfactory.sh    # local weights, no network
# Note: the number of GPUs in GPU_IDS must equal TP.
#
# MODEL takes either a HF repo id or a local directory. To pre-download via a
# mirror (useful when huggingface.co is unreachable):
#   HF_ENDPOINT=https://hf-mirror.com hf download LARK-Lab/EnvFactory-1.7B \
#     --local-dir models/EnvFactory-1.7B

set -e

MODEL=${MODEL:-"LARK-Lab/EnvFactory-1.7B"}
PORT=${PORT:-8000}
GPU_IDS=${GPU_IDS:-"0,1"}
TP=${TP:-2}
# Default float32 (NOT bf16): SFT-1.7B in bf16 spuriously emits an extra '}' at the
# end of tool_call JSON ("}}}" instead of "}}"). sglang's hermes tool-call parser
# then fails -> tau2 sees an empty AssistantMessage -> retail tasks die first turn
# with "AssistantMessage must have either content or tool_calls". Same weights are
# clean under float32 (greedy stays on the right side of the }} vs <|im_end|> logit
# boundary). ~2x slower/VRAM, fine for 1.7B on H100. mem-fraction-static stays 0.9;
# if float32 OOMs, lower it manually.
DTYPE=${DTYPE:-bfloat16}

export CUDA_VISIBLE_DEVICES=$GPU_IDS

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Workarounds for an incomplete container image (no root available) --------
# Both of these should be deleted once the image ships libnuma1 and
# cuda-nvcc-12-8 (`apt-get install -y libnuma1 cuda-nvcc-12-8`).
#
# 1. sgl_kernel's prebuilt .so links against libnuma.so.1, absent from this
#    image. .venv/extra-lib holds a copy from conda-forge.
export LD_LIBRARY_PATH="$REPO_ROOT/.venv/extra-lib:$LD_LIBRARY_PATH"
#
# 2. sglang 0.5.9 JIT-compiles its rope kernel at CUDA-graph capture time, so it
#    needs nvcc. /usr/local/cuda here is runtime-only (no bin/), so we point
#    CUDA_HOME at a CUDA 12.8 toolchain assembled from conda-forge packages
#    (matching torch's cu128). ninja lives in .venv/bin and must be on PATH too.
export CUDA_HOME="$REPO_ROOT/.venv/cuda-nvcc"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"
# -----------------------------------------------------------------------------

# 同卡上没有其它程序时，--mem-fraction-static可以设到0.9，有其它的话要降低一些，比如到0.7

echo "=== Serving $MODEL on port $PORT (GPU: $GPU_IDS, TP=$TP) ==="

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port $PORT \
  --tp-size $TP \
  --dtype $DTYPE \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes