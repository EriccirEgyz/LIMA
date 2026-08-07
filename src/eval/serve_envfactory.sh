#!/bin/bash
# Serve EnvFactory-1.7B with SGLang for tau2-bench evaluation
#
# Aligned with EnvFactory paper:
#   - SGLang framework
#   - reasoning-parser qwen3 (Qwen3 thinking model)
#   - TP=1 sufficient for 1.7B; set TP=2 for larger models
#   - context-length not explicitly set; defaults to model's max_position_embeddings (32768)

set -e

MODEL=${MODEL:-"LARK-Lab/EnvFactory-1.7B"}
PORT=${PORT:-8000}
GPU_IDS=${GPU_IDS:-"0,1"}
TP=${TP:-2}

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "=== Serving $MODEL on port $PORT (GPU: $GPU_IDS, TP=$TP) ==="

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port $PORT \
  --tp-size $TP \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --reasoning-parser qwen3