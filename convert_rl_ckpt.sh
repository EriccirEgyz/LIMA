#!/bin/bash
# Convert verl FSDP checkpoint to HuggingFace format for evaluation
set -e

CKPT_DIR=${1:-"/workspace/shenchengyu/yizhigao/envfactory_repro/verl/checkpoints/EnvFactory/EnvFactory-RL-0818/global_step_110"}
OUTPUT_DIR=${2:-"/workspace/shenchengyu/yizhigao/envfactory_repro/verl/checkpoints/EnvFactory/EnvFactory-RL-0818/global_step_110_hf"}

# Detect base model from config
HF_MODEL_CONFIG=$(cat "$CKPT_DIR/actor/huggingface/config.json")

echo "=== Converting FSDP checkpoint to HuggingFace format ==="
echo "Source: $CKPT_DIR/actor"
echo "Target: $OUTPUT_DIR"

cd /workspace/shenchengyu/yizhigao/envfactory_repro/verl

# Activate the conda environment for RL
source /workspace/shenchengyu/miniconda3/etc/profile.d/conda.sh
conda activate envfactory-rl

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "$CKPT_DIR/actor" \
    --target_dir "$OUTPUT_DIR"

echo ""
echo "=== Conversion complete ==="
echo "You can now evaluate with:"
echo "  MODEL=$OUTPUT_DIR ./src/eval/serve_envfactory.sh"
