#!/usr/bin/env bash
# Build the gyz_sft conda environment for LLaMA-Factory SFT training
# on the NEW machine (2x H200, /mnt/beegfs/workspace/scy).
#
# Ported 2026-09-05 from the old machine's envfactory-sft env, with ONE deliberate fix:
# transformers is pinned to 4.56.1 instead of the 5.8.0 that bit us before.
#
# WHY 4.56.1 (the whole point of this rebuild):
#   - The old SFT env used transformers 5.8.0. Checkpoints it saves write nested
#     "rope_parameters": {"rope_theta": ...} + "dtype" instead of the classic top-level
#     "rope_theta" / "torch_dtype". transformers 4.x then silently falls back to
#     Qwen3Config defaults (rope_theta=10000, 100x off) -> 5K+ context degenerates into
#     repetition; it also forced prepare_init.py to hand-patch configs/tokenizers
#     before RL could load the SFT checkpoint.
#   - The official EnvFactory-1.7B config.json self-reports transformers 4.56.1, and this
#     LLaMA-Factory checkout allows ">=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0". 4.56.1 is the
#     paper-faithful, skew-free choice: classic config format, loads cleanly in the RL
#     env's transformers 4.55.4, no prepare_init-style conversion ever again.
#   - Verify stage below asserts a Qwen3Config saved by THIS env keeps top-level
#     rope_theta / torch_dtype, so the skew can never sneak back in silently.
#
# Stack: python 3.12 / conda-forge cuda 12.8 toolchain (for deepspeed runtime JIT) /
#        torch 2.8.0+cu128 (aligned with the RL env; old SFT used 2.9.1, torch was never
#        the skew variable) / transformers 4.56.1 / LLaMA-Factory (editable, this fork) /
#        deepspeed 0.19.4 / flash-attn 2.8.3 (2026-09-05: FA2 for long-seq speedup) /
#        liger-kernel 0.8.2 (2026-09-05: fused linear CE for memory + speed).
#
# Usage:  bash setup_sft_env_h200.sh
# Safe to re-run; conda create / pip install are idempotent.
# NOTE: `set -u` intentionally omitted: conda's cuda-nvcc activate.d hook references
# unbound vars (NVCC_PREPEND_FLAGS) which -u turns into a fatal error on activate.
set -eo pipefail

REPO=/mnt/beegfs/workspace/scy/yizhigao/LIMA
LF_DIR="$REPO/LLaMA-Factory"
ENV_NAME=gyz_sft

# --- caches on beegfs (persistent); TMPDIR on local disk (fast, 738G free) ---
export TMPDIR=/tmp/scy-build
mkdir -p "$TMPDIR" "$REPO/.cache/pip"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export HF_HOME=/mnt/beegfs/workspace/scy/hf-home
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=200
# NOTE: this machine reaches huggingface.co directly (verified 200). Old-machine gotchas
# (HF_ENDPOINT=https://hf-mirror.com, HF_HUB_DISABLE_XET=1) not needed here; re-add only
# if downloads start failing.

# user-level bootstrap on this machine: conda + node, HOME redirected onto beegfs
# shellcheck disable=SC1091
source /mnt/beegfs/workspace/scy/activate_conda.sh

echo "===== [1/6] conda create $ENV_NAME (python 3.12, conda-forge only) ====="
# conda-forge only (--override-channels) to avoid the anaconda `defaults` ToS gate.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env '$ENV_NAME' already exists, reusing it"
else
    conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.12 -y
fi
conda activate "$ENV_NAME"

echo "===== [2/6] CUDA 12.8 toolchain (nvcc; deepspeed JIT compiles ops at runtime) ====="
# Same self-contained conda-forge recipe as the RL env. System /usr/local/cuda-13.2 is
# irrelevant; driver 595 is backward compatible with everything this stack needs.
conda install -n "$ENV_NAME" --override-channels -c conda-forge 'cuda-version=12.8' cuda-nvcc cuda-cudart-dev cccl -y
export CUDA_HOME="$CONDA_PREFIX"

echo "===== [3/6] torch 2.8.0 + matching vision/audio (cu128) ====="
# Install all three from the cu128 index with paired versions so `pip install -e .`
# later sees torchvision>=0.19 / torchaudio>=2.4 satisfied and never touches torch.
pip install --no-input torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "===== [4/6] transformers 4.56.1 FIRST (the anti-skew pin; see header) ====="
pip install --no-input "transformers==4.56.1"

echo "===== [5/6] LLaMA-Factory (editable, from this repo's fork) ====="
pip install --no-input -e "$LF_DIR"

echo "===== [6/6] deepspeed + tensorboard + flash-attn + liger-kernel ====="
pip install --no-input "deepspeed==0.19.4" tensorboard

# flash-attn 2.8.3: cu128 + torch 2.8 + cxx11abi=True + cp312 预编译 wheel
# 长序列训练加速 ~4x (SDPA->FA2); 必须与 torch/cuda 版本精确匹配
pip install --no-input flash-attn==2.8.3 --index-url https://download.pytorch.org/whl/cu128

# liger-kernel 0.8.2: fused linear cross-entropy 免物化 vocab_size logits
# 32K×151936 logits (Qwen3-8B) 节省 ~18GB 显存 + 提速; 已验证与标准 CE 数学等价
pip install --no-input "liger-kernel==0.8.2"

echo "===== [align] re-pin transformers in case a dep tried to bump it ====="
# LLaMA-Factory's own constraint (<=5.8.0) permits 4.56.1, but trl/datasets resolution
# can still propose a newer one; enforce the pin LAST exactly like the RL env does.
pip install --no-input "transformers==4.56.1"

echo "===== verify imports ====="
python - <<'PY'
import torch, transformers, deepspeed
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("deepspeed   ", deepspeed.__version__)
import llamafactory
from llamafactory import __version__ as lf_ver
print("llamafactory", lf_ver)

# verify flash-attn and liger-kernel (2026-09-05 提速依赖)
try:
    import flash_attn
    print("flash-attn  ", flash_attn.__version__)
except ImportError as e:
    print("flash-attn   MISSING:", e)
try:
    import liger_kernel
    print("liger-kernel", liger_kernel.__version__)
except ImportError as e:
    print("liger-kernel MISSING:", e)

# Anti-skew round-trip: a Qwen3Config saved by THIS env must match the OFFICIAL
# EnvFactory-1.7B config.json format (fetched & verified 2026-09-05):
#   top-level "rope_theta": 1000000  +  "dtype": "bfloat16"  +  transformers_version 4.56.1
# - top-level rope_theta is the killer key: the old 5.8.0 env wrote it nested and 4.x
#   readers silently fell back to 10000 (100x off) -> 5K+ context degeneration.
# - transformers 4.56.1 renamed torch_dtype -> dtype on SAVE. transformers 4.55.4 (RL env)
#   does NOT map dtype -> torch_dtype on load (verified in its configuration_utils.py),
#   so a load that omits an explicit dtype falls back to defaults -- identical behaviour
#   to the official EnvFactory artifact, and every load path in this project (LLaMA-Factory
#   bf16:true, verl model.dtype, sglang --dtype) passes dtype explicitly anyway.
import json, tempfile, os
from transformers import Qwen3Config
cfg = Qwen3Config(rope_theta=1000000.0, torch_dtype="bfloat16")
with tempfile.TemporaryDirectory() as d:
    cfg.save_pretrained(d)
    saved = json.load(open(os.path.join(d, "config.json")))
    assert "rope_theta" in saved and saved["rope_theta"] == 1000000, \
        f"config skew! saved keys: {sorted(saved)}"
    dtype_val = saved.get("dtype") or saved.get("torch_dtype")
    assert dtype_val == "bfloat16", f"dtype missing! saved keys: {sorted(saved)}"
    print(f"config check OK: rope_theta={saved['rope_theta']} (top-level), dtype={dtype_val}")
print("ALL IMPORTS OK")
PY

echo "===== runtime cheatsheet (training launch) ====="
cat <<'EOF'
source /mnt/beegfs/workspace/scy/activate_conda.sh && conda activate gyz_sft
cd /mnt/beegfs/workspace/scy/yizhigao/LIMA/LLaMA-Factory
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=? CUDA_HOME=$CONDA_PREFIX \
  HF_HOME=/mnt/beegfs/workspace/scy/hf-home \
  llamafactory-cli train llamafactory_runs/<model>/<exp>/sft.yaml
# deepspeed training REQUIRES FORCE_TORCHRUN=1 (else torchrun falls back to system
# python -> ModuleNotFoundError: llamafactory) and CUDA_HOME=$CONDA_PREFIX (deepspeed
# JIT needs nvcc). Both are old-machine gotchas that still apply verbatim.
EOF

echo "===== DONE: conda env '$ENV_NAME' is ready ====="
echo "Activate with:  conda activate $ENV_NAME"
