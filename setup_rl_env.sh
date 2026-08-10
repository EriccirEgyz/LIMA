#!/usr/bin/env bash
# Build the envfactory-rl conda environment for EnvFactory RL reproduction.
#
# Stack is pinned to the verl 0.6 official known-good Docker
#   docker/verl0.6-cu128-torch2.8.0-fa2.7.4  (base + Dockerfile.app.sglang):
#   CUDA 12.8 / python 3.12 / torch 2.8.0+cu128 / transformers 4.55.4 / sglang[all] 0.5.2
#
# The verl fork is checked out at release/v0.6.1 (commit 99f4e378) in ./verl.
#
# flash-attn: verl 0.6's use_fused_kernels=True needs it. There is NO prebuilt wheel for
# flash_attn 2.7.4.post1 + torch 2.8 + cp312 (official release tops out at torch 2.7), so
# it is BUILT FROM SOURCE (exactly like the verl Docker). This is an OPTIONAL final stage
# below (~10-15 min); the env is usable without it if you set use_fused_kernels=False.
#
# Usage:  bash setup_rl_env.sh
# This is safe to re-run; conda create / pip install are idempotent.
# NOTE: `set -u` is intentionally omitted: conda's cuda-nvcc activate.d hook references
# unbound vars (NVCC_PREPEND_FLAGS) which -u turns into a fatal error on `conda activate`.
set -eo pipefail

REPO=/workspace/shenchengyu/yizhigao/envfactory_repro
VERL_DIR="$REPO/verl"
ENV_NAME=envfactory-rl
CONDA_BASE=/workspace/shenchengyu/miniconda3

# --- rootfs / is 100% full: force every cache / tmp onto /workspace ---
export TMPDIR=/workspace/shenchengyu/tmp
mkdir -p "$TMPDIR" "$REPO/.cache/pip"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export UV_CACHE_DIR=/workspace/shenchengyu/.uv-cache
export HF_HOME=/workspace/shenchengyu/hf-home
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=200

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "===== [1/8] conda create $ENV_NAME (python 3.12, conda-forge only) ====="
# conda-forge only (--override-channels) to avoid the anaconda `defaults` ToS gate.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env '$ENV_NAME' already exists, reusing it"
else
    conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.12 -y
fi
conda activate "$ENV_NAME"

echo "===== [2/8] CUDA 12.8 toolchain (nvcc; needed for sglang runtime JIT) + libnuma ====="
# Self-contained toolchain matching the SFT env recipe (conda-forge cuda-nvcc/cudart-dev/cccl).
# libnuma: sgl_kernel's prebuilt .so links against libnuma.so.1, absent from this container image.
conda install -n "$ENV_NAME" --override-channels -c conda-forge 'cuda-version=12.8' cuda-nvcc cuda-cudart-dev cccl libnuma -y
export CUDA_HOME="$CONDA_PREFIX"

echo "===== [3/8] torch 2.8.0 (cu128) ====="
pip install --no-input torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "===== [4/8] HF / verl core deps (transformers pinned to 4.55.4 per Docker) ====="
pip install --no-input \
    "transformers[hf_xet]==4.55.4" \
    "numpy<2.0.0" \
    accelerate datasets peft hf-transfer \
    tensordict torchdata pandas pyarrow huggingface_hub

echo "===== [5/8] sglang[all]==0.5.2 (rollout backend) ====="
# sglang[all] pulls its own torch-memory-saver (0.0.8) + flashinfer + sgl-kernel.
# Do NOT pin torch-memory-saver here: pinning 0.0.9rc1 conflicts with sglang's 0.0.8 requirement.
pip install --no-input "sglang[all]==0.5.2"

echo "===== [6/8] verl trainer dependencies ====="
pip install --no-input \
    "ray[default]>=2.10" hydra-core codetiming dill pybind11 pylatexenc \
    liger-kernel math_verify latex2sympy2_extended \
    fastapi uvicorn wandb tensorboard packaging

echo "===== [7/8] verl (editable, --no-deps) from fork checkout release/v0.6.1 ====="
pip install --no-input --no-deps -e "$VERL_DIR"

echo "===== [8/8] EnvFactory runtime extras ====="
pip install --no-input "fastmcp==3.1.0" jsonlines

echo "===== [align] re-pin verl-compatible versions (sglang/step-6 pulled newer) ====="
# sglang[all] and the latest trainer deps pull numpy>=2 / scipy>=1.16 / tensordict>0.10 /
# transformers>4.55, which violate verl 0.6.1's declared constraints. Pin them LAST so the
# final environment matches the verl 0.6 known-good stack.
pip install --no-input "numpy<2.0.0" "transformers==4.55.4" "tensordict==0.10.0" "scipy<1.16"

echo "===== [optional] flash_attn==2.7.4.post1 (source build; needs nvcc, ~10-15 min) ====="
# No prebuilt wheel exists for flash_attn 2.7.4.post1 + torch2.8 + cp312, so build from
# source against the installed torch (matches the verl 0.6 Docker). Skip by setting
# use_fused_kernels=False in the launch script. To re-run only this stage, comment out
# the stages above (env already exists / deps installed).
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS="${MAX_JOBS:-12}"   # tune to RAM: each job can use a few-to-tens of GB
if ! python -c "import flash_attn" 2>/dev/null; then
    pip install --no-input --no-build-isolation flash_attn==2.7.4.post1
fi

echo "===== verify imports ====="
python - <<'PY'
import torch, transformers, numpy, ray
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy       ", numpy.__version__)
print("ray         ", ray.__version__)
import sglang; print("sglang      ", sglang.__version__)
import verl;    print("verl        ", verl.__file__)
import fastmcp; print("fastmcp     ", fastmcp.__version__ if hasattr(fastmcp,'__version__') else 'ok')
import flash_attn; print("flash_attn  ", flash_attn.__version__)
print("ALL IMPORTS OK")
PY

echo "===== DONE: conda env '$ENV_NAME' is ready ====="
echo "Activate with:  conda activate $ENV_NAME"
