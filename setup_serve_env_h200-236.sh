#!/usr/bin/env bash
# SGLang serving setup for this H200 machine.
#
# Usage (downloads packages and installs the environment):
#   bash setup_serve_env_h200-236.sh
# Optional path overrides:
#   WORK_ROOT=/path/to/writable/storage bash setup_serve_env_h200-236.sh
#   CONDA_BASE=/path/to/miniconda3 ENV_PREFIX=/path/to/gyz_serve bash ...
#
# Stack: Python 3.12, CUDA toolkit 12.8, PyTorch 2.9.1+cu128,
# SGLang 0.5.9, Transformers 4.57.1, FastMCP 3.1.0, project runtime deps.
# Keeps Miniconda, environments, caches and temporary files under WORK_ROOT
# (defaults to the current user's HOME, independently of the repository path).
# Does not run conda init, change HOME, modify existing launch scripts or start
# a model server. Reuses a completed installation on subsequent runs.
#
# Deliberately omit set -u: CUDA's conda activation hooks may read unset vars.
set -eo pipefail
trap 'echo "ERROR: setup failed at line $LINENO; fix the error above and rerun." >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-$HOME}"
CONDA_BASE="${CONDA_BASE:-$WORK_ROOT/miniconda3}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/gyz_serve}"
CACHE_ROOT="${CACHE_ROOT:-$WORK_ROOT/.cache/sglang}"
# These defaults deliberately avoid the nearly-full system /tmp partition.
export TMPDIR="${BUILD_TMPDIR:-$WORK_ROOT/tmp/sglang-build}"
export CONDA_PKGS_DIRS="$WORK_ROOT/.cache/conda/pkgs"
export CONDA_ENVS_PATH="$WORK_ROOT/envs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer-base"
export TVM_FFI_CACHE_DIR="$CACHE_ROOT/tvm-ffi"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=200
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-16}"

echo "Repository: $REPO_ROOT"
echo "Miniconda:  $CONDA_BASE"
echo "Serve env:  $ENV_PREFIX"
echo "Cache:      $CACHE_ROOT"
echo "Build tmp:  $TMPDIR"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "ERROR: this script targets Linux x86_64 H200 machines." >&2
    exit 1
fi
for tool in curl bash gcc g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: required tool '$tool' was not found in PATH." >&2
        exit 1
    fi
done
if [[ ! -f "$REPO_ROOT/requirements.txt" ]]; then
    echo "ERROR: missing $REPO_ROOT/requirements.txt" >&2
    exit 1
fi
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" \
    "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" \
    "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer" "$TVM_FFI_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "===== [1/6] bootstrap Miniconda (no shell-profile changes) ====="
if [[ -x "$CONDA_BASE/bin/conda" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "Reusing Miniconda: $CONDA_BASE"
else
    if [[ -e "$CONDA_BASE" ]]; then
        echo "ERROR: $CONDA_BASE exists but is not a complete Conda installation." >&2
        echo "Inspect it or select a different CONDA_BASE; nothing will be deleted." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$CONDA_BASE")" "$CACHE_ROOT/installers"
    INSTALLER="$CACHE_ROOT/installers/Miniconda3-latest-Linux-x86_64.sh"
    curl --fail --location --retry 3 --connect-timeout 20 \
        --output "$INSTALLER" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$INSTALLER" -b -p "$CONDA_BASE"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "===== [2/6] Python 3.12 and CUDA 12.8 toolchain ====="
# Override channels for every solver operation; do not use Anaconda defaults.
# The system nvcc is CUDA 13.0. Install a matching CUDA 12.8 toolchain here for
# the cu128 Torch wheels and runtime JIT. Driver 580 supports this CUDA runtime.
if [[ -d "$ENV_PREFIX/conda-meta" && -x "$ENV_PREFIX/bin/python" ]]; then
    CONDA_ACTION=install
elif [[ -e "$ENV_PREFIX" ]]; then
    echo "ERROR: $ENV_PREFIX exists but is not a complete Conda environment." >&2
    echo "Inspect it or choose another ENV_PREFIX; nothing will be deleted." >&2
    exit 1
else
    CONDA_ACTION=create
fi
conda "$CONDA_ACTION" --prefix "$ENV_PREFIX" --override-channels -c conda-forge \
    --yes 'python=3.12' pip 'cuda-version=12.8' 'cuda-nvcc=12.8.*' \
    'cuda-cudart-dev=12.8.*' 'cuda-cccl=12.8.*' libnuma
conda activate "$ENV_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: nvcc is missing from the environment: $CUDA_HOME/bin/nvcc" >&2
    exit 1
fi
"$CUDA_HOME/bin/nvcc" --version

echo "===== [3/6] CUDA 12.8 PyTorch wheels ====="
python -m pip install --no-input --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.9.1' 'torchvision==0.24.1' 'torchaudio==2.9.1'

echo "===== [4/6] SGLang and project dependencies ====="
# Resolve the project's unpinned dependencies alongside the serving stack so
# the resolver cannot silently replace SGLang's required versions.
python -m pip install --no-input \
    'sglang[all]==0.5.9' 'transformers==4.57.1' 'fastmcp==3.1.0' \
    'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' 'torchaudio==2.9.1+cu128' \
    --requirement "$REPO_ROOT/requirements.txt"

echo "===== [5/6] dependency and import checks ====="
python -m pip check
python - <<'PY'
from importlib.metadata import version
import sys
import torch
import transformers
import sglang
import sgl_kernel
import flashinfer
import fastmcp
import jsonlines
import litellm
import openai
import pandas

print("python:", sys.version.split()[0])
for package in ("torch", "torchvision", "torchaudio", "transformers", "sglang",
                "sgl-kernel", "flashinfer-python", "fastmcp", "litellm", "openai", "pandas"):
    print(f"{package}: {version(package)}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"ERROR: expected torch CUDA 12.8, got {torch.version.cuda}")
print("All imports and dependency checks passed.")
PY

echo "===== [6/6] GPU visibility check (no model loaded) ====="
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: packages installed, but CUDA is unavailable. Check GPU visibility/driver and rerun outside the sandbox.")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print("GPU visibility check passed; model loading and runtime JIT are not tested here.")
PY

echo "===== DONE: serving environment is ready ====="
# Print commands only. Cache/CUDA exports must also be set in future shells;
# conda activation alone does not persist this script's exports.
cat <<EOF

# Activate in a new shell:
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export CUDA_HOME="$ENV_PREFIX"
export CUDA_PATH="\$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export TMPDIR="$TMPDIR"
export HF_HOME="$HF_HOME"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE_BASE"
export TVM_FFI_CACHE_DIR="$TVM_FFI_CACHE_DIR"
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"

# Optional: launch Qwen3-8B after checking that GPU 3 and port 8000 are free.
# This command is printed, not executed. Ctrl+C stops the foreground server.
# Replace the model ID with your local model directory if already downloaded.
# Using the Hugging Face model ID below may download weights on first launch.
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \\
  --model-path "Qwen/Qwen3-8B" \\
  --host 127.0.0.1 --port 8000 --tp-size 1 --dtype bfloat16 \\
  --mem-fraction-static 0.9 --reasoning-parser qwen3 --tool-call-parser hermes
EOF
