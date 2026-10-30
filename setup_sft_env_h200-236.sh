#!/usr/bin/env bash
# LLaMA-Factory SFT setup for this H200 machine, following setup_serve_env_h200-236.sh.
#
# Usage (downloads packages and installs the environment):
#   bash setup_sft_env_h200-236.sh
# Optional path overrides:
#   WORK_ROOT=/path/to/writable/storage bash setup_sft_env_h200-236.sh
#   CONDA_BASE=/path/to/miniconda3 ENV_PREFIX=/path/to/gyz_sft bash ...
#
# Stack: Python 3.12, CUDA toolkit 12.8, PyTorch 2.8.0+cu128,
# Transformers 4.56.1, this checkout's editable LLaMA-Factory, DeepSpeed 0.19.4,
# FlashAttention 2.8.3, Liger Kernel 0.8.2 and TensorBoard.
# Keep Transformers 4.56.1 for the existing Qwen3 training recipe: the config
# check below guards against saving rope_theta in a format older loaders miss.
# dtype/torch_dtype spelling may differ between consumers; load with explicit bf16.
#
# Keeps environments, caches and build files under WORK_ROOT (defaults to HOME).
# Reuses complete Conda installations; does not change shell profiles or HOME,
# edit training YAMLs, download models or start training.
#
# Deliberately omit set -u: CUDA's conda activation hooks may read unset vars.
set -eo pipefail
trap 'echo "ERROR: setup failed at line $LINENO; fix the error above and rerun." >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_DIR="$REPO_ROOT/LLaMA-Factory"
WORK_ROOT="${WORK_ROOT:-$HOME}"
CONDA_BASE="${CONDA_BASE:-$WORK_ROOT/miniconda3}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/gyz_sft}"
CACHE_ROOT="${CACHE_ROOT:-$WORK_ROOT/.cache/sft}"
# Avoid the nearly-full system /tmp partition, including pip builds and JIT.
export TMPDIR="${BUILD_TMPDIR:-$WORK_ROOT/tmp/sft-build}"
export CONDA_PKGS_DIRS="$WORK_ROOT/.cache/conda/pkgs"
export CONDA_ENVS_PATH="$WORK_ROOT/envs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch-extensions"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=200
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-16}"

# Carry these pins through every later pip resolution; do not rely on repinning
# Transformers after another package has already selected incompatible deps.
TORCH_PINS=('torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128')
CORE_PINS=("${TORCH_PINS[@]}" 'transformers==4.56.1')
# Official v2.8.3 release asset for CPython 3.12 / Torch 2.8 / CUDA 12 / ABI TRUE.
# The PyTorch cu128 package index does not host this FlashAttention 2 package.
FLASH_ATTN_WHEEL_URL='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb'

echo "Repository: $REPO_ROOT"
echo "Miniconda:  $CONDA_BASE"
echo "SFT env:    $ENV_PREFIX"
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
if [[ ! -f "$LF_DIR/pyproject.toml" ]]; then
    echo "ERROR: missing LLaMA-Factory checkout: $LF_DIR/pyproject.toml" >&2
    exit 1
fi
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" \
    "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" \
    "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "===== [1/7] bootstrap Miniconda (no shell-profile changes) ====="
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

echo "===== [2/7] Python 3.12 and CUDA 12.8 toolchain ====="
# Override channels for every solver operation; do not use Anaconda defaults.
# System nvcc is CUDA 13.0; use an environment-local 12.8 toolchain for cu128 JIT.
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
    'cuda-cudart-dev=12.8.*' 'cuda-cccl=12.8.*'
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

echo "===== [3/7] CUDA 12.8 PyTorch wheels ====="
python -m pip install --no-input --index-url https://download.pytorch.org/whl/cu128 \
    "${TORCH_PINS[@]}"
# Fail before downloading the native FlashAttention wheel if its ABI cannot match.
python - <<'PY'
import sys
import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"ERROR: expected Python 3.12, got {sys.version}")
if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
    raise SystemExit(f"ERROR: unexpected Torch/CUDA: {torch.__version__}, {torch.version.cuda}")
if not torch.compiled_with_cxx11_abi():
    raise SystemExit("ERROR: the selected FlashAttention wheel requires CXX11 ABI TRUE.")
print("Python, Torch, CUDA and CXX11 ABI match the FlashAttention wheel.")
PY

echo "===== [4/7] LLaMA-Factory, FlashAttention and Liger Kernel ====="
# Resolve the editable checkout and training dependencies with the core pins.
# Keep build isolation for LLaMA-Factory's Hatchling backend. Install DeepSpeed's
# build prerequisites here so its separate build below can see this exact Torch.
python -m pip install --no-input "${CORE_PINS[@]}" \
    --editable "$LF_DIR" 'liger-kernel==0.8.2' "$FLASH_ATTN_WHEEL_URL" \
    tensorboard setuptools wheel packaging ninja psutil

echo "===== [5/7] DeepSpeed (runtime JIT with the installed Torch) ====="
DS_BUILD_OPS=0 python -m pip install --no-input --no-build-isolation \
    "${CORE_PINS[@]}" 'liger-kernel==0.8.2' 'flash-attn==2.8.3' 'deepspeed==0.19.4'

echo "===== [6/7] dependency, import and Qwen3 config checks ====="
python -m pip check
python - "$LF_DIR" <<'PY'
from importlib.metadata import version
from pathlib import Path
import json
import sys
import tempfile

import torch
import torchvision
import torchaudio
import transformers
import deepspeed
import flash_attn
import liger_kernel
import llamafactory
import tensorboard
from transformers import Qwen3Config, Qwen3ForCausalLM
from llamafactory.hparams import get_train_args

expected = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "torchaudio": "2.8.0+cu128",
    "transformers": "4.56.1",
    "deepspeed": "0.19.4",
    "flash-attn": "2.8.3",
    "liger-kernel": "0.8.2",
}
for package, wanted in expected.items():
    actual = version(package)
    print(f"{package}: {actual}")
    # DeepSpeed/FlashAttention wheels may include a local build suffix.
    checked = actual.split("+", 1)[0] if package in ("deepspeed", "flash-attn") else actual
    if checked != wanted:
        raise SystemExit(f"ERROR: expected {package}=={wanted}, got {actual}")
for package in ("llamafactory", "tensorboard", "accelerate", "peft", "trl", "datasets"):
    print(f"{package}: {version(package)}")
if torch.version.cuda != "12.8" or not torch.compiled_with_cxx11_abi():
    raise SystemExit("ERROR: Torch CUDA/ABI changed while installing training dependencies.")
if not Path(llamafactory.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()):
    raise SystemExit(f"ERROR: LLaMA-Factory was imported from another checkout: {llamafactory.__file__}")

# Test saving and reloading a config only; no model weights are loaded/downloaded.
cfg = Qwen3Config(rope_theta=1000000.0, torch_dtype="bfloat16")
with tempfile.TemporaryDirectory(prefix="sft-config-check-") as directory:
    cfg.save_pretrained(directory)
    saved = json.loads((Path(directory) / "config.json").read_text())
    if saved.get("rope_theta") != 1000000:
        raise SystemExit(f"ERROR: config lost top-level rope_theta: {sorted(saved)}")
    dtype_value = saved.get("dtype") or saved.get("torch_dtype")
    if dtype_value != "bfloat16":
        raise SystemExit(f"ERROR: unexpected saved dtype: {dtype_value}")
    loaded = Qwen3Config.from_pretrained(directory, local_files_only=True)
    if loaded.rope_theta != 1000000:
        raise SystemExit(f"ERROR: config round-trip changed rope_theta: {loaded.rope_theta}")
    print(f"Qwen3 config check passed: rope_theta={saved['rope_theta']}, dtype={dtype_value}")
print("All imports, pinned versions and dependency checks passed.")
PY

echo "===== [7/7] GPU visibility check (no model loaded) ====="
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: packages installed, but CUDA is unavailable. Check GPU visibility/driver and rerun outside the sandbox.")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
print("GPU visibility check passed; training, GPU kernels and DeepSpeed JIT are not tested here.")
PY

echo "===== DONE: SFT environment installation and checks passed ====="
# Print commands only. Conda activation does not persist the cache/CUDA exports.
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
export TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export MAX_JOBS="$MAX_JOBS"
cd "$LF_DIR"

# Before training, select available GPUs and update the YAML's model/data/output
# paths for this machine. Use an absolute output_dir under $REPO_ROOT/models.
# Substitute your GPU IDs and YAML path in this example; it is not executed:
# FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=<available-gpu-ids> \\
#   llamafactory-cli train llamafactory_runs/<model>/<experiment>/sft.yaml
EOF
