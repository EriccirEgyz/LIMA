#!/usr/bin/env bash
# Build the gyz_serve conda environment for SGLang 0.5.9 serving
# on the NEW machine (8x H200, /mnt/beegfs/workspace/scy).
#
# This is the standalone serving stack consumed by serve_envfactory.sh and tau2/BFCL
# evaluation scripts. It is SEPARATE from the RL (gyz_rl, verl 0.6.1) and SFT
# (gyz_sft, LLaMA-Factory) training environments.
#
# Stack: python 3.12 / conda-forge cuda 12.8 toolchain (for sglang runtime JIT) /
#        torch 2.9.1+cu124 (pinned by sglang 0.5.9) / transformers 4.57.1 /
#        sglang 0.5.9 / fastmcp 3.1.0 + project requirements.txt runtime deps.
#
# serve_envfactory.sh workarounds (libnuma, cuda-nvcc) are BUILT INTO this env:
#   - libnuma: sgl_kernel's prebuilt .so links against libnuma.so.1
#   - cuda-nvcc: sglang JIT-compiles rope kernel at CUDA-graph capture time
# Both are installed via conda here, so serve_envfactory.sh can source this env
# directly without extra LD_LIBRARY_PATH / CUDA_HOME gymnastics.
#
# Usage:  bash setup_serve_env_h200.sh
# Safe to re-run; conda create / pip install are idempotent.
# NOTE: `set -u` intentionally omitted: conda's cuda-nvcc activate.d hook references
# unbound vars (NVCC_PREPEND_FLAGS) which -u turns into a fatal error on activate.
set -eo pipefail

REPO=/mnt/beegfs/workspace/scy/yizhigao/LIMA
ENV_NAME=gyz_serve

# --- caches on beegfs (persistent); TMPDIR on local disk (fast, 735G free) ---
export TMPDIR=/tmp/scy-build
mkdir -p "$TMPDIR" "$REPO/.cache/pip"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export HF_HOME=/mnt/beegfs/workspace/scy/hf-home
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=200

# user-level bootstrap on this machine: conda + node, HOME redirected onto beegfs
# shellcheck disable=SC1091
source /mnt/beegfs/workspace/scy/activate_conda.sh

echo "===== [1/5] conda create $ENV_NAME (python 3.12, conda-forge only) ====="
# conda-forge only (--override-channels) to avoid the anaconda `defaults` ToS gate.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env '$ENV_NAME' already exists, reusing it"
else
    conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.12 -y
fi
conda activate "$ENV_NAME"

echo "===== [2/5] CUDA 12.8 toolchain (nvcc; sglang JIT) + libnuma ====="
# Same self-contained conda-forge recipe as the RL/SFT envs. System /usr/local/cuda-13.2
# is irrelevant; driver 595 is backward compatible with everything this stack needs.
# libnuma: sgl_kernel's prebuilt .so links against libnuma.so.1.
conda install -n "$ENV_NAME" --override-channels -c conda-forge 'cuda-version=12.8' cuda-nvcc cuda-cudart-dev cccl libnuma -y
export CUDA_HOME="$CONDA_PREFIX"

echo "===== [3/5] sglang[all]==0.5.9 (pulls torch 2.9.1+cu124) ====="
# sglang 0.5.9 pins torch==2.9.1, transformers==4.57.1, sgl-kernel==0.3.21,
# flashinfer_python==0.6.3. Install it first so its torch constraint wins.
pip install --no-input "sglang[all]==0.5.9"

echo "===== [4/5] fastmcp + project runtime deps (requirements.txt) ====="
# fastmcp 3.1.0: MCP server framework used by rollout scripts and serve_envfactory.sh.
# requirements.txt: jsonlines, toml, python-dotenv, matplotlib, networkx, scipy,
# openai-agents, litellm, nest-asyncio, pytz, pylint, flask, mpmath, pyvis, pandas,
# pyarrow, seaborn, aiofiles -- runtime deps for evaluation / trajectory synthesis.
pip install --no-input "fastmcp==3.1.0"
pip install --no-input -r "$REPO/requirements.txt"

echo "===== [5/5] pin transformers (in case a dep tried to bump it) ====="
# sglang 0.5.9 pins transformers==4.57.1; enforce it LAST in case fastmcp or any
# requirements.txt dep proposed a newer one.
pip install --no-input "transformers==4.57.1"

echo "===== verify imports ====="
python - <<'PY'
import torch, transformers
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
import sglang; print("sglang      ", sglang.__version__)
import fastmcp; print("fastmcp     ", fastmcp.__version__ if hasattr(fastmcp,'__version__') else 'ok')

# Verify sgl_kernel links correctly (the libnuma.so.1 check)
try:
    import sgl_kernel
    print("sgl_kernel   ok")
except ImportError as e:
    print("sgl_kernel   MISSING:", e)

# Verify key requirements.txt deps
import jsonlines, litellm, openai, pandas
print("jsonlines   ", jsonlines.__version__ if hasattr(jsonlines,'__version__') else 'ok')
print("litellm     ", litellm.__version__)
print("openai      ", openai.__version__)
print("pandas      ", pandas.__version__)
print("ALL IMPORTS OK")
PY

echo "===== runtime cheatsheet (serving) ====="
cat <<'EOF'
source /mnt/beegfs/workspace/scy/activate_conda.sh && conda activate gyz_serve
cd /mnt/beegfs/workspace/scy/yizhigao/LIMA
bash src/eval/serve_envfactory.sh
# Or override MODEL / PORT / GPU_IDS / TP:
#   MODEL=models/your-checkpoint GPU_IDS=2,3 TP=2 PORT=8001 bash src/eval/serve_envfactory.sh
EOF

echo "===== DONE: conda env '$ENV_NAME' is ready ====="
echo "Activate with:  conda activate $ENV_NAME"
