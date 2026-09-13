#!/bin/bash
# Redirect every tool cache into this repo. `source` this early in any eval /
# serve / setup script (after REPO_ROOT is known; it self-derives if not).
#
# WHY THIS EXISTS
# activate_conda.sh does `export HOME=/mnt/beegfs/workspace/scy`, and that dir
# plus its .cache/ are owned by uid 1001 with mode drwxr-xr-x, while we run as
# uid 1000. So anything caching under $HOME/.cache hits
#   Permission denied (os error 13)
# Only scy/yizhigao/ was ever chown'd to 1000; scy/ itself is deliberately not
# (it holds other people's work) -- so do NOT "fix" this by chowning scy/.cache.
# Note the trap fires even in scripts that never source activate_conda.sh, since
# HOME is inherited from whatever shell sourced it earlier.
#
# Observed casualties, both of which died before doing any real work:
#   - flashinfer: builds its JIT logger at *import* time, and sglang imports it
#     just to PARSE ARGS (server_args -> model_config -> quantization -> awq ->
#     unquant -> flashinfer), so serve died on
#     .cache/flashinfer/0.6.3/90a/flashinfer_jit.log with a long quantization
#     traceback that looks nothing like a permissions problem.
#   - uv: "Failed to initialize cache at .../.cache/uv" on `uv run`, killing
#     tau2 at the first domain.
#
# Each library reads its OWN env var; XDG_CACHE_HOME is only a partial catch-all
# and does NOT work for flashinfer, which hardcodes .cache/flashinfer beneath
# FLASHINFER_WORKSPACE_BASE (see flashinfer/jit/env.py). Hence the explicit list.
# All values respect a pre-set override, so callers can still point elsewhere.

_CACHE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:="$(cd "$_CACHE_ENV_DIR/../.." && pwd)"}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"   # .gitignore'd

export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT/flashinfer-base}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$CACHE_ROOT/tvm-ffi}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"

mkdir -p "$UV_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer" \
         "$TVM_FFI_CACHE_DIR" "$HF_HOME" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
