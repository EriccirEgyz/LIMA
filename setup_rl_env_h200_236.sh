#!/usr/bin/env bash
# EnvFactory/verl RL environment for h200-236 (host currently reports gpu10).
# Keeps the existing RL recipe; does not start training or change other environments.
# Usage: bash setup_rl_env_h200_236.sh [--verify-only | --install-only]
# Activate: source ~/miniconda3/etc/profile.d/conda.sh
#           conda activate /data/home/chengyushen/envs/gyz_rl
# For the eventual training launch, pass this Hydra override explicitly:
#   +ray_kwargs.ray_init._temp_dir=/data/home/chengyushen/.cache/rl-h200-236
# The long home path plus Ray's default nested /ray/session_... path can exceed
# Linux's Unix socket path limit. Verification uses the shorter root above.
# No `set -u`: conda CUDA activation hooks reference optional unset variables.
set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${RL_CONDA_BASE:-/data/home/chengyushen/miniconda3}"
ENV_PREFIX="${RL_ENV_PREFIX:-/data/home/chengyushen/envs/gyz_rl}"
CACHE_ROOT="${RL_CACHE_ROOT:-/data/home/chengyushen/.cache/rl-h200-236}"
LOG_ROOT="$REPO/logs/setup_rl_h200_236"
MODE="${1:-install}"
case "$MODE" in install|--verify-only|--install-only) ;; *) echo "Unknown option: $MODE" >&2; exit 2;; esac
test -f "$REPO/verl/setup.py"
test -x "$CONDA_BASE/bin/conda"
mkdir -p "$CACHE_ROOT" "$LOG_ROOT"
exec 9>"$CACHE_ROOT/setup.lock"
flock -n 9 || { echo 'Another RL setup/verification is running.' >&2; exit 1; }
RUN_DIR="$LOG_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/setup.log") 2>&1
trap 'echo "FAILED at line $LINENO; log: $RUN_DIR/setup.log" >&2' ERR
export REPO ENV_PREFIX CACHE_ROOT RUN_DIR
export TMPDIR="$CACHE_ROOT/tmp" PIP_CACHE_DIR="$CACHE_ROOT/pip" UV_CACHE_DIR="$CACHE_ROOT/uv"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda-pkgs"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch-extensions" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer" RAY_TMPDIR="$CACHE_ROOT/ray"
export PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=200
export MAX_JOBS="${MAX_JOBS:-12}" NVCC_THREADS="${NVCC_THREADS:-2}"
# Future source builds need only Hopper on this machine. Existing compatible
# installations/wheels are reused rather than rebuilt to change architecture coverage.
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
# On this host the inherited localhost proxy fails TLS, while official sources
# are reachable directly. Only change this process; leave shell/global settings alone.
if [[ "${RL_USE_PROXY:-0}" != 1 ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
export CONDA_REPORT_ERRORS=false CONDA_NOTIFY_OUTDATED_CONDA=false
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" \
    "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$RAY_TMPDIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ "$MODE" != --verify-only ]]; then
    echo '[1/6] Python 3.12 and private CUDA 12.8 toolchain'
    if [[ ! -d "$ENV_PREFIX/conda-meta" ]]; then
        [[ ! -e "$ENV_PREFIX" ]] || { echo "Refusing to overwrite $ENV_PREFIX" >&2; exit 1; }
        conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
            python=3.12 pip 'cuda-version=12.8' 'cuda-nvcc=12.8.93' \
            'cuda-cudart-dev=12.8.90' 'cuda-cccl=12.8.90' libnuma
    fi
fi
conda activate "$ENV_PREFIX"
export CUDA_HOME="$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
nvcc --version
nvcc --version | grep -q 'release 12.8'

if [[ "$MODE" != --verify-only ]]; then
    # Constraints apply from the first resolution; unrelated packages cannot upgrade the core stack.
    cat > "$CACHE_ROOT/constraints.txt" <<'CONSTRAINTS'
torch==2.8.0+cu128
torchaudio==2.8.0+cu128
torchvision==0.23.0+cu128
transformers==4.55.4
numpy==1.26.4
scipy==1.15.3
tensordict==0.10.0
torchdata==0.11.0
sglang==0.5.2
sgl-kernel==0.3.9.post2
flashinfer-python==0.3.1
torch-memory-saver==0.0.8
flash-attn==2.7.4.post1
uvicorn==0.40.0
fastmcp==3.1.0
ray==2.56.0
peft==0.17.1
liger-kernel==0.6.2
accelerate==1.10.1
datasets==4.0.0
huggingface-hub==0.34.4
compressed-tensors==0.11.0
cuda-python>=12.8,<13
pyarrow==20.0.0
pandas==2.2.3
CONSTRAINTS
    export PIP_CONSTRAINT="$CACHE_ROOT/constraints.txt"
    echo '[2/6] PyTorch 2.8.0 CUDA 12.8 wheels'
    python -m pip install --no-input 'pip>=25,<27' 'setuptools>=77,<81' wheel packaging ninja
    python -m pip install --no-input --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128'

    echo '[3/6] Explicit SGLang runtime dependencies and verl dependencies'
    python -m pip install --no-input --no-deps 'sglang==0.5.2'
    python - <<'PY'
import importlib.metadata as md
import os
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# Upstream sglang[all] 0.5.2 requires transformers 4.56.1. The existing project
# intentionally overrides it to 4.55.4 (verl's reference image / old setup recipe).
# Expand the selected extras, recording the Transformers override. Omit decord:
# its published 0.6.0 wheel declares cp36, and this is a text/tool-call RL setup.
pending, seen, selected = ['', 'all'], set(), set()
while pending:
    extra = pending.pop()
    if extra in seen:
        continue
    seen.add(extra)
    for raw in md.requires('sglang'):
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({'extra': extra}):
            continue
        if canonicalize_name(req.name) == 'sglang':
            pending.extend(req.extras)
            continue
        req.marker = None
        if canonicalize_name(req.name) == 'decord':
            print('Omitting optional video dependency decord (published wheel has cp36 tag)')
            continue
        if canonicalize_name(req.name) == 'transformers':
            print(f'Compatibility override: {req} -> transformers==4.55.4')
            req = Requirement('transformers==4.55.4')
        selected.add(str(req))
target = Path(os.environ['CACHE_ROOT']) / 'sglang-runtime-requirements.txt'
target.write_text('\n'.join(sorted(selected)) + '\n')
print('Expanded SGLang extras:', sorted(seen), 'requirements:', len(selected))
PY
    # Clean up the optional wheel if resuming an earlier installation attempt.
    if python -c 'import importlib.metadata as m; m.version("decord")' >/dev/null 2>&1; then
        python -m pip uninstall -y decord
    fi
    python -m pip install --no-input --no-build-isolation -r "$CACHE_ROOT/sglang-runtime-requirements.txt" \
        'accelerate==1.10.1' 'datasets==4.0.0' 'peft==0.17.1' 'tensordict==0.10.0' \
        'torchdata==0.11.0' 'numpy==1.26.4' 'scipy==1.15.3' 'ray[default]==2.56.0' \
        'fastmcp==3.1.0' 'uvicorn==0.40.0' 'liger-kernel==0.6.2' \
        hydra-core codetiming dill pybind11 pylatexenc math-verify latex2sympy2-extended \
        'pandas==2.2.3' 'pyarrow==20.0.0' wandb tensorboard jsonlines

    echo '[4/6] Editable local verl (its sglang extra incorrectly selects 0.5.5 for this recipe)'
    python -m pip install --no-input --no-build-isolation --no-deps -e "$REPO/verl"

    echo '[5/6] Build FlashAttention 2.7.4.post1 against this PyTorch/CUDA ABI'
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    python -m pip install --no-input --no-build-isolation --no-binary=flash-attn 'flash-attn==2.7.4.post1'

    echo '[6/6] Per-environment activation settings and reproducibility records'
    python - <<'PY'
import os
import shlex
from pathlib import Path

prefix = Path(os.environ['ENV_PREFIX'])
cache = Path(os.environ['CACHE_ROOT'])
settings = {
    'CUDA_HOME': str(prefix), 'TMPDIR': str(cache / 'tmp'),
    'TORCH_EXTENSIONS_DIR': str(cache / 'torch-extensions'),
    'TRITON_CACHE_DIR': str(cache / 'triton'),
    'FLASHINFER_WORKSPACE_BASE': str(cache / 'flashinfer'),
    'RAY_TMPDIR': str(cache / 'ray'), 'PYTHONDONTWRITEBYTECODE': '1',
}
activate, deactivate = [], []
for key, value in settings.items():
    old = f'_GYZ_RL_OLD_{key}'
    activate += [f'export {old}_SET="${{{key}+yes}}"', f'export {old}="${{{key}-}}"',
                 f'export {key}={shlex.quote(value)}']
    deactivate += [f'if [ "${{{old}_SET-}}" = yes ]; then export {key}="${{{old}-}}"; '
                   f'else unset {key}; fi', f'unset {old} {old}_SET']
activate += ['export _GYZ_RL_OLD_LD_LIBRARY_PATH_SET="${LD_LIBRARY_PATH+yes}"',
             'export _GYZ_RL_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"',
             f'export LD_LIBRARY_PATH={shlex.quote(str(prefix / "lib"))}'
             '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"']
deactivate += ['if [ "${_GYZ_RL_OLD_LD_LIBRARY_PATH_SET-}" = yes ]; then '
               'export LD_LIBRARY_PATH="${_GYZ_RL_OLD_LD_LIBRARY_PATH-}"; '
               'else unset LD_LIBRARY_PATH; fi',
               'unset _GYZ_RL_OLD_LD_LIBRARY_PATH _GYZ_RL_OLD_LD_LIBRARY_PATH_SET']
for kind, lines in [('activate', activate), ('deactivate', deactivate)]:
    dest = prefix / 'etc/conda' / f'{kind}.d' / 'zz-gyz-rl.sh'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('# Managed by setup_rl_env_h200_236.sh\n' + '\n'.join(lines) + '\n')
PY
    cp "$CACHE_ROOT/constraints.txt" "$CACHE_ROOT/sglang-runtime-requirements.txt" "$RUN_DIR/"
fi

python -m pip check
python -m pip freeze --all > "$RUN_DIR/pip-freeze.txt"
conda list --explicit > "$RUN_DIR/conda-explicit.txt"
git -C "$REPO/verl" rev-parse HEAD > "$RUN_DIR/verl-commit.txt"

if [[ "$MODE" != --install-only ]]; then
    echo 'Verifying imports, CUDA/FlashAttention, local Ray and MCP'
    # Verification is deliberately small; no model download or RL training.
    export RAY_USAGE_STATS_ENABLED=0 WANDB_MODE=offline
    export PYTHONPATH="$REPO/verl${PYTHONPATH:+:$PYTHONPATH}"
    python - <<'PY'
"""Small, local checks for the existing EnvFactory/verl RL stack; no training."""
import asyncio
from contextlib import suppress
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

from packaging.requirements import Requirement

repo = Path(os.environ['REPO'])
cache = Path(os.environ['CACHE_ROOT'])
records = Path(os.environ['RUN_DIR'])
assert sys.version_info[:2] == (3, 12), sys.version
for filename in ('constraints.txt', 'sglang-runtime-requirements.txt'):
    for line in (cache / filename).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        req = Requirement(line)
        assert md.version(req.name) in req.specifier, (req, md.version(req.name))
print('All core constraints and effective SGLang runtime requirements match', flush=True)

versions = {name: md.version(name) for name in [
    'torch', 'transformers', 'sglang', 'sgl-kernel', 'flashinfer-python',
    'flash-attn', 'ray', 'fastmcp', 'mcp', 'uvicorn', 'numpy', 'scipy', 'tensordict']}
print(json.dumps(versions, indent=2), flush=True)
for module in [
    'sgl_kernel', 'flashinfer', 'fastmcp', 'verl.trainer.main_ppo',
    'verl.workers.fsdp_workers', 'verl.experimental.agent_loop.tool_agent_loop',
    'verl.workers.rollout.sglang_rollout.async_sglang_server',
    'sglang.srt.models.qwen3', 'EnvFactory.reward.tool_reward_fcn',
]:
    importlib.import_module(module)
    print('IMPORT OK:', module, flush=True)

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
import sgl_kernel
import flashinfer

assert torch.version.cuda == '12.8'
assert torch.cuda.is_available(), 'GPU validation requires access to /dev/nvidia*'
device = torch.device('cuda:0')
torch.manual_seed(42)
q, k, v = [torch.randn(1, 32, 4, 64, device=device, dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3)]
actual = flash_attn_func(q, k, v, causal=True)
expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), is_causal=True).transpose(1, 2)
torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
actual.float().square().mean().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
x = torch.randn(8, 128, device=device, dtype=torch.bfloat16)
weight = torch.ones(128, device=device, dtype=torch.bfloat16)
ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
torch.testing.assert_close(sgl_kernel.rmsnorm(x, weight, 1e-6), ref, atol=0.02, rtol=0.02)
torch.testing.assert_close(flashinfer.norm.rmsnorm(x, weight, 1e-6), ref, atol=0.02, rtol=0.02)
torch.cuda.synchronize()
print('CUDA / FlashAttention forward+backward / SGL kernel / FlashInfer JIT OK:',
      torch.cuda.get_device_name(device), flush=True)

async def check_http():
    from fastapi import FastAPI
    import httpx
    from verl.workers.rollout import utils

    app = FastAPI()

    @app.get('/health')
    def health():
        return {'ok': True}

    # Capture the instance only to close our own listening sockets after checking
    # the unmodified verl helper's unusual should_exit/startup semantics.
    original = utils.uvicorn.Server
    created = []

    class Server(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    utils.uvicorn.Server = Server
    task = None
    try:
        port, task = await utils.run_unvicorn(app, None, '127.0.0.1')
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            for _ in range(2):
                r = await client.get(f'http://127.0.0.1:{port}/health')
                assert r.status_code == 200 and r.json() == {'ok': True}
        print('verl run_unvicorn / uvicorn 0.40.0: HTTP health twice OK', flush=True)
    finally:
        utils.uvicorn.Server = original
        for server in created:
            if server.started:
                await server.shutdown()
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

asyncio.run(check_http())

import ray
from EnvFactory.manager.mcp_manager import MCPManagerActor

config_path = records / 'mcp-smoke-config.json'
config_path.write_text(json.dumps({'mcpServers': {'Calendar': {
    'tool_path': str(repo / 'verl/EnvFactory/tools/Calendar.py'), 'stateless': False,
}}}))
# Keep this path short enough for Linux's 107-byte Unix socket path limit.
ray.init(address='local', num_cpus=2, num_gpus=0, include_dashboard=False,
         object_store_memory=100 * 1024 * 1024, _node_ip_address='127.0.0.1',
         _temp_dir=str(cache), log_to_driver=False)
actor = None
try:
    actor = MCPManagerActor.remote()
    ray.get(actor.init_config.remote(str(config_path)), timeout=90)
    schemas = ray.get(actor.filter_tools.remote(['Calendar']), timeout=30)
    assert any(s['function']['name'] == 'Calendar-create_event' for s in schemas)

    def call(client, tool, args):
        return ray.get(actor.call_tool.remote(client, f'Calendar-{tool}', args), timeout=60)

    a, b = 'Calendar-install-smoke-a', 'Calendar-install-smoke-b'
    assert 'Successfully' in call(a, 'load_scenario', {'scenario': {}})
    assert 'Successfully' in call(b, 'load_scenario', {'scenario': {}})
    event = json.loads(call(a, 'create_event', {
        'summary': 'installation smoke test', 'start_time': '2026-11-24T10:00:00Z',
        'end_time': '2026-11-24T11:00:00Z',
    }))
    updated = json.loads(call(a, 'update_event', {
        'event_id': event['event_id'], 'summary': 'updated by smoke test',
    }))
    assert updated['summary'] == 'updated by smoke test'
    state_a = json.loads(call(a, 'save_scenario', {}))
    state_b = json.loads(call(b, 'save_scenario', {}))
    assert state_a['events'][event['event_id']]['summary'] == 'updated by smoke test'
    assert event['event_id'] not in state_b['events']
    print('Actual Ray MCPManagerActor / Calendar subprocess / create-update / session isolation OK',
          flush=True)
finally:
    if actor:
        with suppress(Exception):
            ray.get(actor.shutdown.remote(), timeout=60)
        ray.kill(actor)
    ray.shutdown()

(records / 'verified-versions.json').write_text(json.dumps(versions, indent=2) + '\n')
print('ALL CHECKS PASSED (formal RL training was not started)', flush=True)
PY
fi
echo "Completed. Records: $RUN_DIR"
echo "Activate: conda activate $ENV_PREFIX"
echo "Training Ray override: +ray_kwargs.ray_init._temp_dir=$CACHE_ROOT"
