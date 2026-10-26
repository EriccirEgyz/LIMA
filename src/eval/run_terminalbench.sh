#!/usr/bin/env bash
# Terminus 2 calls the model on the host; Claude Code calls it inside the task.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
TB_PROJECT="$EVAL_DIR/terminalbench"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash src/eval/run_terminalbench.sh [--dry-run | --check | --resume JOB_DIR]
Default: Terminal-Bench 4.0, Terminus 2, local Docker, all tasks, five attempts/task.

  --dry-run       Validate and print config (model/sampling, tasks, output paths).
                  Offline; no model calls, sandboxes or evaluation result/log files.
  --check         Validate config, query model /models and verify the model ID;
                  check Docker daemon/Compose and NVIDIA runtime for GPU tasks.
                  For Claude's private Docker route, briefly check a local relay.
                  With TB_ENVIRONMENT=modal, check Modal credentials/connectivity.
                  No model generation, task downloads, sandboxes or evaluations.
                  Does not verify task execution, sandbox capacity or dataset access.
  --resume DIR    Resume Harbor's saved config (same model/sampling/task settings).

Preparation: verify `docker info` and `docker compose version`.
If you were recently added to the docker group, log in again or use `newgrp docker`.
GPU tasks also require NVIDIA Container Toolkit configured in the Docker daemon.
Select one dedicated task GPU with TB_GPU_DEVICE; this is independent of the
model server's CUDA_VISIBLE_DEVICES. Both task and separate verifier use that GPU.
After configuring the model endpoint, run --dry-run, then --check.
Run without either flag to launch tasks (consumes model and local resources).
Both check modes may create tool-cache directories via cache_env.sh.
Optional cloud backend: TB_ENVIRONMENT=modal; authenticate with
  src/eval/terminalbench/.venv/bin/modal token new
Modal execution consumes billable cloud resources.

Shared model settings:
  TB_AGENT            terminus-2 (default) or claude-code (official Harbor agent).
  MODEL               Weights path; used for experiment and checkpoint naming.
                      Default: models/envfactory_baseline/EnvFactory-1.7B.
  SGLANG_PORT         Default: 8000.
  AGENT_BASE_URL      Default: http://localhost:$SGLANG_PORT/v1.
  AGENT_MODEL         Request model ID; auto-detected from /models if unset.
  AGENT_API_KEY       Terminus: falls back to OPENAI_API_KEY. Claude: remote API
                      falls back to ANTHROPIC_API_KEY, then OPENAI_API_KEY;
                      localhost uses a placeholder unless AGENT_API_KEY is set.
  AGENT_CONTEXT_LENGTH  Served context limit. Claude defaults to 40960; Terminus
                      uses MODEL/config.json's max_position_embeddings or 40960.
  MAX_OUTPUT_TOKENS   Claude defaults to 4096; Terminus leaves the output budget
                      to the server unless set. Includes reasoning tokens.

Terminus 2 (OpenAI-compatible Chat Completions API):
  ENABLE_THINK        true/false/auto; default true for local API, auto for remote.
                      auto leaves the server's chat template setting unchanged.
  AGENT_TEMPERATURE   Default 0.7 with thinking, 0 without; unset in auto mode.
  AGENT_SUMMARIZATION_THRESHOLD  Free-context tokens below which Terminus
                      summarizes history; default 8000, independent of output cap.
                      Set 0 to disable proactive summarization (passive stays on).
  AGENT_INPUT_PRICE / AGENT_OUTPUT_PRICE  USD per 1M tokens; default 0 (unpriced).

Claude Code (Anthropic Messages API; requires server support):
  TB_CLAUDE_VERSION   Exact CLI version; default 2.1.80. Other versions need
                      separate compatibility testing with the model server.
  TB_CLAUDE_PREINSTALL  Default 1: new 2.1.80 jobs on this Linux x86_64 host's
                      private Docker use the verified cached official binary at
                      .cache/terminalbench-claude-verify-20261026/claude-2.1.80-linux-x64.
                      Its single-file bind is read-only and agent-only; SHA256,
                      ELF architecture and executable permission are checked.
                      Set 0 for a new job to use Harbor's normal installation.
                      Other versions, remote Docker and Modal do not auto-mount.
                      Resume verifies its saved mount/identity; this flag never
                      adds or removes mounts from an existing job.
  TB_CLAUDE_BASE_URL  Optional container-reachable API origin, WITHOUT /v1.
                      AGENT_BASE_URL remains the host /v1 endpoint for discovery.
                      With private DOCKER_HOST=unix:///run/lima-docker/docker.sock
                      and unset DOCKER_CONTEXT, a local HTTP endpoint is relayed
                      through 10.255.254.1 automatically. Other Docker/Modal
                      deployments need an explicit reachable origin for localhost.
  TB_CLAUDE_RELAY_PORT  Default 18004. Bound only while the launcher runs/checks;
                      an occupied port fails. Resume reuses the saved port/route.
  TB_CONTAINER_PROXY  Installer proxy; defaults to http://10.255.254.1:17890 for
                      private Docker, unset elsewhere. Explicit empty disables it.
  TB_CONTAINER_NO_PROXY  Private default: localhost,127.0.0.1,::1,10.255.254.1,
                      deb.debian.org. Adjust for other task package repositories.
  The saved agent env sets CLAUDE_CODE_AUTO_COMPACT_WINDOW and
  CLAUDE_CODE_MAX_OUTPUT_TOKENS from the shared settings above. No percentage
  override, custom prompt/tools, sampling/thinking override or setup-timeout
  override is added. 4096 is this experiment's cap, NOT Claude Code's default.
  Terminus-only flags do not apply. Inherited host CLAUDE_*/ANTHROPIC_* settings
  are cleared before Harbor starts; the saved config controls new runs/resume.
  Native contextWindow metadata may still say 200000; the auto-compact window
  is a separate setting. This does not change the model server's context limit.
  Default setup remains Harbor's 360 seconds; prior cold installs exceeded this.
  A matching preinstalled version skips installation AND dependency provisioning.
  --check verifies the daemon's Linux/x86_64 identity for preinstalled jobs;
  task-image ABI compatibility has only been exercised on the CAD image.
  --check does not validate installation or the Messages/compaction flow.

Evaluation:
  TB_ENVIRONMENT      docker (default) or modal.
  TB_GPU_DEVICE       One dedicated task GPU index/UUID, e.g. 4. Required for
                      local GPU tasks. No automatic selection of model GPUs.
  NUM_TRIALS          Attempts per task, default 5.
  MAX_CONCURRENCY     Default 1. Must be 1 when TB_GPU_DEVICE is set to prevent
                      separate trials from sharing that GPU. CPU-only jobs can
                      unset TB_GPU_DEVICE and use higher concurrency.
  NUM_TASKS           Empty = all; applied after filters.
  TASK_NAMES / EXCLUDE_TASK_NAMES  Comma-separated task names or glob patterns.
  MAX_STEPS           Empty = native agent default; a cap changes the protocol.
  MAX_RETRIES         Infrastructure retries, default 0; no reward-based retries.
  MODAL_TOKEN_ID / MODAL_TOKEN_SECRET  Only for the optional Modal backend.
  EXPERIMENT / RUN_TAG  Same output naming convention as other LIMA eval scripts.
                      Claude's automatic tag includes claude-code-<version>.
  TB_DATA_DIR         Default data/eval/terminalbench/4.0.0 (task cache).
  TB_JOBS_DIR / TB_LOG  Optional output directory and log file overrides.

Examples (run from the LIMA root):
  TB_GPU_DEVICE=4 bash src/eval/run_terminalbench.sh --dry-run
  TB_GPU_DEVICE=4 MAX_CONCURRENCY=1 bash src/eval/run_terminalbench.sh --check
  TASK_NAMES=math-eval-grader TB_GPU_DEVICE=4 NUM_TRIALS=1 MAX_CONCURRENCY=1 \
    bash src/eval/run_terminalbench.sh
  MODEL=models/spbench/qwen3-1.7b-sft-ep1 SGLANG_PORT=8002 NUM_TRIALS=5 TB_GPU_DEVICE=4 MAX_CONCURRENCY=1 \
    bash src/eval/run_terminalbench.sh
  bash src/eval/run_terminalbench.sh --resume results/<exp>/terminalbench/<ckpt>/<tag>
  # Claude preflight for the CPU subset on this machine (no model generation):
  env -u DOCKER_CONTEXT -u TB_GPU_DEVICE \
    DOCKER_HOST=unix:///run/lima-docker/docker.sock TB_AGENT=claude-code \
    MODEL=models/Qwen3-8B SGLANG_PORT=8004 \
    EXCLUDE_TASK_NAMES=fp8-rmsnorm-gemm,math-eval-grader,jax-speedrun-gpu \
    bash src/eval/run_terminalbench.sh --check

Results: results/<experiment>/terminalbench/<checkpoint>/<run_tag>/
Logs:    logs/<experiment>/terminalbench_<checkpoint>_<run_tag>.log
View:    src/eval/terminalbench/.venv/bin/harbor view <TB_JOBS_DIR>
Task CPU/memory/GPU-count/timeouts are inherited. Docker uses the selected local
GPU, ignoring gpu_types (e.g. H100); record the actual hardware in comparisons.
--check starts no containers and cannot prove that GPU execution/scoring works.
--resume keeps the saved backend; old Modal jobs cannot be resumed as Docker.
Start a new run for Docker; explicitly set TB_ENVIRONMENT=modal to resume Modal.
Filtered/capped runs must be reported as such (CPU subset: 63 of 66 tasks).
For papers, record CLI/Harbor versions, both budgets, server sampling/thinking,
task selection and hardware. Claude and Terminus are separate harness conditions.
Credentials use runtime env references, never literal keys in config.
EOF
    exit 0
fi
# shellcheck source=cache_env.sh
source "$EVAL_DIR/cache_env.sh"
[[ -x "$TB_PROJECT/.venv/bin/python" ]] || {
    echo "ERROR: Run bash $EVAL_DIR/setup_terminalbench.sh first." >&2; exit 1;
}
export REPO_ROOT
# Harbor imports the local GPU adapter in run.py for both agent and verifier.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Use the versioned LiteLLM metadata shipped in the locked environment.
# Otherwise importing Terminus may fetch a moving price/context table.
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONDONTWRITEBYTECODE=1
exec "$TB_PROJECT/.venv/bin/python" -B "$TB_PROJECT/run.py" "$@"
