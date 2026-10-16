#!/usr/bin/env bash
# Harbor's Terminus 2 calls the model on this host; local Docker executes tasks.
# Therefore AGENT_BASE_URL can point at a localhost SGLang endpoint.
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

Model (OpenAI-compatible Chat Completions API):
  MODEL               Weights path; used for experiment and checkpoint naming.
                      Default: models/envfactory_baseline/EnvFactory-1.7B.
  SGLANG_PORT         Default: 8000.
  AGENT_BASE_URL      Default: http://localhost:$SGLANG_PORT/v1.
  AGENT_MODEL         Request model ID; auto-detected from /models if unset.
  AGENT_API_KEY       Falls back to OPENAI_API_KEY; placeholder for localhost.
  ENABLE_THINK        true/false/auto; default true for local API, auto for remote.
                      auto leaves the server's chat template setting unchanged.
  AGENT_TEMPERATURE   Default 0.7 with thinking, 0 without; unset in auto mode.
  AGENT_CONTEXT_LENGTH  Served context limit. Defaults to MODEL/config.json's
                      max_position_embeddings, otherwise 40960. Override if the
                      server uses a different context limit or for remote models.
  MAX_OUTPUT_TOKENS   Optional output cap (includes reasoning tokens). Unset/empty
                      leaves the output budget to the model server.
  AGENT_SUMMARIZATION_THRESHOLD  Free-context tokens below which Terminus
                      summarizes history; default 8000, independent of output cap.
                      Set 0 to disable proactive summarization (passive stays on).
  AGENT_INPUT_PRICE / AGENT_OUTPUT_PRICE  USD per 1M tokens; default 0 (unpriced).

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
  MAX_STEPS           Empty = Terminus default; a cap changes the protocol.
  MAX_RETRIES         Infrastructure retries, default 0; no reward-based retries.
  MODAL_TOKEN_ID / MODAL_TOKEN_SECRET  Only for the optional Modal backend.
  EXPERIMENT / RUN_TAG  Same output naming convention as other LIMA eval scripts.
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

Results: results/<experiment>/terminalbench/<checkpoint>/<run_tag>/
Logs:    logs/<experiment>/terminalbench_<checkpoint>_<run_tag>.log
View:    src/eval/terminalbench/.venv/bin/harbor view <TB_JOBS_DIR>
Task CPU/memory/GPU-count/timeouts are inherited. Docker uses the selected local
GPU, ignoring gpu_types (e.g. H100); record the actual hardware in comparisons.
--check starts no containers and cannot prove that GPU execution/scoring works.
--resume keeps the saved backend; old Modal jobs cannot be resumed as Docker.
Start a new run for Docker; explicitly set TB_ENVIRONMENT=modal to resume Modal.
Filtered/capped runs must be reported as such. Credentials are never put in config.
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
exec "$TB_PROJECT/.venv/bin/python" "$TB_PROJECT/run.py" "$@"
