#!/usr/bin/env bash
# Harbor's Terminus 2 calls the model on this host; Modal executes task commands.
# Therefore AGENT_BASE_URL can point at a localhost SGLang endpoint.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
TB_PROJECT="$EVAL_DIR/terminalbench"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash src/eval/run_terminalbench.sh [--dry-run | --check | --resume JOB_DIR]
Default: Terminal-Bench 4.0, Terminus 2, Modal, all tasks, one attempt/task.

  --dry-run       Validate and print config; no network, sandboxes or run files.
  --check         Check model /models and Modal authentication; no sandboxes.
  --resume DIR    Resume Harbor's saved config (same model/sampling/task settings).

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
  MAX_OUTPUT_TOKENS   Default: 8192 (includes reasoning tokens).
  AGENT_INPUT_PRICE / AGENT_OUTPUT_PRICE  USD per 1M tokens; default 0 (unpriced).

Evaluation:
  NUM_TRIALS          Attempts per task, default 1; use 5 for repeated trials.
  MAX_CONCURRENCY     Default 2; use 1 for an initial smoke run.
  NUM_TASKS           Empty = all; applied after filters.
  TASK_NAMES / EXCLUDE_TASK_NAMES  Comma-separated task names or glob patterns.
  MAX_STEPS           Empty = Terminus default; a cap changes the protocol.
  MAX_RETRIES         Infrastructure retries, default 0; no reward-based retries.
  MODAL_TOKEN_ID / MODAL_TOKEN_SECRET  Or credentials from `modal token new`.
  EXPERIMENT / RUN_TAG  Same output naming convention as other LIMA eval scripts.
  TB_DATA_DIR         Default data/eval/terminalbench/4.0.0 (task cache).
  TB_JOBS_DIR / TB_LOG  Optional output directory and log file overrides.

Examples (run from the LIMA root):
  bash src/eval/run_terminalbench.sh --dry-run
  bash src/eval/run_terminalbench.sh --check
  NUM_TASKS=1 MAX_CONCURRENCY=1 bash src/eval/run_terminalbench.sh
  MODEL=models/spbench/qwen3-1.7b-sft-ep1 SGLANG_PORT=8002 NUM_TRIALS=5 \
    bash src/eval/run_terminalbench.sh
  bash src/eval/run_terminalbench.sh --resume results/<exp>/terminalbench/<ckpt>/<tag>

Results: results/<experiment>/terminalbench/<checkpoint>/<run_tag>/
Logs:    logs/<experiment>/terminalbench_<checkpoint>_<run_tag>.log
View:    src/eval/terminalbench/.venv/bin/harbor view <TB_JOBS_DIR>
Task CPU/memory/GPU/timeouts are inherited; GPU tasks use Modal's GPU runtime.
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
# Use the versioned LiteLLM metadata shipped in the locked environment.
# Otherwise importing Terminus may fetch a moving price/context table.
export LITELLM_LOCAL_MODEL_COST_MAP=True
exec "$TB_PROJECT/.venv/bin/python" "$TB_PROJECT/run.py" "$@"
