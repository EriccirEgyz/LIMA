#!/usr/bin/env bash
# Terminal-Bench 4.0: independent Harbor environment + Modal sandboxes.
# Official dataset: https://www.tbench.ai/news/terminal-bench-4-0
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
TB_PROJECT="$EVAL_DIR/terminalbench"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash src/eval/setup_terminalbench.sh [--help]
Installs locked Harbor/Modal dependencies in terminalbench/.venv (Python 3.12).
Downloads the pinned dataset into data/eval/terminalbench/4.0.0 by default.

Overrides:
  DOWNLOAD_TASKS=false   Install dependencies only; download at evaluation time.
  TB_DATA_DIR=/path      Override the task cache (also set it when running).

After setup, authenticate Modal using ONE of these methods:
  src/eval/terminalbench/.venv/bin/modal token new
  export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
Then start the model server and use run_terminalbench.sh --check.
Setup does not start sandboxes or launch evaluations.
EOF
    exit 0
fi
[[ $# == 0 ]] || { echo "ERROR: Unknown argument: $1" >&2; exit 2; }
# shellcheck source=cache_env.sh
source "$EVAL_DIR/cache_env.sh"
export LITELLM_LOCAL_MODEL_COST_MAP=True
command -v uv >/dev/null || { echo "ERROR: uv is required." >&2; exit 1; }
TB_DATA_DIR="${TB_DATA_DIR:-$REPO_ROOT/data/eval/terminalbench/4.0.0}"
DOWNLOAD_TASKS="${DOWNLOAD_TASKS:-true}"
[[ "$DOWNLOAD_TASKS" == true || "$DOWNLOAD_TASKS" == false ]] || {
    echo "ERROR: DOWNLOAD_TASKS must be true or false." >&2; exit 2;
}

uv sync --project "$TB_PROJECT" --python 3.12 --locked
"$TB_PROJECT/.venv/bin/python" -c \
    'from importlib.metadata import version; print("Harbor:", version("harbor"), "Modal:", version("modal"))'
if [[ "$DOWNLOAD_TASKS" == true ]]; then
    mkdir -p "$TB_DATA_DIR"
    "$TB_PROJECT/.venv/bin/harbor" download \
        'terminal-bench/terminal-bench@4.0.0' --cache --output-dir "$TB_DATA_DIR"
fi
echo "Setup complete. Authenticate Modal, start SGLang, then run:"
echo "  bash $EVAL_DIR/run_terminalbench.sh --check"
echo "  NUM_TASKS=1 MAX_CONCURRENCY=1 bash $EVAL_DIR/run_terminalbench.sh"
