#!/bin/bash
# Install VitaBench for EnvFactory evaluation (managed with uv)
#
# Version pinning:
#   - Official maintained repository: https://github.com/meituan-longcat/vitabench
#   - Commit 742e240855bf8686a0842360749d5ea970ea3987 (2026-02-23 Beijing time),
#     the main HEAD verified on 2026-10-14. This is a reproducibility pin, not
#     an upstream release tag or a claim of a "latest stable" release.
#   - Replaces meituan/vitabench@973756f. The maintained repository includes
#     corrected tasks/tools and reasoning-message handling; keep old results
#     associated with their original commit when comparing experiments.
#   - The benchmark includes 66 tools across three domains with 100 cross-scenario
#     and 300 single-scenario tasks. Agents interact through multi-turn dialogue.
#
# Runtime notes:
#   - Requires Python 3.10+
#   - Uses uv for dependency management
#   - Configure model settings via models.yaml (see VITA_MODEL_CONFIG_PATH)
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
VITABENCH_DIR="$EVAL_DIR/vitabench"
PINNED_COMMIT="742e240855bf8686a0842360749d5ea970ea3987"

# Redirect caches out of the uid-1001 $HOME (permissions). See cache_env.sh.
# shellcheck disable=SC1091
source "$EVAL_DIR/cache_env.sh"
# Override the clone URL if github over HTTPS is blocked (e.g. behind a firewall
# that terminates TLS on 443). Point it at an SSH remote that works for you:
#   VITABENCH_REPO_URL="git@github.com:meituan-longcat/vitabench.git" bash setup_vitabench.sh
VITABENCH_REPO_URL=${VITABENCH_REPO_URL:-"https://github.com/meituan-longcat/vitabench.git"}
command -v git >/dev/null || { echo "ERROR: git is required." >&2; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv is required." >&2; exit 1; }

if [ ! -e "$VITABENCH_DIR" ]; then
    echo "=== Cloning vitabench ==="
    git clone "$VITABENCH_REPO_URL" "$VITABENCH_DIR"
fi

# Preserve the existing clone, ignored outputs and virtual environment. Refuse
# to switch a checkout containing local edits instead of deleting/recloning it.
if [ ! -e "$VITABENCH_DIR/.git" ]; then
    echo "ERROR: $VITABENCH_DIR exists but is not a Git checkout." >&2
    exit 1
fi
if [ -n "$(git -C "$VITABENCH_DIR" status --porcelain --untracked-files=normal)" ]; then
    echo "ERROR: VitaBench has local changes; preserve or commit them before setup." >&2
    git -C "$VITABENCH_DIR" status --short
    exit 1
fi

git -C "$VITABENCH_DIR" remote set-url origin "$VITABENCH_REPO_URL"
if [ "$(git -C "$VITABENCH_DIR" rev-parse HEAD)" != "$PINNED_COMMIT" ]; then
    git -C "$VITABENCH_DIR" fetch --no-tags origin main
    if ! git -C "$VITABENCH_DIR" cat-file -e "$PINNED_COMMIT^{commit}"; then
        git -C "$VITABENCH_DIR" fetch --no-tags origin "$PINNED_COMMIT"
    fi
fi

echo "=== Checking out pinned commit $PINNED_COMMIT ==="
cd "$VITABENCH_DIR"
git checkout --detach -q "$PINNED_COMMIT"

echo "=== Installing vitabench with uv ==="
if [ ! -d .venv ]; then
    uv venv --python "${VITABENCH_PYTHON:-3.12}" .venv
fi
if [ ! -x .venv/bin/python ]; then
    echo "ERROR: Existing .venv is unusable; rebuild it on this machine." >&2
    exit 1
fi
uv pip install --python .venv/bin/python -e .

echo "=== vitabench installed successfully ==="
echo "Virtual environment at: $VITABENCH_DIR/.venv"
echo "Upstream: $(git remote get-url origin)"
echo "Pinned at commit: $(git rev-parse HEAD)"
echo ""
echo "Next steps:"
echo "  1. Start the agent model server."
echo "  2. Export USER_BASE_URL and USER_API_KEY for the user simulator/evaluator."
echo "  3. Run: bash src/eval/run_vitabench.sh"
echo "     The runner renders vitabench_models.yaml using these environment variables."
