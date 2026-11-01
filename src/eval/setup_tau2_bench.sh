#!/bin/bash
# Install tau2-bench for EnvFactory evaluation (managed with uv)
#
# Version pinning:
#   - Keep v1.0.1 for the 1.x text benchmark. Upstream explicitly recommends
#     this version for comparison with 1.x results; v2.0 scores are not comparable.
#   - Retail/airline/telecom task data are unchanged from v1.0.0, but the
#     framework also changed. This pin alone does not reproduce every setting
#     used in the EnvFactory paper.
#
# Requires Python >=3.12,<3.14 and uv. Default interpreter: Python 3.12.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TAU2_DIR="$EVAL_DIR/tau2-bench"
PINNED_TAG="v1.0.1"
TAU2_PYTHON=${TAU2_PYTHON:-3.12}

command -v git >/dev/null || { echo "ERROR: git is required." >&2; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv is required." >&2; exit 1; }
# Use the same project cache locations as run_tau2_bench.sh.
# shellcheck disable=SC1091
source "$EVAL_DIR/cache_env.sh"

# Override the clone URL if github over HTTPS is blocked (e.g. behind a firewall
# that terminates TLS on 443). Point it at an SSH remote that works for you:
#   TAU2_REPO_URL="git@github-gyz:sierra-research/tau2-bench.git" bash setup_tau2_bench.sh
TAU2_REPO_URL=${TAU2_REPO_URL:-"https://github.com/sierra-research/tau2-bench.git"}

if [ ! -e "$TAU2_DIR" ]; then
    echo "=== Cloning tau2-bench ==="
    git clone --branch "$PINNED_TAG" --single-branch "$TAU2_REPO_URL" "$TAU2_DIR"
fi
if [ ! -e "$TAU2_DIR/.git" ]; then
    echo "ERROR: $TAU2_DIR exists but is not a Git checkout." >&2
    exit 1
fi

cd "$TAU2_DIR"
echo "=== Checking out pinned tag $PINNED_TAG ==="
if ! git rev-parse --verify --quiet "refs/tags/$PINNED_TAG^{commit}" >/dev/null; then
    git fetch --quiet "$TAU2_REPO_URL" "refs/tags/$PINNED_TAG:refs/tags/$PINNED_TAG"
fi
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "refs/tags/$PINNED_TAG^{commit}")" ]; then
    if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
        echo "ERROR: tau2-bench has local changes; refusing to switch versions." >&2
        exit 1
    fi
    git checkout --detach -q "$PINNED_TAG"
fi

echo "=== Installing tau2-bench (uv sync) ==="
uv sync --python "$TAU2_PYTHON"

echo "=== tau2-bench installed successfully ==="
echo "Pinned at: $(git rev-parse --short HEAD) ($(git describe --tags 2>/dev/null || echo $PINNED_TAG))"
echo "Python: $(.venv/bin/python --version)"
echo "Export USER_BASE_URL and USER_API_KEY for the user simulator/evaluator."
echo "Run with MODEL and SGLANG_PORT matching the served weights:"
echo "  bash src/eval/run_tau2_bench.sh"
