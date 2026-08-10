#!/bin/bash
# Install tau2-bench for EnvFactory evaluation (managed with uv)
#
# Version pinning:
#   - Tag v1.0.0 (2026-03-18) was the latest tau2-bench release at EnvFactory
#     paper submission time (arXiv 2605.18703, May 2026). v1.0.1 (2026-07-16)
#     postdates the paper and only re-grades `banking_knowledge` tasks
#     (README: results from <1.0.1 are not comparable with >=1.0.1, but other
#     domains are unaffected). The paper evaluates on retail/airline/telecom, so
#     v1.0.0 matches the paper exactly. Pin banking_knowledge to v1.0.1 only if
#     you specifically need that domain.
#
# Requires Python 3.12+ and uv.
set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TAU2_DIR="$EVAL_DIR/tau2-bench"
PINNED_TAG="v1.0.1"
# Override the clone URL if github over HTTPS is blocked (e.g. behind a firewall
# that terminates TLS on 443). Point it at an SSH remote that works for you:
#   TAU2_REPO_URL="git@github-gyz:sierra-research/tau2-bench.git" bash setup_tau2_bench.sh
TAU2_REPO_URL=${TAU2_REPO_URL:-"https://github.com/sierra-research/tau2-bench.git"}

if [ ! -d "$TAU2_DIR/.git" ]; then
    echo "=== Cloning tau2-bench ==="
    git clone "$TAU2_REPO_URL" "$TAU2_DIR"
fi

cd "$TAU2_DIR"
echo "=== Checking out pinned tag $PINNED_TAG ==="
git fetch --tags --quiet
git checkout -q "$PINNED_TAG"

echo "=== Installing tau2-bench (uv sync) ==="
uv sync

echo "=== tau2-bench installed successfully ==="
echo "Pinned at: $(git rev-parse --short HEAD) ($(git describe --tags 2>/dev/null || echo $PINNED_TAG))"
echo "Copy .env.example to .env and fill in API keys for the user simulator."
