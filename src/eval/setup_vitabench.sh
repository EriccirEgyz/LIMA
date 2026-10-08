#!/bin/bash
# Install VitaBench for EnvFactory evaluation (managed with uv)
#
# Version pinning:
#   - Commit 973756f (2025-10-17), latest stable version as of October 2026.
#     VitaBench is actively developed by Meituan for evaluating LLM agents on
#     versatile interactive tasks in real-world applications (food delivery,
#     in-store consumption, online travel).
#   - The benchmark includes 66 tools across three domains with 100 cross-scenario
#     and 300 single-scenario tasks. Agents interact through multi-turn dialogue.
#
# Runtime notes:
#   - Requires Python 3.10+
#   - Uses uv for dependency management
#   - Configure model settings via models.yaml (see VITA_MODEL_CONFIG_PATH)
set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
VITABENCH_DIR="$EVAL_DIR/vitabench"
PINNED_COMMIT="973756f"

# Redirect caches out of the uid-1001 $HOME (permissions). See cache_env.sh.
# shellcheck disable=SC1091
source "$EVAL_DIR/cache_env.sh"
# Override the clone URL if github over HTTPS is blocked (e.g. behind a firewall
# that terminates TLS on 443). Point it at an SSH remote that works for you:
#   VITABENCH_REPO_URL="git@github.com:meituan/vitabench.git" bash setup_vitabench.sh
VITABENCH_REPO_URL=${VITABENCH_REPO_URL:-"https://github.com/meituan/vitabench.git"}

if [ -d "$VITABENCH_DIR" ] && [ -d "$VITABENCH_DIR/.venv" ]; then
    EXISTING_COMMIT=$(cd "$VITABENCH_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    if [ "$EXISTING_COMMIT" = "$PINNED_COMMIT" ]; then
        echo "VitaBench already installed at correct commit ($PINNED_COMMIT)"
        exit 0
    else
        echo "WARNING: Existing VitaBench checkout at commit $EXISTING_COMMIT does not match pinned $PINNED_COMMIT"
        echo "Removing and recloning..."
        rm -rf "$VITABENCH_DIR"
    fi
fi

if [ ! -d "$VITABENCH_DIR" ]; then
    echo "=== Cloning vitabench ==="
    git clone "$VITABENCH_REPO_URL" "$VITABENCH_DIR"
fi

echo "=== Checking out pinned commit $PINNED_COMMIT ==="
cd "$VITABENCH_DIR"
git checkout -q "$PINNED_COMMIT"

echo "=== Installing vitabench with uv ==="
rm -rf .venv 2>/dev/null || true
uv venv
uv pip install -e .

echo "=== vitabench installed successfully ==="
echo "Virtual environment at: $VITABENCH_DIR/.venv"
echo "Pinned at commit: $(git rev-parse --short HEAD)"
echo ""
echo "Next steps:"
echo "  1. Configure models.yaml with your model endpoints and API keys"
echo "     Example config: src/eval/vitabench_models.yaml.example"
echo "  2. Set VITA_MODEL_CONFIG_PATH if using a custom config location"
echo "     export VITA_MODEL_CONFIG_PATH=/path/to/models.yaml"
echo "  3. Run evaluation: bash src/eval/run_vitabench.sh"
