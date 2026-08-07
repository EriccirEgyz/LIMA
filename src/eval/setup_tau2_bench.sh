#!/bin/bash
# Install tau2-bench for evaluation
# Requires Python 3.12+ and uv
set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TAU2_DIR="$EVAL_DIR/tau2-bench"

if [ -d "$TAU2_DIR" ]; then
    echo "tau2-bench already cloned at $TAU2_DIR"
    cd "$TAU2_DIR"
    uv sync
    echo "Done."
    exit 0
fi

echo "=== Cloning tau2-bench ==="
git clone https://github.com/sierra-research/tau2-bench "$TAU2_DIR"

echo "=== Installing tau2-bench ==="
cd "$TAU2_DIR"
uv sync

echo "=== tau2-bench installed successfully ==="
echo "Make sure to copy .env.example to .env and fill in API keys."