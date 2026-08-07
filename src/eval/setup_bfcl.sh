#!/bin/bash
# Install BFCL v3 for evaluation (managed with uv)
#
# Uses --skip-server-setup at runtime to connect to the existing SGLang server
# started by serve_envfactory.sh. No need to install sglang backend here.
set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
GORILLA_DIR="$EVAL_DIR/gorilla"
BFCL_DIR="$GORILLA_DIR/berkeley-function-call-leaderboard"

if [ -d "$BFCL_DIR" ] && [ -d "$BFCL_DIR/.venv" ]; then
    echo "BFCL already installed at $BFCL_DIR"
    exit 0
fi

if [ ! -d "$GORILLA_DIR" ]; then
    echo "=== Cloning gorilla repo ==="
    git clone https://github.com/ShishirPatil/gorilla.git "$GORILLA_DIR"
fi

echo "=== Setting up BFCL with uv ==="
cd "$BFCL_DIR"
uv venv
uv pip install -e .

echo "=== BFCL v3 installed successfully ==="
echo "Virtual environment at: $BFCL_DIR/.venv"
