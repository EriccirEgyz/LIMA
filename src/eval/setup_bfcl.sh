#!/bin/bash
# Install BFCL v3 for EnvFactory evaluation (managed with uv)
#
# Version pinning:
#   - Gorilla commit cd9429c (2025-08-07), which is the last commit before
#     BFCL V4 Release (58f57e9, 2025-08-25). This represents BFCL v3 era.
#   - Rationale: EnvFactory paper (arXiv 2605.18703) evaluates on BFCL v3.
#     v1.3 tag is earlier (2025-07-17) and lacks cd9429c's Qwen3 chat template fix.
#     cd9429c is the best approximation of "main at paper's time".
#   - Data files are identical between v1.3 and cd9429c (46 BFCL_v3_* files).
#
# Runtime notes:
#   - Uses --skip-server-setup to connect to the existing SGLang server from
#     serve_envfactory.sh. No need to install sglang backend here.
set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
GORILLA_DIR="$EVAL_DIR/gorilla"
BFCL_DIR="$GORILLA_DIR/berkeley-function-call-leaderboard"
PINNED_COMMIT="cd9429c"

if [ -d "$BFCL_DIR" ] && [ -d "$BFCL_DIR/.venv" ]; then
    EXISTING_COMMIT=$(cd "$GORILLA_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    if [ "$EXISTING_COMMIT" = "$PINNED_COMMIT" ]; then
        echo "BFCL v3 already installed at correct commit ($PINNED_COMMIT)"
        exit 0
    else
        echo "WARNING: Existing BFCL checkout at commit $EXISTING_COMMIT does not match pinned $PINNED_COMMIT"
        echo "Removing and recloning..."
        rm -rf "$GORILLA_DIR"
    fi
fi

if [ ! -d "$GORILLA_DIR" ]; then
    echo "=== Cloning gorilla repo at BFCL v3 commit ==="
    git clone https://github.com/ShishirPatil/gorilla.git "$GORILLA_DIR"
fi

echo "=== Checking out pinned commit $PINNED_COMMIT ==="
cd "$GORILLA_DIR"
git checkout -q "$PINNED_COMMIT"

echo "=== Setting up BFCL venv with uv ==="
cd "$BFCL_DIR"
rm -rf .venv 2>/dev/null || true
uv venv
uv pip install -e .

# BFCL leaves several real runtime deps undeclared (upstream metadata gaps).
# Install them explicitly so a fresh `uv pip install -e .` is actually usable.
#
# - soundfile: qwen-agent imports it unconditionally at module top level but
#   doesn't list it; drop once qwen-agent fixes its metadata.
# - transformers: the local OSS handler needs AutoTokenizer/AutoConfig to load
#   the tokenizer offline (base_oss_handler.py). Not in bfcl's deps because it
#   would normally arrive transitively via the oss_eval_sglang extra
#   (["sglang[all]"]), which we intentionally don't install (the model is served
#   by the separate sglang server, not loaded here). Note: transformers alone
#   does NOT pull torch, which is what we want.
uv pip install soundfile transformers

echo "=== BFCL v3 installed successfully ==="
echo "Virtual environment at: $BFCL_DIR/.venv"
echo "Pinned at commit: $(cd "$GORILLA_DIR" && git rev-parse --short HEAD)"
