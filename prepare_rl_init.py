#!/usr/bin/env python
"""Build an RL-ready init directory: SFT weights + canonical base tokenizer.

Why this exists
---------------
verl 0.6.1 runs transformers 4.55.4 (its official Docker pin). The SFT checkpoints were
saved by the SFT env's transformers 5.8.0, which writes tokenizer artifacts that 4.55.4
cannot load (e.g. an `extra_special_tokens` LIST that 4.55.4's loader calls .keys() on ->
AttributeError). This version skew will recur for every new SFT checkpoint.

Since full fine-tuning does NOT change the tokenizer (vocab / special tokens / chat
template are identical to the base model), RL should always pair the SFT WEIGHTS with the
canonical BASE-MODEL tokenizer. This script assembles exactly that, once per SFT ckpt:

    <out>/model.safetensors ...   -> SYMLINK to <sft_ckpt>      (the trained weights)
    <out>/config.json             -> SYMLINK to <sft_ckpt>      (matches the weights)
    <out>/generation_config.json  -> SYMLINK to <sft_ckpt>
    <out>/tokenizer*.{json,...}   <-  COPY from <base_model>     (canonical, 4.55.4-loadable)

Run with the RL env so the verification step uses transformers 4.55.4:
    conda activate envfactory-rl
    python prepare_rl_init.py \
        --sft-ckpt LLaMA-Factory/models/qwen3-1.7b/env_factory_sft/checkpoint-104 \
        --base-model models/Qwen3-1.7B \
        --out models/rl_init/qwen3-1.7b-sft-ep1
"""
import argparse
import glob
import os
import shutil
import sys

# weight / model-config files: take from the SFT checkpoint (must match the trained weights)
WEIGHT_GLOBS = ["model.safetensors", "model-*.safetensors", "model.safetensors.index.json"]
SYMLINK_FROM_CKPT = ["config.json", "generation_config.json"]
# tokenizer files: take the canonical ones from the base model (4.55.4-loadable)
TOKENIZER_FILES = [
    "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "tokenizer.model",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-ckpt", required=True, help="SFT checkpoint dir (has model.safetensors)")
    ap.add_argument("--base-model", required=True, help="original base model dir (canonical tokenizer)")
    ap.add_argument("--out", required=True, help="output RL init dir")
    ap.add_argument("--force", action="store_true", help="overwrite if <out> exists")
    args = ap.parse_args()

    sft = os.path.abspath(args.sft_ckpt)
    base = os.path.abspath(args.base_model)
    out = os.path.abspath(args.out)
    for d, name in [(sft, "sft-ckpt"), (base, "base-model")]:
        if not os.path.isdir(d):
            sys.exit(f"ERROR: --{name} not a directory: {d}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        if args.force:
            shutil.rmtree(out)
        else:
            sys.exit(f"ERROR: {out} exists; rerun with --force to overwrite")
    os.makedirs(out)

    # 1) symlink trained weights + matching model config from the SFT checkpoint
    linked = []
    for pat in WEIGHT_GLOBS:
        for f in sorted(glob.glob(os.path.join(sft, pat))):
            os.symlink(f, os.path.join(out, os.path.basename(f)))
            linked.append(os.path.basename(f))
    for f in SYMLINK_FROM_CKPT:
        src = os.path.join(sft, f)
        if os.path.exists(src):
            os.symlink(src, os.path.join(out, f))
            linked.append(f)
    if not glob.glob(os.path.join(out, "model*.safetensors")):
        sys.exit("ERROR: no model*.safetensors found in the SFT checkpoint")
    print(f"[1/3] symlinked {len(linked)} weight/config file(s) from SFT ckpt:")
    for f in linked:
        print(f"        - {f}")

    # 2) copy canonical tokenizer files from the base model
    copied = []
    for f in TOKENIZER_FILES:
        src = os.path.join(base, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, f))
            copied.append(f)
    if not os.path.exists(os.path.join(out, "tokenizer_config.json")):
        sys.exit("ERROR: base model has no tokenizer_config.json")
    print(f"[2/3] copied {len(copied)} tokenizer file(s) from base model: {copied}")

    # 3) verify loadability with the INSTALLED transformers (run this under the RL env)
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    print(f"[3/3] OK: vocab_size={tok.vocab_size} | <|im_start|> id={im_start} | arch={cfg.architectures}")
    print(f"\nRL init dir ready: {out}")
    print("Set MODEL_PATH to this path in the launch script (and keep checkpoint untouched).")


if __name__ == "__main__":
    main()
