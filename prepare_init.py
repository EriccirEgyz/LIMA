#!/usr/bin/env python
"""Build an RL/inference-ready init directory: SFT weights + version-neutral config/tokenizer.

Why this exists
---------------
Transformers version skew between training and consumer environments:
- SFT training env (LLaMA-Factory): transformers 5.8.0
- RL env (verl 0.6.1):              transformers 4.55.4 (official Docker pin)
- Inference env (envfactory):       transformers 4.57.1

Two independent incompatibilities in what transformers 5.8.0 writes:

1) config.json -- SILENT AND DANGEROUS.
   5.x nests RoPE settings and renames the dtype key:
       "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}
       "dtype": "bfloat16"
   and drops the top-level "rope_theta" / "torch_dtype" that 4.x reads. A 4.x
   loader therefore falls back to Qwen3Config defaults:
       rope_theta  -> 10000.0   (100x too small; long-context output degenerates)
       torch_dtype -> None      (loads in fp32; 2x memory)
   `from_pretrained` SUCCEEDS, so this never raises -- it just produces a
   quietly broken model. sglang reads it via
   `getattr(config, "rope_theta", 1000000)`, so it picks up the bad 10000.0.
   Measured on this repo's 5.8.0-saved Qwen3-1.7B SFT ckpt served through
   sglang (temp 0, thinking off): coherent at 376/1056/2076/3436 prompt tokens,
   degenerates into token repetition from ~5136 tokens up.

   Fix: emit a MERGED config carrying BOTH spellings, so every version reads the
   correct value without relying on a compatibility shim. Verified to resolve
   rope_theta=1000000 and bfloat16 under 4.55.4 / 4.57.1 / 5.8.0.

   Do NOT "fix" this by re-saving through transformers 4.x: that loads the bad
   10000.0 and then bakes it into the file permanently. Verified.

2) tokenizer files -- loud, fails fast.
   5.x writes `extra_special_tokens` as a LIST; transformers 4.x calls .keys()
   on it -> AttributeError. Full fine-tuning does not change the tokenizer, so
   we take the canonical base-model tokenizer instead. (The two tokenizer.json
   files are not byte-identical -- they differ in ByteLevel `trim_offsets` /
   `add_prefix_space` / `use_regex` -- but were verified to produce IDENTICAL
   encode and decode results, including per-token streaming decode.)

Layout produced:
    <out>/model*.safetensors      -> link to <sft_ckpt>   (trained weights)
    <out>/config.json             <- MERGED (ckpt values + 4.x-readable keys)
    <out>/generation_config.json  <- MERGED (ckpt values + base bos_token_id)
    <out>/tokenizer*.{json,...}   <- COPY from <base_model>  (4.x-loadable)

Usage:
    # RL init (verify under transformers 4.55.4)
    conda activate envfactory-rl
    python prepare_init.py \
        --sft-ckpt LLaMA-Factory/models/qwen3-1.7b/env_factory_sft/checkpoint-104 \
        --base-model models/Qwen3-1.7B \
        --out models/qwen3-1.7b-sft-ep1 --force

    # Inference / evaluation (verify under transformers 4.57.1)
    uv run python prepare_init.py \
        --sft-ckpt LLaMA-Factory/models/qwen3-1.7b/env_factory_sft/checkpoint-208 \
        --base-model models/Qwen3-1.7B \
        --out models/qwen3-1.7b-sft-ep2 --force
"""

import argparse
import glob
import json
import os
import shutil
import sys

# weight files: must come from the SFT checkpoint (these are the trained params)
WEIGHT_GLOBS = ["model.safetensors", "model-*.safetensors", "model.safetensors.index.json"]
# tokenizer files: take the canonical ones from the base model (4.x-loadable)
TOKENIZER_FILES = [
    "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "tokenizer.model",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def build_config(ckpt_cfg, base_cfg):
    """Merge the ckpt config with the legacy key spellings transformers 4.x reads.

    The checkpoint config is authoritative -- it describes the actual weights.
    We only ADD older spellings of keys that 5.x renamed or nested, so 4.x and
    5.x resolve identical values. Nothing from the ckpt is dropped or altered,
    except use_cache (training-only setting, see below).
    """
    cfg = dict(ckpt_cfg)
    notes = []

    # RoPE: 5.x nests this under rope_parameters and omits the top-level key.
    rope_params = cfg.get("rope_parameters")
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        theta = rope_params["rope_theta"]
        if cfg.get("rope_theta") != theta:
            cfg["rope_theta"] = theta
            notes.append(f"rope_theta={theta} (lifted from rope_parameters)")
        rope_type = rope_params.get("rope_type", "default")
        if rope_type == "default":
            cfg.setdefault("rope_scaling", None)
        elif "rope_scaling" not in cfg:
            cfg["rope_scaling"] = dict(rope_params)
            notes.append(f"rope_scaling from rope_parameters (rope_type={rope_type})")
    elif "rope_theta" not in cfg:
        # No RoPE info anywhere in the ckpt: fall back to the base model.
        if "rope_theta" in base_cfg:
            cfg["rope_theta"] = base_cfg["rope_theta"]
            notes.append(f"rope_theta={base_cfg['rope_theta']} (from base model)")

    # dtype: 5.x renamed torch_dtype -> dtype. Keep both spellings.
    dtype = cfg.get("dtype") or cfg.get("torch_dtype") or base_cfg.get("torch_dtype")
    if dtype is not None:
        if cfg.get("torch_dtype") != dtype:
            cfg["torch_dtype"] = dtype
            notes.append(f"torch_dtype={dtype}")
        cfg.setdefault("dtype", dtype)

    # bos_token_id: present in base and in the official release, dropped by 5.8.0.
    if cfg.get("bos_token_id") is None and base_cfg.get("bos_token_id") is not None:
        cfg["bos_token_id"] = base_cfg["bos_token_id"]
        notes.append(f"bos_token_id={base_cfg['bos_token_id']} (from base model)")

    # Training checkpoints save use_cache=False (gradient checkpointing).
    # Inference and rollout both want the KV cache enabled.
    if cfg.get("use_cache") is not True:
        cfg["use_cache"] = True
        notes.append("use_cache=True (was False for training)")

    return cfg, notes


def build_generation_config(ckpt_gen, base_gen):
    """Carry over generation defaults, restoring keys 5.8.0 dropped."""
    gen = dict(ckpt_gen)
    notes = []
    if gen.get("bos_token_id") is None and base_gen.get("bos_token_id") is not None:
        gen["bos_token_id"] = base_gen["bos_token_id"]
        notes.append(f"bos_token_id={base_gen['bos_token_id']} (from base model)")
    return gen, notes


def guard_output_dir(out, force):
    """Refuse to blow away anything that looks like a real model directory."""
    if not os.path.exists(out):
        return
    if not force:
        sys.exit(f"ERROR: {out} exists; rerun with --force to overwrite")
    real_payload = [
        f for f in glob.glob(os.path.join(out, "*"))
        if not os.path.islink(f) and os.path.isfile(f) and os.path.getsize(f) > 100 * 1024 * 1024
    ]
    if real_payload:
        sys.exit(
            f"ERROR: refusing --force on {out}: it holds real (non-symlink) large files:\n  "
            + "\n  ".join(os.path.basename(f) for f in real_payload)
            + "\nThis does not look like a generated init dir. Remove it by hand if you are sure."
        )
    shutil.rmtree(out)


def verify(out, expect_rope_theta, expect_dtype):
    """Assert the config resolves correctly under the INSTALLED transformers.

    This is the part that matters. `from_pretrained` succeeding proves nothing:
    the 5.x-format config loads fine while silently yielding wrong defaults.
    """
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)

    rope_theta = getattr(cfg, "rope_theta", None)
    rope_params = getattr(cfg, "rope_parameters", None)
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        # transformers 5.x resolves RoPE through the nested dict.
        rope_theta = rope_params["rope_theta"]

    dtype = getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")

    print(f"[3/3] verify under transformers {transformers.__version__}:")
    print(f"        rope_theta   = {rope_theta}")
    print(f"        dtype        = {dtype}")
    print(f"        use_cache    = {cfg.use_cache}")
    print(f"        vocab_size   = {cfg.vocab_size} | <|im_start|> = {im_start}")
    print(f"        arch         = {cfg.architectures}")

    errs = []
    if rope_theta is None or float(rope_theta) != float(expect_rope_theta):
        errs.append(f"rope_theta resolved to {rope_theta}, expected {expect_rope_theta}")
    if dtype is None or str(expect_dtype) not in str(dtype):
        errs.append(f"dtype resolved to {dtype}, expected {expect_dtype}")
    if cfg.use_cache is not True:
        errs.append(f"use_cache is {cfg.use_cache}, expected True")
    if im_start != 151644:
        errs.append(f"<|im_start|> resolved to {im_start}, expected 151644")
    if errs:
        sys.exit("ERROR: verification FAILED:\n  " + "\n  ".join(errs))
    print("        all assertions passed")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sft-ckpt", required=True, help="SFT checkpoint dir (has model.safetensors)")
    ap.add_argument("--base-model", required=True, help="original base model dir (canonical tokenizer)")
    ap.add_argument("--out", required=True, help="output directory (for RL init or inference)")
    ap.add_argument("--force", action="store_true", help="overwrite if <out> exists")
    ap.add_argument(
        "--copy-weights", action="store_true",
        help="copy weights instead of symlinking; use when the training dir may be "
             "pruned or deleted, since symlinks would silently break",
    )
    ap.add_argument("--skip-verify", action="store_true", help="skip the load-back check")
    args = ap.parse_args()

    sft = os.path.abspath(args.sft_ckpt)
    base = os.path.abspath(args.base_model)
    out = os.path.abspath(args.out)
    for d, name in [(sft, "sft-ckpt"), (base, "base-model")]:
        if not os.path.isdir(d):
            sys.exit(f"ERROR: --{name} is not a directory: {d}")
    for d, name in [(sft, "sft-ckpt"), (base, "base-model")]:
        if not os.path.exists(os.path.join(d, "config.json")):
            sys.exit(f"ERROR: --{name} has no config.json: {d}")

    guard_output_dir(out, args.force)
    os.makedirs(out)

    # 1) trained weights from the SFT checkpoint
    placed = []
    for pat in WEIGHT_GLOBS:
        for f in sorted(glob.glob(os.path.join(sft, pat))):
            dst = os.path.join(out, os.path.basename(f))
            if args.copy_weights:
                shutil.copy2(f, dst)
            else:
                os.symlink(f, dst)
            placed.append(os.path.basename(f))
    if not glob.glob(os.path.join(out, "model*.safetensors")):
        sys.exit("ERROR: no model*.safetensors found in the SFT checkpoint")
    how = "copied" if args.copy_weights else "symlinked"
    print(f"[1/3] {how} {len(placed)} weight file(s) from SFT ckpt: {placed}")
    if not args.copy_weights:
        print(f"        NOTE: these point into {sft}")
        print("        If that dir is pruned, this init dir breaks. Use --copy-weights to detach.")

    # 2) merged config + generation config, canonical tokenizer
    base_cfg = load_json(os.path.join(base, "config.json"))
    ckpt_cfg = load_json(os.path.join(sft, "config.json"))
    cfg, notes = build_config(ckpt_cfg, base_cfg)
    write_json(os.path.join(out, "config.json"), cfg)
    print("[2/3] wrote merged config.json; compatibility keys added:")
    for n in notes or ["(none needed)"]:
        print(f"        - {n}")

    ckpt_gen_path = os.path.join(sft, "generation_config.json")
    base_gen_path = os.path.join(base, "generation_config.json")
    if os.path.exists(ckpt_gen_path):
        gen, gen_notes = build_generation_config(
            load_json(ckpt_gen_path),
            load_json(base_gen_path) if os.path.exists(base_gen_path) else {},
        )
        write_json(os.path.join(out, "generation_config.json"), gen)
        for n in gen_notes:
            print(f"        - generation_config: {n}")

    copied = []
    for f in TOKENIZER_FILES:
        src = os.path.join(base, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, f))
            copied.append(f)
    if not os.path.exists(os.path.join(out, "tokenizer_config.json")):
        sys.exit("ERROR: base model has no tokenizer_config.json")
    print(f"        copied {len(copied)} tokenizer file(s) from base model: {copied}")

    # 3) prove it actually resolves correctly here
    if args.skip_verify:
        print("[3/3] skipped (--skip-verify)")
    else:
        verify(out, expect_rope_theta=cfg["rope_theta"], expect_dtype=cfg["torch_dtype"])

    print(f"\nInit dir ready: {out}")
    print("Point MODEL_PATH at this path (the training checkpoint stays untouched).")


if __name__ == "__main__":
    main()
