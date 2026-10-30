#!/usr/bin/env python3
"""Filter complete SFT examples using the local LLaMA-Factory training encoder.

Run with the SFT environment, from LIMA (or use absolute paths)::

    python -B src/select/filter_sft_by_length.py \
      --config LLaMA-Factory/llamafactory_runs/qwen3-8b_full/spbench_route1_sft_v2.yaml \
      --output data/sft/spbench_route1/spbench_route1_sft_v2_32k.json --dry-run

Remove --dry-run to write the JSON subset, .manifest.jsonl, and
.length_audit.jsonl. Existing outputs are never overwritten. Only complete
examples are selected; retained JSON values and histories are not edited.

Supports local JSON-array Alpaca SFT data, text-only templates, and no packing.
Relative dataset/model paths inside YAML are resolved against LLaMA-Factory,
as when launching training from that directory. The source dataset is resolved
through dataset_info.json; --input can explicitly override its JSON path.
An adjacent source manifest is used when available, or can be set by --manifest.
No model weights, Trainer, or training job are loaded.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import fields, replace
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LF_ROOT = ROOT / "LLaMA-Factory"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_from(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def within_cutoff(length, cutoff):
    return length <= cutoff


def validate_destinations(output, protected, dry_run):
    if output.suffix != ".json":
        raise ValueError("--output must end in .json")
    paths = [output, output.with_suffix(".manifest.jsonl"), output.with_suffix(".length_audit.jsonl")]
    for path in paths:
        if path.resolve() in {p.resolve() for p in protected if p is not None}:
            raise ValueError(f"Output would replace an input/config file: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {path.parent}")
        if not dry_run and os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return paths


def manifest_owners(rows, sample_count):
    owners = []
    for index, row in enumerate(rows):
        count = row.get("n_samples")
        if type(count) is not int or count < 0 or row.get("kept") is not (count > 0):
            raise ValueError(f"Invalid manifest count/kept at row {index}")
        owners.extend([index] * count)
    if len(owners) != sample_count:
        raise ValueError(f"Manifest covers {len(owners)} examples, dataset has {sample_count}")
    return owners


def filtered_manifest(rows, audit, metadata):
    kept_counts = Counter(row["manifest_index"] for row in audit if row["kept"])
    derived = []
    source_start = output_start = 0
    for index, row in enumerate(rows):
        count, kept = row["n_samples"], kept_counts[index]
        item = dict(row)
        item.update(n_samples=kept, kept=kept > 0)
        if count and not kept:
            item["reason"] = "all_samples_over_cutoff"
        # Original conversion HIDE/call audits remain untouched. Length filtering
        # removes supervision examples, not turns from the saved histories.
        item["length_filter"] = {
            **metadata,
            "source_kept": row["kept"], "source_reason": row.get("reason"),
            "source_n_samples": count, "n_kept": kept, "n_removed": count - kept,
            "source_index_range": [source_start, source_start + count],
            "output_index_range": [output_start, output_start + kept],
        }
        derived.append(item)
        source_start += count
        output_start += kept
    return derived


def load_runtime(config, lf_root, data_dir):
    # Use this checkout even if another LLaMA-Factory is installed in the env.
    sys.path.insert(0, str(lf_root / "src"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from llamafactory.data.converter import AlpacaDatasetConverter
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model.loader import load_tokenizer

    model_path = resolve_from(config["model_name_or_path"], lf_root)
    if not model_path.is_dir():
        raise ValueError(f"Only a local model/tokenizer directory is supported: {model_path}")
    data_fields = {f.name for f in fields(DataArguments) if f.init}
    data_args = DataArguments(**{k: v for k, v in config.items() if k in data_fields})
    if data_args.cutoff_len <= 0:
        raise ValueError("cutoff_len must be positive")
    if data_args.tokenized_path or data_args.streaming:
        raise ValueError("Use raw local JSON data, without tokenized_path or streaming")
    data_args.dataset_dir = str(data_dir)
    data_args.media_dir = str(data_dir)
    if len(data_args.dataset or []) != 1:
        raise ValueError("Select exactly one dataset in the YAML")
    attr = get_dataset_list(data_args.dataset, str(data_dir))[0]
    if attr.load_from != "file" or attr.formatting != "alpaca" or attr.ranking or attr.kto_tag:
        raise ValueError("Only local, non-ranking Alpaca SFT datasets are supported")
    if any((attr.images, attr.videos, attr.audios)) or attr.num_samples is not None:
        raise ValueError("Media columns and dataset num_samples are not supported")

    model_fields = {f.name for f in fields(ModelArguments) if f.init}
    model_config = {k: v for k, v in config.items() if k in model_fields}
    model_config["model_name_or_path"] = str(model_path)
    model_args = ModelArguments(**model_config)
    model_args.model_max_length = data_args.cutoff_len
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    if type(template.mm_plugin).__name__ != "BasePlugin":
        raise ValueError("Only text-only templates using BasePlugin are supported")
    # Repeated system/history strings dominate these step-wise SFT datasets.
    tokenizer.encode = lru_cache(maxsize=32768)(tokenizer.encode)
    converter = AlpacaDatasetConverter(attr, data_args)
    processor = SupervisedDatasetProcessor(template, tokenizer, None, data_args)
    return attr, data_args, converter, processor, model_path


def full_encoding(converted, processor):
    if len(converted["_prompt"]) % 2 != 1 or len(converted["_response"]) != 1:
        raise ValueError("Invalid SFT prompt/response structure")
    if any(converted[k] for k in ("_images", "_videos", "_audios")):
        raise ValueError("Only text-only examples are supported")
    template, tok = processor.template, processor.tokenizer
    messages = template.mm_plugin.process_messages(
        converted["_prompt"] + converted["_response"], [], [], [], None
    )
    pairs = template.encode_multiturn(
        tok, messages, converted["_system"], converted["_tools"],
        processor.data_args.mask_history and not template.preserve_thinking,
    )
    complete_ids = [token for source, target in pairs for part in (source, target) for token in part]
    if template.efficient_eos:
        complete_ids.append(tok.eos_token_id)
    kwargs = dict(prompt=converted["_prompt"], response=converted["_response"],
                  system=converted["_system"], tools=converted["_tools"], images=[], videos=[], audios=[])
    # Encode once with exactly enough room, using the real LF label construction.
    full_processor = type(processor)(template, tok, None,
                                     replace(processor.data_args, cutoff_len=len(complete_ids)))
    full_ids, full_labels = full_processor._encode_data_example(**kwargs)
    if full_ids != complete_ids or len(full_labels) != len(full_ids):
        raise ValueError("LF full encoding differs from the untruncated template; refusing to filter")
    if not any(label != -100 for label in full_labels):
        raise ValueError("Example contains no supervised target tokens")
    if within_cutoff(len(full_ids), processor.data_args.cutoff_len):
        ids, labels = processor._encode_data_example(**kwargs)
        if ids != full_ids or labels != full_labels:
            raise ValueError("Retained example changes tokens/labels at the training cutoff")
    return len(full_ids), sum(label != -100 for label in full_labels)


def write_outputs(paths, samples, manifests, audit):
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for path in paths:
                streams.append(stack.enter_context(path.open("x", encoding="utf-8")))
                created.append(path)
            # Stream one sample at a time instead of constructing another huge JSON string.
            streams[0].write("[\n")
            for index, sample in enumerate(samples):
                if index:
                    streams[0].write(",\n")
                json.dump(sample, streams[0], ensure_ascii=False)
            streams[0].write("\n]\n")
            for stream, rows in zip(streams[1:], (manifests, audit)):
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except BaseException:
        # Only remove partial files exclusively created by this invocation.
        for path in created:
            path.unlink()
        raise


def run(args):
    import yaml

    start = time.monotonic()
    config_path = Path(args.config).expanduser().resolve()
    lf_root = Path(args.llamafactory_dir).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or config.get("stage", "sft") != "sft":
        raise ValueError("--config must be an SFT YAML mapping")
    if config.get("packing") or config.get("neat_packing"):
        raise ValueError("Packing changes sequence semantics; this filter supports packing=false only")
    data_dir = resolve_from(config.get("dataset_dir", "data"), lf_root)
    info_path = data_dir / "dataset_info.json"
    # Check output collisions before importing the training dependencies.
    names = config.get("dataset", [])
    names = names.split(",") if isinstance(names, str) else names
    if not isinstance(names, list) or len(names) != 1:
        raise ValueError("Select exactly one dataset in the YAML")
    entry = json.loads(info_path.read_text())[names[0].strip()]
    input_path = (Path(args.input).expanduser().resolve() if args.input
                  else resolve_from(entry["file_name"], data_dir))
    if input_path.suffix != ".json":
        raise ValueError("Input must be a JSON array file ending in .json")
    manifest_path = (Path(args.manifest).expanduser().resolve() if args.manifest
                     else input_path.with_suffix(".manifest.jsonl"))
    if args.manifest and not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_path = manifest_path if manifest_path.is_file() else None
    output = Path(os.path.abspath(Path(args.output).expanduser()))
    protected = [input_path, manifest_path, config_path, info_path]
    paths = validate_destinations(output, protected, args.dry_run)
    source_hashes = {str(p): sha256(p) for p in protected if p is not None}
    data = json.loads(input_path.read_text())
    if not isinstance(data, list) or not all(isinstance(s, dict) for s in data):
        raise ValueError("Input must be a JSON array of objects")
    if manifest_path:
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    else:
        rows = [{"source": str(input_path), "n_samples": len(data), "kept": bool(data),
                 "reason": "ok" if data else "empty_dataset"}]
    owners = manifest_owners(rows, len(data))
    attr, data_args, converter, processor, model_path = load_runtime(config, lf_root, data_dir)
    if data_args.max_samples is not None and data_args.max_samples < len(data):
        raise ValueError("This filter processes the whole source file; increase/remove max_samples in the config")
    cutoff = data_args.cutoff_len
    audit, retained = [], []
    print(f"Input: {input_path}\nExamples: {len(data)}, cutoff: {cutoff}, template: {data_args.template}", flush=True)
    for index, sample in enumerate(data):
        try:
            length, target_tokens = full_encoding(converter(sample), processor)
        except Exception as exc:
            raise ValueError(f"Example {index}: {exc}") from exc
        keep = within_cutoff(length, cutoff)
        manifest_index = owners[index]
        owner = rows[manifest_index]
        audit.append({
            "source_index": index, "output_index": len(retained) if keep else None,
            "manifest_index": manifest_index,
            **{key: owner.get(key) for key in ("global_id", "task_id", "task_index", "sample_idx")},
            "turn": len(sample.get(attr.history) or []) if attr.history else 0,
            "full_length": length, "supervised_tokens": target_tokens,
            "cutoff_len": cutoff, "kept": keep,
            "reason": "within_cutoff" if keep else "over_cutoff",
        })
        if keep:
            retained.append(sample)
        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1}/{len(data)}; kept {len(retained)}; {time.monotonic() - start:.1f}s", flush=True)

    source_files = ["data/converter.py", "data/template.py", "data/formatter.py", "data/tool_utils.py",
                    "data/mm_plugin.py", "data/processor/supervised.py",
                    "data/processor/processor_utils.py", "hparams/data_args.py"]
    metadata = {
        "policy": "complete_sft_length_lte_cutoff_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_len": cutoff, "source_sft": str(input_path), "source_sha256": source_hashes[str(input_path)],
        "source_manifest": str(manifest_path) if manifest_path else None,
        "source_manifest_sha256": source_hashes.get(str(manifest_path)),
        "config": str(config_path), "config_sha256": source_hashes[str(config_path)],
        "dataset_registration": entry, "model_name_or_path": str(model_path),
        "template": data_args.template, "mask_history": data_args.mask_history,
        "train_on_prompt": data_args.train_on_prompt, "enable_thinking": data_args.enable_thinking,
        "preserve_thinking": data_args.preserve_thinking,
        "versions": {"python": sys.version.split()[0],
                     **{name: version(name) for name in ("transformers", "tokenizers")}},
        "tokenizer_sha256": {p.name: sha256(p) for p in model_path.iterdir()
                             if p.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                           "added_tokens.json", "chat_template.jinja", "config.json"}},
        "lf_source_sha256": {p: sha256(lf_root / "src/llamafactory" / p) for p in source_files},
        "script_sha256": sha256(Path(__file__)), "length_audit": str(paths[2]),
    }
    derived = filtered_manifest(rows, audit, metadata)
    if sum(row["n_samples"] for row in derived) != len(retained):
        raise ValueError("Derived manifest count does not match output")
    # Avoid publishing results if inputs were edited while tokenization was running.
    for path, digest in source_hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Input/config changed during filtering: {path}")
    summary = {
        "input_samples": len(data), "kept_samples": len(retained), "removed_samples": len(data) - len(retained),
        "tasks_before": len({r.get("global_id") for r in rows if r["n_samples"] and r.get("global_id") is not None}),
        "tasks_after": len({r.get("global_id") for r in derived if r["n_samples"] and r.get("global_id") is not None}),
        "max_kept_length": max((r["full_length"] for r in audit if r["kept"]), default=0),
        "supervised_tokens_before": sum(r["supervised_tokens"] for r in audit),
        "supervised_tokens_removed": sum(r["supervised_tokens"] for r in audit if not r["kept"]),
        "source_sha256": metadata["source_sha256"], "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        write_outputs(paths, retained, derived, audit)
        for path in paths:
            print(f"Wrote {path}", flush=True)
    else:
        print("Dry run: no dataset/manifest/audit files written.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Training YAML providing tokenizer/template/cutoff settings")
    parser.add_argument("--output", required=True, help="New filtered dataset JSON; existing outputs are refused")
    parser.add_argument("--input", help="Override the source JSON from dataset_info.json")
    parser.add_argument("--manifest", help="Source manifest; defaults to the input's adjacent .manifest.jsonl")
    parser.add_argument("--llamafactory-dir", default=str(LF_ROOT), help="Local LLaMA-Factory checkout")
    parser.add_argument("--dry-run", action="store_true", help="Encode and validate, without writing outputs")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
