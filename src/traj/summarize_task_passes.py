#!/usr/bin/env python3
"""Count full-score attempts per task in an OmniaBench aggregate JSON.

Run from the LIMA root:
    python3 src/traj/summarize_task_passes.py \
        --input data/trajectories/dag_s/full_glm_pass7

Add --output-dir results/dag_s/full_glm_pass7 to save CSV and JSON.
Without --output-dir, this program only prints results. Existing output
files are never overwritten. Source trajectory files are always read-only.

The score rule matches OmniaBench/evaluation/scripts/route_scores.py:
verifier_eval.score -> rubric_eval.avg_result -> total_reward, with
abs(score - 1.0) < 1e-6. Missing scores count as unsuccessful attempts.
This is a score count, not the stricter rollout-to-SFT acceptance filter.
Only the selected aggregate is read; runs/shards are not added again.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOL = 1e-6
RULE = (
    "verifier_eval.score -> rubric_eval.avg_result -> total_reward; "
    "abs(score - 1.0) < 1e-6; missing scores remain in the denominator"
)


def iter_entries(stream, chunk_size=1024 * 1024):
    """Read a JSON object or top-level array without loading the whole array."""
    decoder = json.JSONDecoder()
    buffer = ""

    def refill_nonempty(text):
        text = text.lstrip()
        while not text:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError("Unexpected end of JSON")
            text = chunk.lstrip()
        return text

    def decode_value(text):
        while True:
            try:
                value, end = decoder.raw_decode(text)
                return value, text[end:]
            except json.JSONDecodeError:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise
                text += chunk

    buffer = refill_nonempty(buffer)
    if buffer.startswith("{"):
        value, buffer = decode_value(buffer)
        yield value
    elif buffer.startswith("["):
        buffer = refill_nonempty(buffer[1:])
        if buffer.startswith("]"):
            buffer = buffer[1:]
        else:
            while True:
                # OmniaBench entries must be objects; this also rejects trailing commas.
                if not buffer.startswith("{"):
                    raise ValueError("Expected a result/group object in the JSON array")
                value, buffer = decode_value(buffer)
                yield value
                buffer = refill_nonempty(buffer)
                if buffer.startswith("]"):
                    buffer = buffer[1:]
                    break
                if not buffer.startswith(","):
                    raise ValueError("Expected ',' or ']' after a JSON entry")
                buffer = refill_nonempty(buffer[1:])
    else:
        raise ValueError("Expected an OmniaBench result object or array")
    if buffer.strip() or any(chunk.strip() for chunk in iter(lambda: stream.read(chunk_size), "")):
        raise ValueError("Unexpected content after the JSON document")


def field(item, name):
    info = item.get("task_info")
    values = [item.get(name), info.get(name) if isinstance(info, dict) else None]
    present = [value for value in values if value is not None]
    if present and any(str(value) != str(present[0]) for value in present):
        raise ValueError(f"Inconsistent {name}: {present}")
    return present[0] if present else None


def combined_score(sample):
    candidates = [
        ("verifier", sample.get("verifier_eval"), "score"),
        ("rubric", sample.get("rubric_eval"), "avg_result"),
        ("total_reward", sample, "total_reward"),
    ]
    for source, obj, key in candidates:
        if not isinstance(obj, dict) or obj.get(key) is None:
            continue
        try:
            value = float(obj[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {source} score: {obj[key]!r}")
        return source, value
    return None, None


def summarize(entries):
    tasks = {}
    source_counts = Counter()
    status_counts = Counter()
    termination_counts = Counter()
    missing_scores = 0
    parse_failures = 0
    for group in entries:
        if not isinstance(group, dict):
            raise ValueError("Result entries must be objects")
        grouped = "samples" in group
        samples = group.get("samples") if grouped else [group]
        if not isinstance(samples, list) or not samples:
            raise ValueError("A sample group must contain at least one sample")
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("Samples must be objects")
            identity = {}
            for name in ("global_id", "task_id", "env_id", "task_index"):
                value = field(sample, name)
                parent = field(group, name)
                if value is not None and parent is not None and str(value) != str(parent):
                    raise ValueError(f"Group/sample mismatch for {name}: {parent!r} vs {value!r}")
                identity[name] = value if value is not None else parent
            if identity["global_id"] is None or identity["task_id"] is None:
                raise ValueError("Each task must have global_id and task_id")
            gid = str(identity["global_id"])
            index = field(sample, "sample_idx")
            if index is None and grouped:
                raise ValueError(f"Missing sample_idx for global_id={gid}")
            if index is None:
                index = 1
            try:
                sample_idx = int(index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample_idx: {index!r}") from exc
            if str(index) != str(sample_idx) or sample_idx < 1:
                raise ValueError(f"Invalid sample_idx: {index!r}")
            task = tasks.setdefault(gid, {**identity, "samples": []})
            for name, value in identity.items():
                old = task[name]
                if old is not None and value is not None and str(old) != str(value):
                    raise ValueError(f"Conflicting {name} for global_id={gid}")
                if old is None:
                    task[name] = value
            if any(s["sample_idx"] == sample_idx for s in task["samples"]):
                raise ValueError(f"Duplicate global_id={gid}, sample_idx={sample_idx}")
            source, score = combined_score(sample)
            rubric = sample.get("rubric_eval") or {}
            meta = rubric.get("judge_meta") if isinstance(rubric, dict) else None
            parsed = meta.get("parse_success") if isinstance(meta, dict) else None
            passed = score is not None and abs(score - 1.0) < TOL
            task["samples"].append({
                "sample_idx": sample_idx,
                "score": score,
                "score_source": source,
                "passed": passed,
                "judge_parse_success": parsed,
                "result_status": sample.get("result_status"),
                "termination_reason": sample.get("termination_reason"),
            })
            source_counts[source or "missing"] += 1
            status_counts[sample.get("result_status") or "missing"] += 1
            termination_counts[sample.get("termination_reason") or "missing"] += 1
            missing_scores += score is None
            parse_failures += parsed is False

    if not tasks:
        raise ValueError("No task attempts found")
    rows = []
    for task in tasks.values():
        task["samples"].sort(key=lambda s: s["sample_idx"])
        samples = task["samples"]
        passed_indices = [s["sample_idx"] for s in samples if s["passed"]]
        task.update(
            attempts=len(samples),
            pass_count=len(passed_indices),
            pass_rate=len(passed_indices) / len(samples),
            sample_indices=[s["sample_idx"] for s in samples],
            pass_sample_indices=passed_indices,
            sample_scores=[s["score"] for s in samples],
            missing_score_count=sum(s["score"] is None for s in samples),
            judge_parse_failure_count=sum(s["judge_parse_success"] is False for s in samples),
        )
        rows.append(task)
    rows.sort(key=lambda r: (0, int(str(r["global_id"]))) if str(r["global_id"]).isdigit()
              else (1, str(r["global_id"])))
    attempts = sum(r["attempts"] for r in rows)
    passes = sum(r["pass_count"] for r in rows)
    covered = sum(r["pass_count"] > 0 for r in rows)
    histogram = Counter(r["pass_count"] for r in rows)
    summary = {
        "task_count": len(rows),
        "attempt_count": attempts,
        "pass_count": passes,
        "overall_attempt_pass_rate": passes / attempts,
        "tasks_with_any_pass": covered,
        "task_coverage": covered / len(rows),
        "tasks_with_all_passes": sum(r["pass_count"] == r["attempts"] for r in rows),
        "attempt_count_distribution": dict(sorted(Counter(r["attempts"] for r in rows).items())),
        "pass_count_distribution": {n: histogram[n] for n in range(max(r["attempts"] for r in rows) + 1)},
        "missing_score_count": missing_scores,
        "judge_parse_failure_count": parse_failures,
        "score_sources": dict(source_counts),
        "result_status_counts": dict(status_counts),
        "termination_reason_counts": dict(termination_counts),
    }
    return {"score_rule": RULE, "summary": summary, "tasks": rows}


def build_report(path):
    before = path.stat()
    with path.open(encoding="utf-8") as stream:
        report = summarize(iter_entries(stream))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed during analysis; retry with a stable aggregate")
    report["source"] = {
        "path": str(path),
        "size_bytes": before.st_size,
        "mtime_utc": datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
    }
    return report


def write_outputs(report, output_dir):
    paths = [output_dir / "task_pass_counts.csv", output_dir / "task_pass_counts.json"]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "global_id", "task_id", "env_id", "task_index", "attempts",
        "pass_count", "pass_rate", "missing_score_count", "judge_parse_failure_count",
        "sample_indices", "pass_sample_indices", "sample_scores",
    ]
    with paths[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for task in report["tasks"]:
            row = {key: task[key] for key in columns}
            for key in ("sample_indices", "pass_sample_indices", "sample_scores"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    with paths[1].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Aggregate JSON, or directory containing exactly one aggregate JSON")
    parser.add_argument("--output-dir", type=Path,
                        help="Optionally create task_pass_counts.csv and task_pass_counts.json")
    args = parser.parse_args()
    try:
        source = args.input.expanduser().resolve()
        if source.is_dir():
            candidates = sorted(source.glob("*.json"))
            if len(candidates) != 1:
                raise ValueError(f"Found {len(candidates)} JSON files; specify the aggregate JSON explicitly")
            source = candidates[0]
        report = build_report(source)
        summary = report["summary"]
        print(f"Source: {source}")
        print(f"Score rule: {RULE}")
        print(f"Tasks: {summary['task_count']}; attempts: {summary['attempt_count']}; "
              f"passes: {summary['pass_count']} ({summary['overall_attempt_pass_rate']:.2%})")
        print(f"Tasks with any pass: {summary['tasks_with_any_pass']} "
              f"({summary['task_coverage']:.2%}); all attempts pass: {summary['tasks_with_all_passes']}")
        print(f"Missing scores: {summary['missing_score_count']}; "
              f"judge parse failures: {summary['judge_parse_failure_count']}")
        print("Pass count -> number of tasks:")
        for count, total in summary["pass_count_distribution"].items():
            print(f"  {count}: {total}")
        print("\nglobal_id\tpass_count\tattempts\tpass_sample_indices")
        for task in report["tasks"]:
            indices = ",".join(map(str, task["pass_sample_indices"])) or "-"
            print(f"{task['global_id']}\t{task['pass_count']}\t{task['attempts']}\t{indices}")
        if args.output_dir is not None:
            for path in write_outputs(report, args.output_dir.expanduser().resolve()):
                print(f"Saved: {path}")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
