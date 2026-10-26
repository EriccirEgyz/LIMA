"""Plot cumulative rollout completion from finished Harbor jobs, entirely offline.

Run with terminalbench/.venv/bin/python (no Harbor import or model connection):
  python plot_rollout_completion.py JOB_DIR
  python plot_rollout_completion.py JOB_A JOB_B --label Base --label SFT \
      --output-dir COMPARISON_DIR

Each job is a separate series. X is agent_execution.finished_at - started_at,
in hours; Y is 100 * count(duration <= X) / job.n_total_trials. All outcomes
count, including agent/verifier errors and timeouts. Missing/invalid timings
remain in the denominator, are never assigned zero, and appear in trials.csv.
Timings are not rounded or clipped to the time budget. This is elapsed runtime,
not a measure of task success, and excludes setup and verifier time.

Only direct JOB_DIR/*/result.json files are read (final trial results, not
nested retry archives, logs or artifacts). Source files are never modified.
Jobs must be finished; incomplete jobs are rejected to avoid selection bias.
PDF/SVG/PNG, curve.csv, trials.csv and summary.json are exported by default.
Existing exports require --overwrite. For controlled comparisons, match tasks,
versions, harness, resources, budgets and concurrency across jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL_SOURCE = (
    "https://github.com/harbor-framework/terminal-bench-website/blob/"
    "42783523c549242bf2c745cb0770f86260e5ef3f/"
    "public/blog/terminal-bench-4-0/rollout-charts.html#L254"
)
THRESHOLD_HOURS = (0.25, 0.5, 1, 2, 4, 8)


def read_json(path: Path, hashes: dict) -> dict:
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def execution_duration(timing) -> tuple[float | None, str]:
    if not timing:
        return None, "missing_agent_execution"
    if not isinstance(timing, dict):
        return None, "invalid_agent_execution"
    for key in ("started_at", "finished_at"):
        if not timing.get(key):
            return None, f"missing_{key}"
    try:
        start = datetime.fromisoformat(timing["started_at"])
        end = datetime.fromisoformat(timing["finished_at"])
    except (ValueError, TypeError):
        return None, "invalid_timestamp"
    # Accept pairs of naive timestamps as recorded. Never guess a timezone
    # when only one endpoint has an offset.
    if (start.tzinfo is None) != (end.tzinfo is None):
        return None, "mixed_timezones"
    seconds = (end - start).total_seconds()
    if seconds < 0:
        return None, "negative_duration"
    return seconds, "valid"


def cumulative_points(seconds: list[float], total: int) -> list[dict]:
    if type(total) is not int or total <= 0 or len(seconds) > total:
        raise ValueError("Invalid total trial count")
    if any(not math.isfinite(s) or s < 0 for s in seconds):
        raise ValueError("Durations must be finite and nonnegative")
    points = [{"elapsed_hours": 0.0, "completed": 0, "percent": 0.0}]
    completed = 0
    # A tied duration produces one jump, with all ties included at that X.
    for duration, count in sorted(Counter(seconds).items()):
        completed += count
        points.append({"elapsed_hours": duration / 3600, "completed": completed,
                       "percent": 100 * completed / total})
    return points


def load_job(path: Path, label: str | None = None) -> dict:
    job = path.resolve()
    if job.is_file() and job.name == "result.json":
        job = job.parent
    hashes = {}
    result = read_json(job / "result.json", hashes)
    config = read_json(job / "config.json", hashes)
    total = result.get("n_total_trials")
    if type(total) is not int or total <= 0:
        raise ValueError(f"{job}: n_total_trials must be a positive integer")
    stats = result.get("stats") or {}
    if (not result.get("finished_at") or stats.get("n_running_trials", 0)
            or stats.get("n_pending_trials", 0)):
        raise ValueError(f"{job}: job is unfinished; plot after evaluation completes")
    agents = config.get("agents") or []
    if len(agents) != 1:
        raise ValueError(f"{job}: expected one agent/model per job for a single curve")
    agent = agents[0]
    model = agent.get("model_name") or "unknown model"
    default_label = f"{model.rstrip('/').split('/')[-1]} / {agent.get('name', 'unknown agent')}"
    rows, names, ids = [], set(), set()
    task_counts, task_refs = Counter(), {}
    for file in sorted(job.glob("*/result.json")):
        trial = read_json(file, hashes)
        name, task = trial.get("trial_name"), trial.get("task_name")
        if not name or not task:
            raise ValueError(f"{file}: missing trial_name or task_name")
        if name in names or (trial.get("id") and trial["id"] in ids):
            raise ValueError(f"{file}: duplicate trial identity")
        names.add(name)
        if trial.get("id"):
            ids.add(trial["id"])
        if not trial.get("finished_at"):
            raise ValueError(f"{file}: trial is unfinished despite finished job summary")
        seconds, status = execution_duration(trial.get("agent_execution"))
        timing = trial.get("agent_execution")
        timing = timing if isinstance(timing, dict) else {}
        rows.append({
            "trial_name": name, "task_name": task, "timing_status": status,
            "exception_type": (trial.get("exception_info") or {}).get("exception_type"),
            "agent_started_at": timing.get("started_at"),
            "agent_finished_at": timing.get("finished_at"),
            "agent_execution_seconds": seconds,
            "agent_execution_hours": seconds / 3600 if seconds is not None else None,
        })
        task_counts[task] += 1
        task_refs.setdefault(task, set()).add((trial.get("task_id") or {}).get("ref") or "unknown")
    if len(rows) > total:
        raise ValueError(f"{job}: {len(rows)} trial results exceed n_total_trials={total}")
    durations = sorted(r["agent_execution_seconds"] for r in rows
                       if r["agent_execution_seconds"] is not None)
    status_counts = dict(Counter(r["timing_status"] for r in rows))
    status_counts["missing_trial_result"] = total - len(rows)
    return {
        "job_directory": str(job), "label": label or default_label,
        "job_finished_at": result["finished_at"],
        "agent": {"name": agent.get("name"), "model_name": model,
                  "version": (agent.get("kwargs") or {}).get("version")},
        "datasets": [{k: d[k] for k in ("name", "ref") if k in d}
                     for d in config.get("datasets", [])],
        "n_attempts": config.get("n_attempts"),
        "n_concurrent_trials": config.get("n_concurrent_trials"),
        "n_retries": stats.get("n_retries", 0),
        "n_total_trials": total, "n_trial_results": len(rows),
        "n_timed_trials": len(durations), "n_missing_timings": total - len(durations),
        "timing_coverage_percent": 100 * len(durations) / total,
        "timing_status_counts": status_counts,
        "exception_counts": dict(Counter(r["exception_type"] for r in rows if r["exception_type"])),
        "n_observed_tasks": len(task_counts), "task_trial_counts": dict(task_counts),
        "task_refs": {task: sorted(refs) for task, refs in task_refs.items()},
        "duration_seconds_among_timed_trials": {
            "min": min(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p90_nearest_rank": durations[math.ceil(0.9 * len(durations)) - 1] if durations else None,
            "max": max(durations) if durations else None,
        },
        "thresholds": [{"elapsed_hours": h,
                        "completed": sum(s <= h * 3600 for s in durations),
                        "percent": 100 * sum(s <= h * 3600 for s in durations) / total}
                       for h in THRESHOLD_HOURS],
        "source_sha256": hashes,
        "points": cumulative_points(durations, total), "trials": rows,
    }


def comparison_notes(series: list[dict]) -> list[str]:
    notes = []
    for key, description in (("task_trial_counts", "task sets or attempts per task"),
                             ("task_refs", "task revisions"),
                             ("datasets", "dataset references"),
                             ("n_concurrent_trials", "concurrency")):
        if any(s[key] != series[0][key] for s in series[1:]):
            notes.append(f"Jobs differ in {description}; interpret the overlay accordingly.")
    return notes


def make_figure(series: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333")
    styles = ("-", "--", "-.", ":")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.8, 4.9))
        for index, s in enumerate(series):
            points = s["points"]
            label = (f"{s['label']}\n{s['n_timed_trials']}/{s['n_total_trials']} timed; "
                     f"{s['n_observed_tasks']} observed tasks")
            color = colors[index % len(colors)]
            ax.step([p["elapsed_hours"] for p in points], [p["percent"] for p in points],
                    where="post", linewidth=1.8, color=color,
                    linestyle=styles[(index // len(colors)) % len(styles)], label=label)
            ax.plot(points[-1]["elapsed_hours"], points[-1]["percent"], "s", color=color, markersize=4)
        last_hour = max(s["points"][-1]["elapsed_hours"] for s in series)
        ax.set(xlim=(0, max(last_hour * 1.025, 0.05)), ylim=(0, 102),
               xlabel="Agent execution time (hours)", ylabel="Cumulative rollout completion",
               title="Cumulative rollout completion")
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(color="#e5e7eb", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e5e7eb", fontsize=9)
        fig.text(0.12, 0.035,
                 "All outcomes included (success, failure, timeout). This is runtime, not task success.\n"
                 "Missing timings stay in the denominator. Setup and verifier time are excluded.",
                 fontsize=8, color="#555555", linespacing=1.5)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def export_report(series: list[dict], output: Path, formats: list[str], overwrite: bool = False):
    output = output.resolve()
    filenames = [f"rollout_completion.{fmt}" for fmt in formats] + ["curve.csv", "trials.csv", "summary.json"]
    for name in filenames:
        path = output / name
        if path.exists() and not overwrite:
            raise ValueError(f"Output already exists: {path}; use --overwrite or a new --output-dir")
        if path.is_symlink():
            raise ValueError(f"Refusing to overwrite a symlink: {path}")
    notes = comparison_notes(series)
    # A temporary font cache keeps standalone plotting out of user/host config.
    with tempfile.TemporaryDirectory(prefix="terminalbench-plot-") as cache:
        previous_cache = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            fig = make_figure(series)
        finally:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache
        import matplotlib
        import matplotlib.pyplot as plt
        output.mkdir(parents=True, exist_ok=True)
        try:
            for fmt in formats:
                # Save inside the rc context too: PDF/SVG resolve font settings
                # at export time, not just when the Figure is constructed.
                with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
                    fig.savefig(output / f"rollout_completion.{fmt}", dpi=300, bbox_inches="tight")
        finally:
            plt.close(fig)
    with (output / "curve.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "job_directory", "elapsed_hours",
                                                "completed", "n_total_trials", "percent"])
        writer.writeheader()
        for s in series:
            for p in s["points"]:
                writer.writerow({"label": s["label"], "job_directory": s["job_directory"],
                                 "n_total_trials": s["n_total_trials"], **p})
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "job_directory", "trial_name", "task_name",
                                                "timing_status", "exception_type", "agent_started_at",
                                                "agent_finished_at", "agent_execution_seconds",
                                                "agent_execution_hours"])
        writer.writeheader()
        for s in series:
            for row in s["trials"]:
                writer.writerow({"label": s["label"], "job_directory": s["job_directory"], **row})
    summary = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "cumulative_rollout_completion", "official_reference": OFFICIAL_SOURCE,
        "definition": "100 * count(agent_execution_seconds <= elapsed_hours * 3600) / n_total_trials",
        "missing_timing_policy": "Keep in denominator; no imputation; no curve event.",
        "outcome_policy": "Include all outcomes with valid timing, including errors and timeouts.",
        "retry_policy": "Use direct final trial result only; do not sum or count archived retries.",
        "duration_policy": "Use measured agent duration; no rounding or budget clipping.",
        "notes": notes, "matplotlib_version": matplotlib.__version__,
        "series": [{k: v for k, v in s.items() if k != "trials"} for s in series],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return notes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", type=Path, nargs="+", help="Finished job directories (or their root result.json)")
    parser.add_argument("--label", action="append", help="Legend label; repeat once per job in input order")
    parser.add_argument("--output-dir", type=Path, help="Default: JOB_DIR/plots/rollout_completion; required for multiple jobs")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "svg", "png"])
    parser.add_argument("--overwrite", action="store_true", help="Replace this tool's existing export files")
    args = parser.parse_args(argv)
    if args.label is not None and (len(args.label) != len(args.jobs) or any(not s.strip() for s in args.label)):
        parser.error("--label must be nonempty and provided once per job")
    if len(args.jobs) > 1 and not args.output_dir:
        parser.error("Multiple jobs require --output-dir")
    try:
        series = [load_job(job, args.label[i] if args.label else None) for i, job in enumerate(args.jobs)]
        if len({s["job_directory"] for s in series}) != len(series):
            raise ValueError("Each job may be supplied only once")
        if not args.label:
            labels = Counter(s["label"] for s in series)
            for s in series:
                if labels[s["label"]] > 1:
                    s["label"] += f" / {Path(s['job_directory']).name}"
        if len({s["label"] for s in series}) != len(series):
            raise ValueError("Legend labels must be unique; set distinct --label values")
        output = args.output_dir or Path(series[0]["job_directory"]) / "plots" / "rollout_completion"
        notes = export_report(series, output, list(dict.fromkeys(args.formats)), args.overwrite)
    except ImportError as exc:
        parser.error(f"Missing plotting dependency ({exc}); run uv sync --project src/eval/terminalbench --locked")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for s in series:
        print(f"{s['label']}: {s['n_timed_trials']}/{s['n_total_trials']} timed "
              f"({s['timing_coverage_percent']:.2f}%); {s['n_missing_timings']} missing")
    for note in notes:
        print("NOTE: " + note, file=sys.stderr)
    print(f"Saved to {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
