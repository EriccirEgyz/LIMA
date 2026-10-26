"""Offline regression tests: python -B -m unittest discover -s src/eval/terminalbench -p 'test_plot_rollout_completion.py'."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import plot_rollout_completion as plot


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="terminalbench-plot-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def job(self, name="job", task="example"):
        job = self.root / name
        self.write(job / "result.json", {
            "n_total_trials": 6, "finished_at": "2026-10-26T10:00:00Z",
            "stats": {"n_running_trials": 0, "n_pending_trials": 0, "n_retries": 0},
        })
        self.write(job / "config.json", {
            "agents": [{"name": "terminus-2", "model_name": "openai/test-model"}],
            "datasets": [{"name": "terminal-bench", "ref": "sha256:test"}],
            "n_attempts": 6, "n_concurrent_trials": 2,
        })
        start = datetime(2026, 10, 26, tzinfo=timezone.utc)
        for trial_name, seconds, reward, error in [
            ("pass", 3600, 1, None), ("fail", 3600, 0, None),
            ("timeout", 28800.1, 0, "AgentTimeoutError"),
            ("verifier_error", 1800, None, "VerifierTimeoutError"),
            ("setup_error", None, None, "RuntimeError"),
        ]:
            timing = None if seconds is None else {
                "started_at": start.isoformat(),
                "finished_at": (start + timedelta(seconds=seconds)).isoformat(),
            }
            self.write(job / trial_name / "result.json", {
                "id": trial_name, "trial_name": trial_name, "task_name": task,
                "task_id": {"ref": "sha256:task"}, "agent_execution": timing,
                # Deliberately different from execution time: do not use overall
                # trial duration (which also includes setup and verification).
                "started_at": "2026-10-25T00:00:00Z",
                "finished_at": "2026-10-26T10:00:00Z",
                "exception_info": {"exception_type": error} if error else None,
                "verifier_result": {"rewards": {"reward": reward}},
            })
        return job

    def test_all_outcomes_and_missing_records_keep_total_denominator(self):
        s = plot.load_job(self.job())
        self.assertEqual(s["n_trial_results"], 5)
        self.assertEqual(s["n_timed_trials"], 4)
        self.assertEqual(s["n_missing_timings"], 2)
        self.assertEqual(s["timing_status_counts"]["missing_agent_execution"], 1)
        self.assertEqual(s["timing_status_counts"]["missing_trial_result"], 1)
        self.assertAlmostEqual(s["points"][-1]["percent"], 100 * 4 / 6)
        self.assertEqual(s["duration_seconds_among_timed_trials"]["median"], 3600)
        self.assertEqual(s["exception_counts"], {
            "AgentTimeoutError": 1, "VerifierTimeoutError": 1, "RuntimeError": 1})
        self.assertEqual(s["points"][2]["completed"], 3)  # Both tied 1h trials.
        self.assertEqual(s["thresholds"][-1]["completed"], 3)  # No clipping to 8h.
        self.assertGreater(s["points"][-1]["elapsed_hours"], 8)

    def test_zero_and_ties_are_right_continuous(self):
        points = plot.cumulative_points([3600, 0, 3600, 1800], 5)
        self.assertEqual([p["elapsed_hours"] for p in points], [0, 0, 0.5, 1])
        self.assertEqual([p["completed"] for p in points], [0, 1, 2, 4])
        self.assertEqual([p["percent"] for p in points], [0, 20, 40, 80])
        for durations, total in [([1], 0), ([1, 2], 1), ([-1], 1), ([float("nan")], 1)]:
            with self.subTest(durations=durations, total=total), self.assertRaises(ValueError):
                plot.cumulative_points(durations, total)

    def test_timezone_and_invalid_timing_policy(self):
        self.assertEqual(plot.execution_duration({
            "started_at": "2026-10-26T00:00:00Z", "finished_at": "2026-10-26T09:00:00+08:00",
        }), (3600, "valid"))
        self.assertEqual(plot.execution_duration({
            "started_at": "2026-10-26T00:00:00", "finished_at": "2026-10-26T00:00:01",
        }), (1, "valid"))
        cases = [
            (None, "missing_agent_execution"),
            ({"started_at": "2026-10-26T00:00:00Z"}, "missing_finished_at"),
            ({"started_at": "bad", "finished_at": "bad"}, "invalid_timestamp"),
            ({"started_at": "2026-10-26T00:00:00", "finished_at": "2026-10-26T00:00:00Z"}, "mixed_timezones"),
            ({"started_at": "2026-10-26T00:00:01Z", "finished_at": "2026-10-26T00:00:00Z"}, "negative_duration"),
        ]
        for timing, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(plot.execution_duration(timing), (None, expected))

    def test_nested_results_are_not_counted(self):
        job = self.job()
        self.write(job / "pass" / "retry_1" / "result.json", {"not": "a final result"})
        self.write(job / "pass" / "artifacts" / "result.json", {"not": "a trial"})
        s = plot.load_job(job / "result.json")
        self.assertEqual(s["n_trial_results"], 5)
        self.assertEqual(len(s["source_sha256"]), 7)

    def test_reject_unfinished_and_inconsistent_jobs(self):
        job = self.job()
        root_result = json.loads((job / "result.json").read_text())
        for change in [{"finished_at": None}, {"stats": {"n_pending_trials": 1}},
                       {"n_total_trials": 4}, {"n_total_trials": 0}]:
            self.write(job / "result.json", {**root_result, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                plot.load_job(job)
        self.write(job / "result.json", root_result)
        result_path = job / "pass" / "result.json"
        trial = json.loads(result_path.read_text())
        self.write(result_path, {**trial, "finished_at": None})
        with self.assertRaisesRegex(ValueError, "trial is unfinished"):
            plot.load_job(job)
        self.write(result_path, {**trial, "id": "fail"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plot.load_job(job)

    def test_reject_malformed_result(self):
        job = self.job()
        (job / "pass" / "result.json").write_text("{")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            plot.load_job(job)

    def test_no_valid_timings_does_not_become_100_percent(self):
        job = self.job()
        for path in job.glob("*/result.json"):
            trial = json.loads(path.read_text())
            trial["agent_execution"] = None
            self.write(path, trial)
        s = plot.load_job(job)
        self.assertEqual(s["n_missing_timings"], 6)
        self.assertEqual(s["points"], [{"elapsed_hours": 0, "completed": 0, "percent": 0}])
        self.assertIsNone(s["duration_seconds_among_timed_trials"]["median"])

    def test_cli_overlay_exports_and_preserves_sources(self):
        first, second = self.job("first"), self.job("second", task="different-task")
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for job in (first, second) for p in job.rglob("*.json")}
        output = self.root / "overlay"
        args = [str(first), str(second), "--label", "Base", "--label", "SFT", "--output-dir", str(output)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(plot.main(args), 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual([s["label"] for s in summary["series"]], ["Base", "SFT"])
        self.assertTrue(any("task sets" in note for note in summary["notes"]))
        self.assertEqual((output / "rollout_completion.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue((output / "rollout_completion.pdf").read_bytes().startswith(b"%PDF-"))
        self.assertIn("Cumulative rollout completion", (output / "rollout_completion.svg").read_text())
        with (output / "trials.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(r["timing_status"] != "valid" for r in rows), 2)
        with (output / "curve.csv").open() as file:
            curve = list(csv.DictReader(file))
        self.assertEqual({r["label"] for r in curve}, {"Base", "SFT"})
        self.assertAlmostEqual(float(curve[-1]["percent"]), 100 * 4 / 6)
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            plot.export_report([plot.load_job(first)], output, ["png"])

    def test_cli_validation_happens_before_output_creation(self):
        job = self.job()
        output = self.root / "should-not-exist"
        cases = [[str(job), str(job)], [str(job), "--label", "A", "--label", "B"],
                 [str(job), str(job), "--output-dir", str(output)]]
        for args in cases:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                plot.main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
