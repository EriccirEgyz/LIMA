"""Run with: python -B -m unittest discover -s src/select -p 'test_mix_sft_by_trajectory.py' -v"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import filter_sft_by_length as length_filter
import mix_sft_by_trajectory as mix


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_pool(self, name, sizes=(2, 0, 1, 4, 3), origin=None):
        origin = origin or name
        path = self.root / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        samples, rows = [], []
        for task_index, size in enumerate(sizes):
            for step in range(size):
                samples.append({
                    "instruction": f"{origin} task {task_index} step {step}", "input": "",
                    "output": '<think>检查</think>\n\n<tool_call>{"name":"lookup"}</tool_call>',
                    "system": "保留原文 🧪", "history": [["question", "prior answer"]] * step,
                })
            rows.append({
                "source": f"/trajectories/{origin}", "task_index": task_index, "sample_idx": 1,
                "global_id": task_index + 100, "kept": size > 0, "n_samples": size,
                "reason": "ok" if size else "all_samples_over_cutoff",
                "hidden_turns": [{"turn": 0, "reason": "preserved audit"}],
            })
        path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
        self.write_manifest(path, rows)
        return path, samples, rows

    def write_manifest(self, path, rows):
        path.with_suffix(".manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def sources(self, *items):
        return mix.parse_sources([(str(path), str(count)) for path, count in items])

    def run_cli(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = mix.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_exact_quotas_include_every_slice_and_preserve_audits(self):
        a, a_samples, a_rows = self.make_pool("a")
        b, b_samples, b_rows = self.make_pool("b", (5, 1, 0, 2))
        sources = self.sources((a, 3), (b, 2))
        samples, rows, report = mix.prepare_mixture(sources, 42)
        self.assertEqual(report["total_trajectories"], 5)
        self.assertEqual([r["selected_trajectories"] for r in report["sources"]], [3, 2])
        original = {"a": (a_samples, a_rows), "b": (b_samples, b_rows)}
        identities = set()
        for row in rows:
            event = row["trajectory_mixing"][-1]
            start, end = event["source_index_range"]
            out_start, out_end = event["output_index_range"]
            source_samples, source_rows = original[event["source_name"]]
            self.assertEqual(samples[out_start:out_end], source_samples[start:end])
            self.assertEqual(row["n_samples"], end - start)
            self.assertEqual({k: v for k, v in row.items() if k != "trajectory_mixing"},
                             source_rows[event["source_manifest_index"]])
            identities.add((row["source"], row["task_index"], row["sample_idx"]))
        self.assertEqual(len(identities), 5)
        self.assertEqual(sum(row["n_samples"] for row in rows), len(samples))
        self.assertNotIn("_meta", samples[0])
        for summary, path in zip(report["sources"], (a, b)):
            self.assertEqual(summary["dataset_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_equal_quota_selects_all_available_groups(self):
        path, original, _ = self.make_pool("a")
        samples, rows, report = mix.prepare_mixture(self.sources((path, 4)), 9)
        self.assertEqual(samples, original)
        self.assertEqual(len(rows), 4)
        self.assertEqual(report["sources"][0]["trajectory_fraction"], 1)

    def test_seed_is_reproducible_and_does_not_modify_global_rng(self):
        path, _, _ = self.make_pool("a", (1,) * 30)
        sources = self.sources((path, 8))
        state = mix.random.getstate()
        first = mix.prepare_mixture(sources, 42)
        self.assertEqual(first, mix.prepare_mixture(sources, 42))
        self.assertNotEqual(first[1], mix.prepare_mixture(sources, 99)[1])
        self.assertEqual(state, mix.random.getstate())

    def test_selections_are_nested_and_independent_of_other_sources(self):
        a, _, _ = self.make_pool("a", (1,) * 20)
        b, _, _ = self.make_pool("b", (1,) * 20)
        small = mix.prepare_mixture(self.sources((a, 4)), 42)[2]
        large = mix.prepare_mixture(self.sources((a, 7)), 42)[2]
        reversed_sources = mix.prepare_mixture(self.sources((b, 5), (a, 4)), 42)[2]
        ids = lambda summary: {r["task_index"] for r in summary["selected"]}
        self.assertLess(ids(small["sources"][0]), ids(large["sources"][0]))
        self.assertEqual(ids(small["sources"][0]), ids(reversed_sources["sources"][1]))

    def test_renamed_dataset_and_directory_do_not_change_selection(self):
        path, _, _ = self.make_pool("original", (1,) * 30)
        renamed = self.root / "different_directory" / "different_name.json"
        renamed.parent.mkdir()
        renamed.write_bytes(path.read_bytes())
        renamed.with_suffix(".manifest.jsonl").write_bytes(path.with_suffix(".manifest.jsonl").read_bytes())
        first, first_rows, first_report = mix.prepare_mixture(self.sources((path, 8)), 42)
        second, second_rows, second_report = mix.prepare_mixture(self.sources((renamed, 8)), 42)
        self.assertEqual(first, second)
        self.assertEqual([r["task_index"] for r in first_rows], [r["task_index"] for r in second_rows])
        self.assertNotEqual(first_report["sources"][0]["name"], second_report["sources"][0]["name"])

    def test_equal_basenames_are_disambiguated_without_affecting_selection(self):
        a, _, _ = self.make_pool("left/shared", (1,) * 20)
        b, _, _ = self.make_pool("right/shared", (1,) * 20)
        alone = mix.prepare_mixture(self.sources((a, 4)), 42)[2]
        _, rows, together = mix.prepare_mixture(self.sources((a, 4), (b, 3)), 42)
        self.assertEqual(alone["sources"][0]["name"], "shared")
        self.assertEqual([s["name"] for s in together["sources"]], [str(a), str(b)])
        self.assertEqual([s["task_index"] for s in alone["sources"][0]["selected"]],
                         [s["task_index"] for s in together["sources"][0]["selected"]])
        self.assertEqual(len(rows), 7)

    def test_shortages_are_aggregated_without_creating_outputs(self):
        a, _, _ = self.make_pool("a", (1, 0))
        b, _, _ = self.make_pool("b", (0,))
        output = self.root / "new_directory" / "mix.json"
        code, _, error = self.run_cli([
            "--source", str(a), "3", "--source", str(b), "2", "--out", str(output),
        ])
        self.assertEqual(code, 2)
        self.assertIn("a: 请求 3 条，清洗后可用 1 条，缺 2 条", error)
        self.assertIn("b: 请求 2 条，清洗后可用 0 条，缺 2 条", error)
        self.assertFalse(output.parent.exists())

    def test_invalid_quotas_and_duplicate_files(self):
        path, _, _ = self.make_pool("a")
        for count in ("0", "-1", "1.5", "all"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "正整数"):
                mix.parse_sources([(str(path), count)])
        with self.assertRaisesRegex(ValueError, "重复的输入"):
            self.sources((path, 1), (path, 1))
        alias = self.root / "alias.json"
        alias.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "重复的输入"):
            self.sources((path, 1), (alias, 1))
        hardlink = self.root / "hardlink.json"
        hardlink.hardlink_to(path)
        with self.assertRaisesRegex(ValueError, "重复的输入"):
            self.sources((path, 1), (hardlink, 1))

    def test_missing_manifest_is_an_error(self):
        path = self.root / "bare.json"
        path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "缺少配套 manifest"):
            self.sources((path, 1))

    def test_bad_manifest_counts_and_identity_fields(self):
        path, _, rows = self.make_pool("a", (1,))
        cases = [
            {"n_samples": True}, {"n_samples": -1}, {"n_samples": 1.0}, {"kept": 1},
            {"kept": False}, {"task_index": None}, {"sample_idx": True}, {"source": ""},
        ]
        source = self.sources((path, 1))[0]
        for bad in cases:
            with self.subTest(bad=bad):
                self.write_manifest(path, [{**rows[0], **bad}])
                with self.assertRaises(ValueError):
                    mix.load_pool(source)

    def test_duplicate_manifest_identities_are_rejected(self):
        path, _, rows = self.make_pool("a", (1,))
        self.write_manifest(path, rows * 2)
        with self.assertRaisesRegex(ValueError, "重复记录同一轨迹"):
            mix.prepare_mixture(self.sources((path, 1)), 42)

    def test_same_task_id_from_different_origins_is_allowed(self):
        a, _, _ = self.make_pool("a", (1,))
        b, _, _ = self.make_pool("b", (1,))
        _, rows, _ = mix.prepare_mixture(self.sources((a, 1), (b, 1)), 42)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["global_id"], rows[1]["global_id"])

    def test_overlap_in_selected_original_trajectories_is_rejected(self):
        a, _, _ = self.make_pool("a", (1,), origin="shared")
        b, _, _ = self.make_pool("b", (1,), origin="shared")
        with self.assertRaisesRegex(ValueError, "重复选中同一原始轨迹"):
            mix.prepare_mixture(self.sources((a, 1), (b, 1)), 42)

    def test_copied_dataset_with_different_manifest_is_rejected(self):
        a, _, _ = self.make_pool("a", (1,))
        b, _, _ = self.make_pool("b", (1,))
        b.write_bytes(a.read_bytes())
        with self.assertRaisesRegex(ValueError, "文件内容完全相同"):
            mix.prepare_mixture(self.sources((a, 1), (b, 1)), 42)

    def test_count_mismatch_does_not_write_partial_output(self):
        a, _, _ = self.make_pool("a", (1,))
        b, _, rows = self.make_pool("b", (1,))
        rows[0]["n_samples"] = 2
        self.write_manifest(b, rows)
        output = self.root / "not_created" / "mix.json"
        code, _, error = self.run_cli([
            "--source", str(a), "1", "--source", str(b), "1", "--out", str(output),
        ])
        self.assertEqual(code, 2)
        self.assertIn("manifest 覆盖 2 个切片，SFT 实际有 1 个", error)
        self.assertFalse(output.parent.exists())

    def test_even_unselected_samples_are_validated(self):
        path, samples, _ = self.make_pool("a", (1,) * 10)
        source = self.sources((path, 1))[0]
        selected = mix.select_groups(mix.load_pool(source), 42)[0].start
        samples[(selected + 1) % len(samples)]["history"] = ["invalid"]
        path.write_text(json.dumps(samples))
        with self.assertRaisesRegex(ValueError, "history"):
            mix.prepare_mixture([source], 42)

    def test_streaming_json_handles_tiny_chunks_unicode_and_hashes(self):
        path, original, _ = self.make_pool("a")
        raw = (" \r\n" + json.dumps(original, ensure_ascii=False, indent=2) + "\r\n").encode("utf-8")
        path.write_bytes(raw)
        for chunk_size in (1, 7, 31, 1024):
            hasher = hashlib.sha256()
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(list(mix.iter_samples(path, hasher, chunk_size)), original)
                self.assertEqual(hasher.hexdigest(), hashlib.sha256(raw).hexdigest())

    def test_streaming_json_rejects_malformed_arrays(self):
        path = self.root / "malformed.json"
        invalid = ["", "{}", "[", "[{}", "[{},]", "[{},", "[{} {}]", "[]x", "[1]", '[{"x":NaN}]']
        for raw in invalid:
            with self.subTest(raw=raw):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    list(mix.iter_samples(path, hashlib.sha256(), 2))

    def test_dry_run_checks_data_but_creates_no_directory(self):
        path, _, _ = self.make_pool("a")
        output = self.root / "new_directory" / "mix.json"
        code, stdout, stderr = self.run_cli([
            "--source", str(path), "2", "--out", str(output), "--dry-run",
        ])
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("[dry-run]", stdout)
        self.assertFalse(output.parent.exists())

    def test_outputs_roundtrip_and_remain_compatible_with_length_filter(self):
        a, _, _ = self.make_pool("a")
        b, _, _ = self.make_pool("b")
        output = self.root / "output" / "mix.json"
        code, _, stderr = self.run_cli([
            "--source", str(a), "3", "--source", str(b), "2", "--out", str(output),
        ])
        self.assertEqual((code, stderr), (0, ""))
        samples = json.loads(output.read_text())
        rows = [json.loads(line) for line in output.with_suffix(".manifest.jsonl").read_text().splitlines()]
        summary = json.loads(output.with_suffix(".mixture.json").read_text())
        owners = length_filter.manifest_owners(rows, len(samples))
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(owners), len(samples))
        self.assertEqual(summary["total_samples"], len(samples))
        for index, row in enumerate(rows):
            start, end = row["trajectory_mixing"][-1]["output_index_range"]
            self.assertEqual(owners[start:end], [index] * (end - start))
        # A mixed output can itself be sampled without losing original audits.
        again, again_rows, _ = mix.prepare_mixture(self.sources((output, 5)), 42)
        self.assertEqual(again, samples)
        self.assertTrue(all(len(row["trajectory_mixing"]) == 2 for row in again_rows))

    def test_each_existing_sidecar_blocks_all_writes(self):
        path, _, _ = self.make_pool("a")
        for suffix in (".json", ".manifest.jsonl", ".mixture.json"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory(dir=self.root) as directory:
                output = Path(directory) / "mix.json"
                existing = output.with_suffix(suffix)
                existing.write_text("do not overwrite")
                code, _, error = self.run_cli([
                    "--source", str(path), "1", "--out", str(output),
                ])
                self.assertEqual(code, 2)
                self.assertIn("拒绝覆盖", error)
                self.assertEqual(existing.read_text(), "do not overwrite")
                self.assertEqual(list(Path(directory).iterdir()), [existing])

    def test_output_cannot_replace_input_or_follow_dangling_symlink(self):
        path, _, _ = self.make_pool("a")
        sources = self.sources((path, 1))
        with self.assertRaisesRegex(ValueError, "不能替换输入"):
            mix.output_paths(str(path), sources)
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.root / "absent.json")
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            mix.output_paths(str(dangling), sources)

    def test_failed_write_cleans_only_outputs_created_by_this_call(self):
        path, _, _ = self.make_pool("a", (1,))
        sources = self.sources((path, 1))
        samples, manifests, report = mix.prepare_mixture(sources, 42)
        paths = mix.output_paths(str(self.root / "mixed.json"), sources)
        with patch.object(mix.json, "dump", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                mix.write_outputs(paths, samples, manifests, report)
        self.assertTrue(all(not p.exists() for p in paths))
        # Simulate another process creating a sidecar after preflight.
        paths[1].write_text("other process")
        with self.assertRaises(FileExistsError):
            mix.write_outputs(paths, samples, manifests, report)
        self.assertEqual(paths[1].read_text(), "other process")
        self.assertFalse(paths[0].exists())
        self.assertFalse(paths[2].exists())


if __name__ == "__main__":
    unittest.main()
