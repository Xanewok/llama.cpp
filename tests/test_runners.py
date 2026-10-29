from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.instruments import emit, run_polyglot
from scripts.instruments import run_swebench


class RunnerTest(unittest.TestCase):
    def test_polyglot_manifest_order_and_crash_isolation(self) -> None:
        manifest = ROOT / "tests" / "fixtures" / "polyglot_manifest.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.jsonl"
            returncode = run_polyglot.main(
                [
                    "--manifest",
                    str(manifest),
                    "--lane",
                    "qwen",
                    "--quant",
                    "fixture",
                    "--run-index",
                    "7",
                    "--out",
                    str(output),
                    "--jobs",
                    "3",
                    "--timeout",
                    "30",
                    "--dry-run",
                    "--skip-selftest",
                ]
            )

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(returncode, 0)
            self.assertEqual(
                [row["task_id"] for row in rows],
                ["fixture-first", "fixture-crash", "fixture-last"],
            )
            self.assertTrue(rows[0]["task_success"])
            self.assertFalse(rows[1]["task_success"])
            self.assertIn("FileNotFoundError", rows[1]["notes"])
            self.assertTrue(rows[2]["task_success"])
            self.assertTrue(all(row["harness_verified"] is False for row in rows))
            self.assertTrue(all(set(row) == set(emit.RESULT_FIELDS) for row in rows))

    def test_polyglot_refuses_to_run_without_selftest(self) -> None:
        manifest = ROOT / "tests" / "fixtures" / "polyglot_manifest.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.jsonl"
            with self.assertRaisesRegex(SystemExit, "SELF-TEST REQUIRED"):
                run_polyglot.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--lane",
                        "qwen",
                        "--quant",
                        "fixture",
                        "--run-index",
                        "0",
                        "--out",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertFalse(output.exists())

    def test_swebench_dry_run_emits_empty_prediction(self) -> None:
        instances = ROOT / "tests" / "fixtures" / "swebench_instances.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.jsonl"
            predictions = Path(temp_dir) / "predictions.jsonl"
            cache = Path(temp_dir) / "cache"
            returncode = run_swebench.main(
                [
                    "--instances",
                    str(instances),
                    "--lane",
                    "qwen",
                    "--quant",
                    "fixture",
                    "--run-index",
                    "2",
                    "--out",
                    str(output),
                    "--predictions",
                    str(predictions),
                    "--repo-cache",
                    str(cache),
                    "--timeout",
                    "30",
                    "--dry-run",
                ]
            )

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            prediction_rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            self.assertEqual(returncode, 0)
            self.assertEqual([row["task_id"] for row in rows], ["fixture__repo-1"])
            self.assertFalse(rows[0]["task_success"])
            self.assertIn("empty diff", rows[0]["notes"])
            self.assertEqual(prediction_rows[0]["model_patch"], "")
            self.assertEqual(
                set(prediction_rows[0]),
                {"instance_id", "model_name_or_path", "model_patch"},
            )


if __name__ == "__main__":
    unittest.main()


class ExternalLaneTelemetryTest(unittest.TestCase):
    def test_external_lanes_do_not_scrape_llama_server(self) -> None:
        """A luna/sol task must never inherit the local server's counters.

        The counters are global to the llama-server process, so scraping them for
        an API-model lane would silently attribute Qwen's prefill and cache reads
        to the control arm.
        """

        from scripts.instruments import run_polyglot

        calls: list[str] = []
        original = run_polyglot.scrape_metrics
        run_polyglot.scrape_metrics = lambda url, timeout=5.0: calls.append(url) or {}
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = root / "m.json"
                manifest.write_text(json.dumps([]))
                for lane in ("control", "frontier"):
                    run_polyglot.main(
                        [
                            "--manifest", str(manifest),
                            "--skip-selftest",
                            "--lane", lane,
                            "--quant", "none",
                            "--run-index", "1",
                            "--out", str(root / f"{lane}.jsonl"),
                            "--metrics-url", "http://127.0.0.1:8000/v1",
                        ]
                    )
        finally:
            run_polyglot.scrape_metrics = original
        self.assertEqual(calls, [], f"external lanes scraped llama-server: {calls}")


class DependencyInjectionTest(unittest.TestCase):
    def test_repeated_preparation_does_not_duplicate_keys(self) -> None:
        """prepare_test_harness runs once per test cycle, so it must be idempotent.

        A repeated key makes cargo reject the manifest before compiling anything,
        which scores the task failed however good the model's solution is.
        """

        from scripts.instruments import run_polyglot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "ex"
            workdir.mkdir()
            (workdir / "Cargo.toml").write_text(
                '[package]\nname = "decimal"\nversion = "0.1.0"\n\n[dependencies]\n'
            )
            task = {"task_id": "rust/decimal", "language": "rust"}
            for _ in range(3):
                run_polyglot.prepare_test_harness(task, workdir)
            text = (workdir / "Cargo.toml").read_text()
            for name in ("num-bigint", "num-traits"):
                self.assertEqual(
                    text.count(f"{name} ="), 1, f"{name} declared more than once:\n{text}"
                )


class ResumeTest(unittest.TestCase):
    def _write(self, path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_only_this_legs_rows_count_as_done(self) -> None:
        """One file holds every leg, so matching on task_id alone would let a
        different quant's result mark this leg's task complete."""

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            self._write(out, [
                {"task_id": "a", "quant": "Q4", "run_index": 1},
                {"task_id": "b", "quant": "IQ3", "run_index": 1},
                {"task_id": "c", "quant": "Q4", "run_index": 2},
            ])
            self.assertEqual(run_polyglot.completed_task_ids(out, "Q4", 1), {"a"})

    def test_a_torn_final_line_does_not_abort_resume(self) -> None:
        # A killed writer can leave half a row; that must cost one task, not the run.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            out.write_text(json.dumps({"task_id": "a", "quant": "Q4", "run_index": 1})
                           + "\n" + '{"task_id": "b", "quan')
            self.assertEqual(run_polyglot.completed_task_ids(out, "Q4", 1), {"a"})

    def test_missing_file_is_empty_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                run_polyglot.completed_task_ids(Path(tmp) / "nope.jsonl", "Q4", 1), set()
            )


class MergedStatsReachTheRowTest(unittest.TestCase):
    """Every field CodexStats parses has to survive the merge into the row.

    cache_write_tokens was parsed and then dropped here, so the column read null for
    every leg -- and cache writes are billable on some API routes, which makes a silent
    null an understated cost rather than a missing detail.
    """

    def test_no_parsed_field_is_dropped(self) -> None:
        fields = set(emit.CodexStats.__dataclass_fields__)
        merged = set(run_polyglot._merge_stats([]))
        self.assertEqual(fields - merged, set(), "parsed but never written to the row")

    def test_cache_writes_are_summed_across_turns(self) -> None:
        stats = [emit.CodexStats(agent_turns=1, tool_calls=0, prompt_tokens=10,
                                 completion_tokens=5, cache_write_tokens=n)
                 for n in (7, 11)]
        self.assertEqual(run_polyglot._merge_stats(stats)["cache_write_tokens"], 18)
