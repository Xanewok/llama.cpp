from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.instruments import emit


def _emit_many(path: str, prefix: str, count: int, start: object) -> None:
    context = emit.RunContext(out=path, lane="qwen", quant="Q4")
    start.wait()  # type: ignore[attr-defined]
    for index in range(count):
        emit.emit(context, f"{prefix}-{index}", 0, task_success=True)


class EmitTest(unittest.TestCase):
    def test_row_shape_is_fixed_and_missing_values_are_null(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "rows.jsonl"
            context = emit.RunContext(out=output, lane="qwen", quant="Q4")
            first = emit.emit(context, "alpha", 3, task_success=True)
            second = emit.emit(context, "beta", 3, notes="zażółć")

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(set(first), set(emit.RESULT_FIELDS))
            self.assertEqual(set(rows[0]), set(rows[1]))
            self.assertIsNone(rows[0]["decode_tps"])
            self.assertEqual(rows[1]["notes"], "zażółć")
            self.assertEqual(list(rows[0]), sorted(rows[0]))
            self.assertEqual(second["task_id"], "beta")

    def test_append_flush_fsync_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "rows.jsonl"
            context = emit.RunContext(out=output, lane="qwen", quant="Q4")
            real_fsync = os.fsync
            real_flock = emit.fcntl.flock
            with mock.patch("scripts.instruments.emit.os.fsync", wraps=real_fsync) as fsync_mock, mock.patch(
                "scripts.instruments.emit.fcntl.flock", wraps=real_flock
            ) as flock_mock:
                emit.emit(context, "one", 0)
                emit.emit(context, "two", 0)

            self.assertEqual(fsync_mock.call_count, 2)
            self.assertEqual(flock_mock.call_count, 4)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_flock_contention_between_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = str(Path(temp_dir) / "rows.jsonl")
            start = multiprocessing.Event()
            processes = [
                multiprocessing.Process(target=_emit_many, args=(output, prefix, 40, start))
                for prefix in ("left", "right")
            ]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(15)
                self.assertEqual(process.exitcode, 0)

            rows = [json.loads(line) for line in Path(output).read_text().splitlines()]
            self.assertEqual(len(rows), 80)
            self.assertEqual(len({row["task_id"] for row in rows}), 80)

    def test_metrics_parser_tolerates_labels_and_absence(self) -> None:
        body = b"""# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds{model=\"qwen\"} 123.5
llamacpp:predicted_tokens_seconds 45
"""
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = body
        with mock.patch("scripts.instruments.emit.urllib.request.urlopen", return_value=response):
            self.assertEqual(
                emit.scrape_metrics("http://127.0.0.1:8000/v1"),
                {"prefill_tps": 123.5, "decode_tps": 45.0},
            )
        with mock.patch("scripts.instruments.emit.urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertEqual(emit.scrape_metrics("http://127.0.0.1:8000/v1"), {})

    def test_identity_fields_are_required(self) -> None:
        context = emit.RunContext(out="unused", lane="qwen", quant="Q4")
        with self.assertRaises(ValueError):
            emit.build_row(context, "", 0)
        with self.assertRaises(ValueError):
            emit.build_row(context, "task", None)  # type: ignore[arg-type]

    def test_lane_selects_one_codex_invocation_path(self) -> None:
        qwen = emit.codex_command("qwen", "medium")
        control = emit.codex_command("control", "high")
        frontier = emit.codex_command("frontier", "high")

        for command in (qwen, control, frontier):
            self.assertEqual(command[-1], "-", "prompt must arrive on stdin")

        # Local arms select by profile, never by -m, so the served alias stays uniform.
        self.assertIn("qwen38", qwen)
        self.assertNotIn("-m", qwen)

        # control is C2, the direct competitor; frontier is C3, the ceiling.
        self.assertIn("gpt-5.6-luna", control)
        self.assertIn("gpt-5.6-sol", frontier)
        self.assertNotEqual(control, frontier, "external lanes must not collapse to one model")

        for command in (control, frontier):
            self.assertNotIn("qwen38", command)

    def test_unknown_lane_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            emit.codex_command("local", "high")

    def test_recorded_model_matches_the_invoked_model(self) -> None:
        # A row that names a model the lane did not actually invoke corrupts the
        # analysis silently, so the mapping must have exactly one source.
        # Parse real arguments rather than hand-building a Namespace: a hand-built
        # one silently lacks each newly added context flag, so this test broke on the
        # addition of --engine instead of catching anything.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = Path(tmp.name) / "env.json"
        env.write_text(json.dumps({"gpu": "NVIDIA GeForce RTX 3090", "gpu_count": 1}))
        for lane in ("qwen", "quant", "control", "frontier"):
            parser = argparse.ArgumentParser()
            emit.add_context_arguments(parser)
            args = parser.parse_args(
                ["--lane", lane, "--quant", "Q4", "--run-index", "0", "--out", "unused",
                 "--env-json", str(env)]
            )
            context = emit.make_context(args)
            command = emit.codex_command(lane, "xhigh")
            if lane in {"qwen", "quant"}:
                self.assertEqual(context.model, "qwen3.8-27b")
            else:
                self.assertIn(context.model, command)


if __name__ == "__main__":
    unittest.main()
