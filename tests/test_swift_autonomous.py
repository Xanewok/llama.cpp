import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.instruments import swift_autonomous as auto


class AutonomousTests(unittest.TestCase):
    def test_stop_interactive_accepts_absent_unit_but_not_active_stop_failure(self):
        with patch.object(auto, "run_command", side_effect=[
                auto.subprocess.CompletedProcess([], 1, "", "unit not found"),
                auto.subprocess.CompletedProcess([], 3, "inactive\n", "")]):
            rows = []
            auto.stop_interactive(rows.append)
            self.assertEqual(rows[0]["reason"], "interactive_already_stopped")
        with patch.object(auto, "run_command", side_effect=[
                auto.subprocess.CompletedProcess([], 1, "", "stop failed"),
                auto.subprocess.CompletedProcess([], 0, "active\n", "")]):
            with self.assertRaisesRegex(RuntimeError, "interactive stop failed"):
                auto.stop_interactive(lambda _: None)

    def test_falls_back_when_candidate_lacks_full_coverage(self):
        base = {"batch": 2048, "split": "71,29", "ubatch": 128}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempts" / "a"
            root.mkdir(parents=True)
            (root / "result.json").write_text(json.dumps({
                "status": "passed", "job": "batch-batch-1024-x",
                "target": 49152, "task": "javascript/affine-cipher",
                "decode_tps": 50, "config": {**base, "batch": 1024}}))
            selected, evidence = auto.select_config(Path(tmp), base, "batch")
            self.assertEqual(selected, base)
            self.assertEqual(evidence["decision"], "retain base; no fully covered passing candidate")

    def test_selects_fully_covered_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage = [
                ("javascript/affine-cipher", 49152), ("rust/forth", 49152),
                ("javascript/affine-cipher", 98304), ("rust/forth", 98304)]
            for i, (task, target) in enumerate(coverage):
                root = Path(tmp) / "attempts" / f"base-{i}"
                root.mkdir(parents=True)
                (root / "result.json").write_text(json.dumps({
                    "status": "passed", "job": f"batch-matched-control-{i}",
                    "target": target, "task": task,
                    "decode_tps": 40, "config": {"batch": 2048}}))
            for i, (task, target) in enumerate(coverage):
                root = Path(tmp) / "attempts" / str(i)
                root.mkdir(parents=True)
                (root / "result.json").write_text(json.dumps({
                    "status": "passed", "job": f"batch-batch-1024-{i}",
                    "target": target, "task": task,
                    "decode_tps": 50 + i, "config": {"batch": 1024,
                                                       "devices": ["runtime-only"]}}))
            selected, evidence = auto.select_config(Path(tmp), {"batch": 2048}, "batch")
            self.assertEqual(selected["batch"], 1024)
            self.assertNotIn("devices", selected)
            self.assertEqual(evidence["coverage_rows"], 4)

    def test_rejects_candidate_without_five_percent_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage = [
                ("javascript/affine-cipher", 49152), ("rust/forth", 49152),
                ("javascript/affine-cipher", 98304), ("rust/forth", 98304)]
            for i, (task, target) in enumerate(coverage):
                for label, batch, score in (("matched-control", 2048, 40),
                                             ("batch-1024", 1024, 41)):
                    root = Path(tmp) / "attempts" / f"{label}-{i}"
                    root.mkdir(parents=True)
                    (root / "result.json").write_text(json.dumps({
                        "status": "passed", "job": f"batch-{label}-{i}",
                        "target": target, "task": task,
                        "decode_tps": score, "config": {"batch": batch}}))
            selected, evidence = auto.select_config(Path(tmp), {"batch": 2048}, "batch")
            self.assertEqual(selected, {"batch": 2048})
            self.assertIn("at least 5%", evidence["decision"])


if __name__ == "__main__":
    unittest.main()
