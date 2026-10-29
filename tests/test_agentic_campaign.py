from collections import Counter
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.instruments import agentic_campaign as a
from scripts.instruments import hillclimb as h


class AgenticQueueTests(unittest.TestCase):
    def test_all_eight_tasks_cover_control_and_five_quants_task_first(self):
        jobs = h.validate_profiles(a.task_queue())
        self.assertEqual(len(jobs), 48)
        self.assertEqual(Counter(job["model"] for job in jobs), dict.fromkeys(a.MODEL_ORDER, 8))
        self.assertEqual([job["model"] for job in jobs[:6]], a.MODEL_ORDER)
        self.assertEqual({job["task"] for job in jobs[:6]}, {a.TASK_ORDER[0]})
        for job in jobs:
            self.assertEqual(job["workload"], "agentic")
            self.assertEqual(job["configs"][0]["slots"], 1)
            self.assertEqual(job["configs"][0]["maximum"], 3)
            self.assertEqual(job["configs"][0]["context"], 65536 if job["model"] == "iq3-control" else 102400)

    def test_predecessor_safety_stop_does_not_admit_task_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            campaign, predecessor = Path(temp) / "next", Path(temp) / "previous"
            campaign.mkdir()
            predecessor.mkdir()
            h.atomic_json(predecessor / "state.json", {"status": "stopped", "reason": "thermal_limit", "completed": {}})
            self.assertFalse(a.wait_for_predecessor(campaign, predecessor, time.time() + 60))
            self.assertEqual(json.loads((campaign / "stage-status.json").read_text())["phase"], "blocked")
            h.atomic_json(predecessor / "state.json", {"status": "complete", "completed": {}})
            self.assertTrue(a.wait_for_predecessor(campaign, predecessor, time.time() + 60))

    def test_control_is_single_gpu_and_failed_fit_skips_coding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            h.sealed_json(root / "manifest.json", {"port": 8081, "queue": a.task_queue(), "max_attempts": 3})
            h.atomic_json(root / "state.json", {"status": "prepared", "completed": {}, "owned": []})
            runner = a.AgenticRunner(root, 1)
            with patch.object(runner, "ensure_model", return_value={"id": "iq3-control"}), \
                    patch.object(runner, "attempt", return_value={"status": "failed", "reason": "oom"}) as attempt:
                result = runner.execute_job(a.task_queue()[0])
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(attempt.call_count, 1)
            config = attempt.call_args.args[2]
            self.assertEqual(config["devices"], [h.GPU0])
            self.assertEqual(config["mode"], "layer")


if __name__ == "__main__":
    unittest.main()
