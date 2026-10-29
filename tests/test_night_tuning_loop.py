import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.analysis import night_tuning_loop as loop


class NightLoopTests(unittest.TestCase):
    def test_deadline_rolls_to_next_morning(self):
        now = datetime(2026, 9, 15, 23, 0, tzinfo=timezone.utc)
        self.assertEqual(loop.morning_deadline(now).isoformat(),
                         "2026-09-16T08:00:00+00:00")

    def test_snapshot_marks_running_campaign_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign-swift-tuning-x" / "state.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"status": "running", "completed": {}}))
            with mock.patch.object(loop, "units", return_value=[]), \
                    mock.patch.object(loop, "gpu_processes", return_value=[]):
                self.assertEqual(loop.snapshot(Path(tmp))["reason"], "campaign_running")

    def test_snapshot_is_idle_without_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(loop, "units", return_value=[]), \
                    mock.patch.object(loop, "gpu_processes", return_value=[]):
                self.assertEqual(loop.snapshot(Path(tmp))["reason"], "nothing_to_do")


if __name__ == "__main__":
    unittest.main()
