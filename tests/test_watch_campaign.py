import json
import io
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from scripts.analysis import watch_campaign as w


class WatchTests(unittest.TestCase):
    def test_partial_events_are_retried_and_rotation_resets(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            log = w.EventLog(path)
            self.assertEqual(log.read(), [])
            path.write_text('{"kind":"attempt_start"')
            self.assertEqual(log.read(), [])
            with path.open("a") as stream:
                stream.write('}\n')
            self.assertEqual(len(log.read()), 1)
            self.assertEqual(len(log.read()), 1)
            path.unlink()
            path.write_text('{"kind":"job_end"}\n')
            self.assertEqual(log.read(), [{"kind": "job_end"}])

    def test_clock_does_not_trigger_output_but_new_arm_does(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = {"id": "split-65", "model": "swift-iq4-xs", "targets": [98304],
                   "configs": [{"context": 102400, "slots": 1, "split": "65,35"}]}
            (root / "manifest.json").write_text(json.dumps({"queue": [job]}))
            state = {"status": "running", "active": job, "completed": {}}
            (root / "state.json").write_text(json.dumps(state))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            first, text = w.snapshot(root, [], now=100)
            second, _ = w.snapshot(root, [], now=110)
            self.assertEqual(first, second)
            self.assertIn("Why: compare placement", text)
            self.assertIn("split 65,35", text)
            self.assertIn("input target 98304", text)
            self.assertIn("ETA: unknown", text)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            state["status"] = "complete"
            state["completed"] = {job["id"]: "result.json"}
            (root / "state.json").write_text(json.dumps(state))
            third, text = w.snapshot(root, [], now=120)
            self.assertNotEqual(first, third)
            self.assertIn("ETA: complete", text)

    def test_eta_uses_job_wall_time_including_fit_and_labels_extrapolation(self):
        jobs = [{"id": str(i), "targets": [8192 if i < 2 else 98304], "configs": [{}]}
                for i in range(3)]
        events = [{"kind": "attempt_start", "job": "0-fit", "time": 100},
                  {"kind": "attempt_start", "job": "0", "time": 110},
                  {"kind": "job_end", "job": "0", "time": 160, "status": "passed"}]
        eta = w.estimate(jobs, {"0": "done"}, events, 160)
        self.assertIn("~2 min", eta)
        self.assertNotIn("provisional", eta)
        events.append({"kind": "attempt_start", "job": "1-fit", "time": 160})
        self.assertIn("exceeds", w.estimate(jobs, {"0": "done"}, events, 230))

    def test_failed_capacity_does_not_create_performance_eta(self):
        jobs = [{"id": "fit", "targets": [0]}, {"id": "deep", "targets": [98304]}]
        events = [{"kind": "attempt_start", "job": "fit", "time": 100},
                  {"kind": "job_end", "job": "fit", "time": 105, "status": "failed"}]
        self.assertIn("unknown", w.estimate(jobs, {"fit": "failed"}, events, 110))

    def test_prepared_path_follows_newer_running_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "campaign"
            live = root / "campaign-r3"
            old.mkdir(); live.mkdir()
            (old / "state.json").write_text(json.dumps({"status": "prepared"}))
            (live / "state.json").write_text(json.dumps({"status": "running"}))
            self.assertEqual(w.resolve_campaign(old), live)

    def test_redraw_uses_terminal_clear_and_home_only_when_requested(self):
        stream = io.StringIO()
        w.render("dashboard", redraw=True, stream=stream)
        self.assertEqual(stream.getvalue(), "\x1b[2J\x1b[Hdashboard\n")
        stream = io.StringIO()
        w.render("dashboard", redraw=False, stream=stream)
        self.assertEqual(stream.getvalue(), "dashboard\n")

    def test_redraw_bolds_the_job_counter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = {"id": "fit", "targets": [0], "configs": [{}]}
            (root / "manifest.json").write_text(json.dumps({"queue": [job]}))
            (root / "state.json").write_text(json.dumps({"status": "prepared", "completed": {}}))
            _, plain = w.snapshot(root, [], now=100)
            _, bold = w.snapshot(root, [], now=100, redraw=True)
            self.assertIn("0/1 jobs", plain)
            self.assertIn("\x1b[1m0/1\x1b[22m jobs", bold)

    def test_timer_is_combined_with_eta_at_the_bottom(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = {"id": "fit", "targets": [0], "configs": [{}]}
            (root / "manifest.json").write_text(json.dumps({"queue": [job]}))
            (root / "state.json").write_text(json.dumps({"status": "prepared", "completed": {}}))
            _, text = w.snapshot(root, [], now=100, redraw=True, elapsed_since_update=12.9)
            self.assertTrue(text.endswith("Since update: 12s | ETA: unavailable while the campaign is not running"))

    def test_display_timestamp_can_be_frozen_while_eta_clock_uses_current_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = {"id": "fit", "targets": [0], "configs": [{}]}
            (root / "manifest.json").write_text(json.dumps({"queue": [job]}))
            (root / "state.json").write_text(json.dumps({"status": "prepared", "completed": {}}))
            _, text = w.snapshot(root, [], now=200, display_time=100, elapsed_since_update=100)
            self.assertTrue(text.startswith("[" + datetime.fromtimestamp(100).strftime("%H:%M:%S") + "]"))
            self.assertIn("Since update: 100s", text)


if __name__ == "__main__":
    unittest.main()
