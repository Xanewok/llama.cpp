#!/usr/bin/env python3
"""Discriminating CPU checks for output extraction, counters and fail-closed reuse."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import coverage
import summarize
import weekend
import weekend_common as w


class Contracts(unittest.TestCase):
    def test_assistant_only_and_deduplication(self):
        with tempfile.TemporaryDirectory(dir=w.ROOT / "build-tmp") as directory:
            path = Path(directory) / "rollout.jsonl"
            payloads = [
                {"type": "message", "role": "user", "content": [{"text": "excluded"}]},
                {"id": "r", "type": "reasoning", "content": [{"text": "think"}], "summary": [{"text": "duplicate"}]},
                {"id": "r", "type": "reasoning", "content": [{"text": "think"}]},
                {"type": "message", "role": "assistant", "content": [{"text": "answer"}]},
                {"type": "function_call", "arguments": "{\"cmd\":\"true\"}"},
                {"type": "function_call_output", "output": "excluded"},
            ]
            path.write_text("\n".join(json.dumps({"type": "response_item", "payload": p}) for p in payloads))
            self.assertEqual(list(coverage.items(path)), [("reasoning", "think"), ("prose", "answer"), ("tool-call args", '{"cmd":"true"}')])

    def test_counter_rounds_and_negative_control(self):
        with tempfile.TemporaryDirectory(dir=w.ROOT / "build-tmp") as directory:
            root = Path(directory)
            (root / "weekend.json").write_text(json.dumps(weekend.identity(1, "S4", 1, "A", w.MODEL_A, w.SHA_A)))
            (root / "SUMMARY.json").write_text(json.dumps({"state": "complete", "tasks": 8, "solved": 6, "wall_seconds": 10,
                                                           "output_tokens": 160, "decode_seconds": 8, "peak_context_tokens": 9999}))
            for i in range(8):
                (root / f"job-{i}.json").write_text(json.dumps({"status": "passed", "task": str(i), "probe": {"metrics_delta":
                    {"drafts": 20, "draft_tokens": 80, "draft_accepted_tokens": 40}}}))
            row = summarize.campaign(root)
            self.assertEqual((row["round_ms"], row["tokens_per_round"], row["acceptance"], row["decode_tps"]), (50, 3, .5, 20))
            # Dividing draft_tokens by k is wrong when draft lengths vary.
            job = json.loads((root / "job-0.json").read_text())
            job["probe"]["metrics_delta"]["draft_tokens"] = 1
            (root / "job-0.json").write_text(json.dumps(job))
            self.assertEqual(summarize.campaign(root)["round_ms"], 50)
            job["status"] = "failed"
            (root / "job-0.json").write_text(json.dumps(job))
            self.assertFalse(summarize.campaign(root)["eligible"])

    def test_winner_requires_complete_balanced_passes(self):
        data = [{"phase": 1, "arm": arm, "eligible": True, "decode_tps": value}
                for _ in range(3) for arm, value in (("S0", 50), ("S3", 60), ("S4", 70), ("S5", 65))]
        self.assertEqual(summarize.winner(data)[0], "S4")
        data[0]["eligible"] = False
        self.assertEqual(summarize.winner(data)[0], "S3")
        self.assertEqual(summarize.winner(data, "S5")[0], "S5")

    def test_budget_and_dry_gpu_barrier(self):
        self.assertTrue(weekend.enough(10000, 6400))
        self.assertFalse(weekend.enough(10000, 6401))
        self.assertFalse(weekend.enough(10000, 10001))
        with patch.dict(weekend.os.environ, {"DRY_RUN": "1", "RUN_ID": "contract-test"}), \
             patch.object(weekend, "preflight", return_value=(w.MODEL_A, w.SHA_A, {})), \
             patch.object(weekend, "log"), \
             patch.object(weekend, "busy", side_effect=AssertionError("GPU read in dry mode")), \
             patch.object(weekend, "guard", side_effect=AssertionError("NVML in dry mode")), \
             patch.object(weekend, "xid", side_effect=AssertionError("journal in dry mode")), \
             patch.object(weekend.subprocess, "Popen", side_effect=AssertionError("process in dry mode")):
            weekend.main()

    def test_kernel_fault_checks_fail_closed(self):
        with patch.object(weekend.subprocess, "check_output", return_value="kernel boot ok"):
            weekend.xid()
        for journal in ("", "kernel: Xid 79", "GPU has fallen off the bus"):
            with patch.object(weekend.subprocess, "check_output", return_value=journal):
                with self.assertRaises(RuntimeError):
                    weekend.xid()

    def test_exact_queue_k_and_foreign_reuse(self):
        h, s, queue = w.configure("S5", w.MODEL_A, w.SHA_A)
        self.assertEqual(len(queue), 8)
        self.assertTrue(all(j["configs"][0]["maximum"] == 5 for j in queue))
        self.assertNotIn("--spec-draft-n-max", h.SERVER_EXTRA)
        self.assertIn("--spec-draft-backend-sampling", h.SERVER_EXTRA)
        self.assertIn("--backend-sampling", h.SERVER_EXTRA)
        self.assertIs(s.queue.__globals__["_queue"], w._BASE_QUEUE)
        for arm in ("S0", "S3", "S4", "S5"):
            _, s, queue = w.configure(arm, w.MODEL_A, w.SHA_A)
            self.assertIs(s.queue.__globals__["_queue"], w._BASE_QUEUE)
            self.assertTrue(all(j["configs"][0]["maximum"] == w.ARMS[arm] for j in queue))
        with tempfile.TemporaryDirectory(dir=w.ROOT / "build-tmp") as directory:
            path = Path(directory)
            (path / "SUMMARY.json").write_text('{"state":"complete","tasks":8}')
            (path / "manifest.json").write_text(json.dumps({"engine_commit": "wrong"}))
            with self.assertRaises(AssertionError):
                w.check_campaign(path, "S5", w.MODEL_A, w.SHA_A)

    def test_real_runner_manifest_and_reuse_negative_controls(self):
        h, s, queue = w.configure("S4", w.MODEL_A, w.SHA_A)
        with tempfile.TemporaryDirectory(dir=w.ROOT / "build-tmp") as directory:
            path = Path(directory) / "campaign"
            h.initialize(path, w.ROOT / "build-tmp/models-unused", profiles=queue, max_attempts=3, agentic_assets=w.ASSETS)
            (path / "SUMMARY.json").write_text('{"state":"complete","tasks":8}')
            for i in range(8):
                (path / f"job-{i}.json").write_text('{"status":"passed"}')
            attempt = path / "attempts/test"
            attempt.mkdir()
            command = [str(w.FROZEN / "build/bin/llama-server"), "-ctk", "q8_0", "-ctv", "q8_0", "--spec-draft-n-max", "4", *h.SERVER_EXTRA]
            launch = {"command": command, "target_kv": "q8_0/q8_0", "model_sha256": w.SHA_A,
                      "executed_command": [str(w.FROZEN / "build/bin/weekend-srv"), *command[1:]]}
            (attempt / "server.log").write_text("type_k = q8_0\ntype_v = q8_0\n")
            def write_launch():
                (attempt / "launch.json").write_text(json.dumps(launch))
            write_launch()
            w.check_campaign(path, "S4", w.MODEL_A, w.SHA_A)
            command[2] = "f16"
            write_launch()
            with self.assertRaises(AssertionError):
                w.check_campaign(path, "S4", w.MODEL_A, w.SHA_A)
            command[2] = "q8_0"
            command += ["--spec-draft-n-max", "5"]
            write_launch()
            with self.assertRaises(AssertionError):
                w.check_campaign(path, "S4", w.MODEL_A, w.SHA_A)


if __name__ == "__main__":
    unittest.main()
