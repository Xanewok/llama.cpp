import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

from scripts.instruments import hillclimb as h
from scripts.instruments import hillclimb_probe as p


def safe_sample():
    return {"time": time.time(), "ram_available_mib": 8192,
            "gpus": {gpu: {"temperature.gpu": 50, "memory.free": 4000,
                            "power.limit": cap, "power.draw": 30}
                     for gpu, cap in zip(h.DEVICES, [150, 140])}}


class SafetyTests(unittest.TestCase):
    def test_safe(self):
        h.safety_check(safe_sample())

    def test_thermal_and_caps_stop_entire_campaign(self):
        for gpu in h.DEVICES:
            for field, value in [("temperature.gpu", 80), ("power.limit", 180)]:
                sample = safe_sample()
                sample["gpus"][gpu][field] = value
                with self.assertRaises(h.StopCampaign):
                    h.safety_check(sample)

    def test_vram_rejects_arm_without_relaxing_reserve(self):
        for gpu, minimum in zip(h.DEVICES, [512, 256]):
            sample = safe_sample()
            sample["gpus"][gpu]["memory.free"] = minimum
            h.safety_check(sample)
            sample["gpus"][gpu]["memory.free"] -= 1
            with self.assertRaises(h.RejectArm):
                h.safety_check(sample)
            h.safety_check(sample, reserves=False)

    def test_ram_and_missing_telemetry_fail_closed(self):
        sample = safe_sample()
        sample["ram_available_mib"] = 2047
        with self.assertRaises(h.StopCampaign):
            h.safety_check(sample)
        with patch.object(h.subprocess, "run", side_effect=OSError("NVML unavailable")):
            with self.assertRaises(h.StopCampaign):
                h.telemetry()

    def test_occupied_port_including_non_http_service(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            with self.assertRaises(OSError):
                h.endpoint_free(sock.getsockname()[1])

    def test_closed_server_time_wait_is_not_an_occupied_listener(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen(1)
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = server.accept()
                accepted.close()
                self.assertEqual(client.recv(1), b"")
        h.endpoint_free(port)

    def test_exit_classes(self):
        self.assertEqual(h.classify_exit("CUDA out of memory"), "oom")
        self.assertEqual(h.classify_exit("operation not supported"), "unsupported")
        self.assertEqual(h.classify_exit("CUDA illegal memory access"), "gpu_fault")
        self.assertEqual(h.classify_exit("unexpected EOF"), "server_failure")

    def test_gpu_fault_overrides_earlier_unsupported_and_allocation_messages(self):
        prefix = "unsupported Responses tool type 'namespace' skipped\nfailed to allocate scratch\n"
        for failure in ("CUDA error: the launch timed out and was terminated",
                        "CUDA illegal memory access", "device-side assert triggered",
                        "NVRM: Xid (PCI:0000:08:00): 8", "unspecified launch failure"):
            self.assertEqual(h.classify_exit(prefix + failure), "gpu_fault")

    def test_benign_capability_warnings_do_not_classify_a_server_crash(self):
        self.assertEqual(h.classify_exit("unsupported Responses tool type 'namespace' skipped\n"
                                         "backend sampling not supported with SPLIT_MODE_TENSOR\n"
                                         "unexpected EOF"), "server_failure")


class PersistenceTests(unittest.TestCase):
    def test_preparing_existing_campaign_is_idempotent_but_rejects_changes(self):
        jobs = [{"id": "one", "kind": "profile", "model": "unsloth-q4-m"}]
        with tempfile.TemporaryDirectory() as temp:
            campaign = Path(temp)
            manifest = {"queue": jobs, "models": [], "max_attempts": 3}
            h.sealed_json(campaign / "manifest.json", manifest)
            h.atomic_json(campaign / "state.json", {"status": "running", "completed": {"one": "job-one.json"}})
            state = (campaign / "state.json").read_bytes()
            with patch.object(h.shutil, "copytree") as copytree:
                h.initialize(campaign, campaign, profiles=jobs, max_attempts=3)
                h.initialize(campaign, campaign)
            copytree.assert_not_called()
            self.assertEqual((campaign / "state.json").read_bytes(), state)
            with self.assertRaisesRegex(ValueError, "different frozen queue"):
                h.initialize(campaign, campaign, profiles=[dict(jobs[0], id="two")])
            with self.assertRaisesRegex(ValueError, "different frozen retry policy"):
                h.initialize(campaign, campaign, max_attempts=5)
            with patch.object(h, "SERVER_EXTRA", ["-bs"]), \
                    self.assertRaisesRegex(ValueError, "different extra server flags"):
                h.initialize(campaign, campaign)

    def test_profile_queue_validates_fractional_splits_and_dependencies(self):
        jobs = [{"id": "fit", "kind": "profile", "model": "unsloth-q4-xl",
                 "configs": [{"split": "69.5,30.5", "ubatch": 64}]},
                {"id": "compare", "kind": "profile", "model": "unsloth-q4-xl", "base_job": "fit"}]
        self.assertEqual(h.validate_profiles(jobs), jobs)
        for bad in [dict(jobs[0], id="../escape"), dict(jobs[0], base_job="missing"),
                    dict(jobs[0], configs=[{"split": "nan,1"}]),
                    dict(jobs[0], configs=[{"ubatch": 0}]),
                    dict(jobs[0], configs=[{"model": "/unreviewed/model"}])]:
            with self.assertRaises(ValueError):
                h.validate_profiles([bad])

    def test_sealed_records_fail_on_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            h.sealed_json(path, {"status": "passed"})
            self.assertEqual(h.read_sealed(path), {"status": "passed"})
            path.write_text('{"status":"failed"}')
            with self.assertRaises(h.StopCampaign):
                h.read_sealed(path)

    def test_atomic_writes_leave_no_partial_target(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            h.atomic_json(path, {"completed": []})
            with self.assertRaises(ValueError):
                h.atomic_json(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"completed": []})

    def test_queue_is_finite_and_prioritizes_breadth(self):
        jobs = h.queue()
        self.assertEqual(len(jobs), len({j["id"] for j in jobs}))
        self.assertEqual([j["kind"] for j in jobs[:9]], ["smoke"] * 2 + ["control"] * 2 + ["screen"] * 5)
        self.assertEqual([j["model"] for j in jobs[4:9]], [m["id"] for m in h.MODELS[1:]])
        self.assertTrue(all(j.get("slots", 1) in (1, 2, 4) for j in jobs))

    def test_remaining_quants_queue_covers_each_model_without_fallback_sweeps(self):
        path = h.ROOT / "orchestration/local/remaining-quants-overnight.json"
        jobs = h.validate_profiles(json.loads(path.read_text()))
        models = {"unsloth-q4-m", "unsloth-iq4-xs", "jpetrina-iq4-pure"}
        tasks = {"javascript/affine-cipher", "javascript/bowling", "rust/word-count", "rust/forth"}
        self.assertEqual(len(jobs), 66)
        self.assertEqual({job["model"] for job in jobs[:3]}, models)
        for model in models:
            selected = [job for job in jobs if job["model"] == model]
            self.assertEqual(len(selected), 22)
            repeats = [job for job in selected if job["id"].startswith("repeat-")]
            self.assertEqual({(job["task"], job["seed"]) for job in repeats},
                             {(task, seed) for task in tasks for seed in (1730, 1731, 1732)})
            self.assertEqual({job["configs"][0]["slots"] for job in selected
                              if job["id"].startswith("slots-")}, {2, 4})
            self.assertEqual({job["configs"][0]["ubatch"] for job in selected
                              if job["id"].startswith("ubatch-")}, {64, 256, 512})
            self.assertEqual({job["configs"][0]["split"] for job in selected
                              if job["id"].startswith("split-")},
                             {"60,40", "65,35", "70,30", "75,25", "80,20"})
        for job in jobs:
            self.assertEqual(len(job["configs"]), 1)
            self.assertEqual(job["targets"], [98304])
            config = job["configs"][0]
            self.assertEqual(config["context"], 102400)
            self.assertEqual(config["maximum"], 3)
            self.assertEqual(config["mode"], "tensor")
            if not job["id"].startswith("slots-"):
                self.assertEqual(config["slots"], 1)
            if not job["id"].startswith("ubatch-"):
                self.assertEqual(config["ubatch"], 128)
            if not job["id"].startswith("split-"):
                self.assertEqual(config["split"], "69,31")

    def test_process_identity_detects_live_process(self):
        self.assertIsNotNone(h.proc_identity(os.getpid()))
        self.assertIsNone(h.proc_identity(99999999))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name)
        (path / "snapshot").mkdir()
        (path / "attempts").mkdir()
        (path / "artifacts").mkdir()
        manifest = {"port": 8081, "queue": [], "models": h.MODELS, "ratios": h.RATIOS}
        h.sealed_json(path / "manifest.json", manifest)
        h.atomic_json(path / "state.json", {"completed": {}, "status": "prepared", "owned": []})
        self.runner = h.Runner(path, 1)
        self.runner.last_report = time.monotonic()

    def test_deadline_and_stop_markers(self):
        self.runner.deadline = 0
        with self.assertRaisesRegex(h.StopCampaign, "budget_expired"):
            self.runner.poll()
        self.runner.deadline = time.monotonic() + 100
        (self.runner.path / "STOP").touch()
        with self.assertRaisesRegex(h.StopCampaign, "requested_stop"):
            self.runner.poll()

    def test_profile_advances_only_after_safe_fit_and_keeps_depth_results(self):
        job = {"id": "profile", "kind": "profile", "model": "unsloth-q4-xl",
               "configs": [{"ubatch": 128}, {"ubatch": 64}], "targets": [32768, 98304]}
        with patch.object(self.runner, "ensure_model", return_value={"id": job["model"]}), \
                patch.object(self.runner, "attempt", side_effect=[
                    {"status": "failed", "reason": "oom"}, {"status": "passed"},
                    {"status": "passed", "target": 32768},
                    {"status": "passed", "target": 98304}]) as attempt:
            result = self.runner.execute_job(job)
        self.assertEqual([call.args[3] for call in attempt.call_args_list], [0, 0, 32768, 98304])
        self.assertEqual(len(result["probes"]), 2)

    def test_profile_does_not_inherit_failed_fit_or_overflow_context(self):
        job = {"id": "profile", "kind": "profile", "model": "unsloth-q4-xl", "base_job": "failed"}
        with patch.object(self.runner, "ensure_model") as ensure:
            self.assertEqual(self.runner.execute_job(job)["status"], "skipped")
            ensure.assert_not_called()
        job.pop("base_job")
        job["configs"] = [{"context": 102400}]
        job["targets"] = [102400]
        with patch.object(self.runner, "ensure_model", return_value={"id": job["model"]}):
            with self.assertRaisesRegex(h.RejectArm, "exceeds_context"):
                self.runner.execute_job(job)
    def test_hung_child_is_killed_and_owned_record_cleared(self):
        child = self.runner.spawn([sys.executable, "-c", "import time; time.sleep(60)"],
                                  self.runner.path / "child.log")
        with patch.object(self.runner, "poll"):
            with self.assertRaisesRegex(h.RejectArm, "subprocess_timeout"):
                self.runner.wait_child(child, 0.05)
        self.assertIsNotNone(child.poll())
        self.assertEqual(self.runner.state["owned"], [])

    def test_safety_stop_kills_child(self):
        child = self.runner.spawn([sys.executable, "-c", "import time; time.sleep(60)"],
                                  self.runner.path / "child.log")
        with patch.object(self.runner, "poll", side_effect=h.StopCampaign("thermal_limit")):
            with self.assertRaises(h.StopCampaign):
                self.runner.wait_child(child, 10)
        self.assertIsNotNone(child.poll())

    def test_completed_jobs_are_not_executed_on_resume(self):
        dest = self.runner.path / "job-done.json"
        h.sealed_json(dest, {"status": "passed"})
        self.runner.state["completed"] = {"done": dest.name}
        self.runner.manifest["queue"] = [{"id": "done", "kind": "control"}]
        with patch.object(self.runner, "acquire"), patch.object(self.runner, "execute_job") as execute:
            self.runner.run()
        execute.assert_not_called()
        self.assertEqual(self.runner.state["status"], "complete")

    def test_sealed_job_is_recovered_after_crash_before_state_save(self):
        self.runner.manifest["queue"] = [{"id": "done", "kind": "profile"}]
        h.sealed_json(self.runner.path / "job-done.json", {"status": "passed", "decode_tps": 36})
        with patch.object(self.runner, "acquire"), patch.object(self.runner, "poll"), \
                patch.object(self.runner, "execute_job") as execute:
            self.runner.run()
        execute.assert_not_called()
        self.assertEqual(self.runner.state["completed"], {"done": "job-done.json"})

    def test_duplicate_coordinator_does_not_write_live_state_or_report(self):
        with patch.object(self.runner, "acquire", side_effect=h.CampaignBusy("busy")), \
                patch.object(self.runner, "save") as save, patch.object(self.runner, "report") as report:
            self.runner.run()
        save.assert_not_called()
        report.assert_not_called()

    def test_actual_gpu_lock_contention_preserves_state(self):
        state_path = self.runner.path / "state.json"
        before = state_path.read_bytes()
        with (self.runner.path / "qwen-hillclimb.lock").open("a") as owner:
            h.fcntl.flock(owner, h.fcntl.LOCK_EX | h.fcntl.LOCK_NB)
            with patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(self.runner.path)}):
                self.runner.run()
        self.assertEqual(state_path.read_bytes(), before)
        self.assertFalse((self.runner.path / "REPORT.md").exists())

    def test_absolute_deadline_survives_new_runner_budget(self):
        with patch.object(h.time, "time", return_value=1000), \
                patch.object(h.time, "monotonic", return_value=100):
            runner = h.Runner(self.runner.path, 8, deadline_epoch=1060)
            expired = h.Runner(self.runner.path, 8, deadline_epoch=900)
        self.assertEqual(runner.deadline, 160)
        self.assertEqual(expired.deadline, 100)

    def test_transient_retries_are_bounded_backed_off_and_persist_across_resume(self):
        for reason in sorted(h.TRANSIENT_FAILURES):
            with self.subTest(reason=reason), \
                    patch.object(h, "endpoint_free", side_effect=h.RejectArm(reason)) as endpoint, \
                    patch.object(self.runner, "poll"), patch.object(self.runner, "sleep") as sleep, \
                    patch.object(self.runner, "event") as event:
                result = self.runner.attempt(reason, {"id": "test"}, {}, 0)
                self.assertEqual(result["reason"], reason)
                self.assertEqual(endpoint.call_count, 3)
                retries = [call for call in event.call_args_list if call.args[0] == "transient_retry"]
                self.assertEqual([call.kwargs["attempt"] for call in retries], [2, 3])
                for call, expected in zip(retries, (10, 30)):
                    self.assertAlmostEqual(call.kwargs["backoff_seconds"], expected, delta=1)
                resumed = h.Runner(self.runner.path, 1)
                self.assertEqual(resumed.attempt(reason, {"id": "test"}, {}, 0), result)
                self.assertEqual(endpoint.call_count, 3)
                self.assertTrue(sleep.called)

    def test_five_attempt_policy_and_deterministic_failures(self):
        self.runner.max_attempts = 5
        with patch.object(h, "endpoint_free", side_effect=h.RejectArm("server_failure")) as endpoint, \
                patch.object(self.runner, "poll"), patch.object(self.runner, "sleep"), \
                patch.object(self.runner, "event") as event:
            self.runner.attempt("five", {"id": "test"}, {}, 0)
            self.assertEqual(endpoint.call_count, 5)
            retries = [call for call in event.call_args_list if call.args[0] == "transient_retry"]
            self.assertEqual([call.kwargs["attempt"] for call in retries], [2, 3, 4, 5])
        for reason in ("oom", "unsupported", "vram_reserve: " + h.GPU0, "probe_incomplete"):
            with self.subTest(reason=reason), \
                    patch.object(h, "endpoint_free", side_effect=h.RejectArm(reason)) as endpoint, \
                    patch.object(self.runner, "poll"), patch.object(self.runner, "sleep"), \
                    patch.object(self.runner, "event"):
                self.runner.attempt(reason, {"id": "test"}, {}, 0)
                self.assertEqual(endpoint.call_count, 1)

    def test_retry_backoff_honors_safety_stop(self):
        def stop_backoff(seconds, reserves=True):
            if reserves:
                raise h.StopCampaign("thermal_limit")

        with patch.object(h, "endpoint_free", side_effect=h.RejectArm("server_failure")) as endpoint, \
                patch.object(self.runner, "poll"), patch.object(self.runner, "sleep", side_effect=stop_backoff), \
                patch.object(self.runner, "event"):
            with self.assertRaisesRegex(h.StopCampaign, "thermal_limit"):
                self.runner.attempt("safety", {"id": "test"}, {}, 0)
        self.assertEqual(endpoint.call_count, 1)

    def test_failed_fit_never_advances_to_depth(self):
        job = {"id": "screen-x", "kind": "screen", "model": "unsloth-q4-xl"}
        with patch.object(self.runner, "ensure_model", return_value={"id": job["model"]}), \
                patch.object(self.runner, "attempt", return_value={"status": "failed", "reason": "oom"}) as attempt:
            result = self.runner.execute_job(job)
        self.assertEqual(result["reason"], "no_safe_ratio")
        self.assertEqual(attempt.call_count, len(h.RATIOS))
        self.assertTrue(all(call.args[3] == 0 for call in attempt.call_args_list))

    def test_deterministic_oom_is_not_retried_after_resume(self):
        config = {"test": 1}
        spec = {"job": "test", "model": "test", "config": config, "target": 0,
                "task": h.TASKS[0], "seed": 1729}
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        dest = self.runner.path / "attempts" / (key + "-01")
        dest.mkdir()
        h.sealed_json(dest / "result.json", {"status": "failed", "reason": "oom"})
        with patch.object(self.runner, "spawn") as spawn:
            result = self.runner.attempt("test", {"id": "test"}, config, 0)
        self.assertEqual(result["reason"], "oom")
        spawn.assert_not_called()

    def test_download_retries_resume_same_partial_and_verify_before_promotion(self):
        model = copy.deepcopy(h.MODELS[2])
        model["path"] = str(self.runner.path / "models" / "model.gguf")
        model["bytes"] = 3
        self.runner.manifest["models"] = [model]
        partial = Path(model["path"] + ".partial")

        def fake_worker(args, *unused, **kwargs):
            partial.write_bytes(b"abc")
            Path(args[args.index("--out") + 1]).write_text(json.dumps({"native_context": 262144, "block_count": 65, "architecture": "qwen35"}))
            return 0

        with patch.object(self.runner, "spawn") as spawn, \
                patch.object(self.runner, "wait_child", side_effect=[18, 0]), \
                patch.object(self.runner, "sleep"), patch.object(self.runner, "worker", side_effect=fake_worker):
            result = self.runner.ensure_model(model["id"])
        self.assertEqual(spawn.call_count, 2)
        self.assertIn("--continue-at", spawn.call_args.args[0])
        self.assertTrue(Path(result["path"]).exists())
        self.assertFalse(partial.exists())

    def test_bad_artifact_is_not_promoted(self):
        model = copy.deepcopy(h.MODELS[2])
        model["path"] = str(self.runner.path / "models" / "model.gguf")
        self.runner.manifest["models"] = [model]
        with patch.object(self.runner, "spawn"), patch.object(self.runner, "wait_child", return_value=0), \
                patch.object(self.runner, "worker", return_value=1):
            with self.assertRaisesRegex(h.RejectArm, "artifact_verification_failed"):
                self.runner.ensure_model(model["id"])
        self.assertFalse(Path(model["path"]).exists())


    def test_manifest_server_extra_is_appended_to_the_launched_command(self):
        self.runner.manifest.update(binary="/fake/server", server_extra=["-bs", "--spec-mtp-fuse-catchup"])
        model = {"id": "test", "path": "/fake/model", "sha256": "test"}
        config = {"devices": [h.GPU0], "context": 4096, "slots": 1, "maximum": 3,
                  "mode": "layer", "split": "1", "ubatch": 512}
        def api(url, path, **kwargs):
            return {"status": "ok"} if path == "/health" else {
                "total_slots": 1, "default_generation_settings": {"n_ctx": 4096}, "model_path": "/fake/model"}
        server = MagicMock()
        server.poll.return_value = None
        with patch.object(h, "endpoint_free"), patch.object(h, "api", side_effect=api), \
                patch.object(self.runner, "spawn", return_value=server) as spawn, \
                patch.object(self.runner, "poll"), patch.object(self.runner, "sleep"), \
                patch.object(self.runner, "event"), patch.object(self.runner, "run_probe", return_value=1):
            self.runner.attempt("extra", model, config, 0)
        command = spawn.call_args[0][0]
        self.assertEqual(command[-2:], ["-bs", "--spec-mtp-fuse-catchup"])
        launch = h.read_sealed(next((self.runner.path / "attempts").glob("*/launch.json")))
        self.assertEqual(launch["command"], command)


class ProbeTests(unittest.TestCase):
    def test_incomplete_stream_is_persisted_and_classified_for_bounded_retry(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {}
        raw = b'data: {"choices":[{"delta":{"reasoning_content":"partial"}}]}\n'
        response.__iter__.return_value = iter([raw])
        with tempfile.TemporaryDirectory() as temp, patch.object(p.urllib.request, "urlopen", return_value=response):
            result = p.one_request("http://local", "task", 1, 2048, 0, threading.Barrier(1), Path(temp))
            self.assertEqual(result["failure_class"], "transport_failure")
            self.assertEqual((Path(temp) / "request-0.sse").read_bytes(), raw)

    def test_artifact_size_and_hash_checked_before_loading_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.gguf"
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "size"):
                p.inspect_model(path, "bad", 2, temp)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                p.inspect_model(path, "bad", 3, temp)

    def test_prompt_count_includes_template(self):
        task = {"task_id": "javascript/affine-cipher", "prompt": "stub"}
        with patch.object(p.longctx, "build_padding", return_value=("x" * 2000, 2000)), \
                patch.object(p, "api", side_effect=lambda u, e, payload, **kw: {"prompt": "t" * 50 + payload["messages"][0]["content"]}), \
                patch.object(p.longctx, "_tokenize", side_effect=lambda text, url: len(text)):
            prompt, count = p.prepared_prompt("http://local", task, 1000)
        self.assertGreaterEqual(count, 968)
        self.assertLessEqual(count, 1000)
        self.assertEqual(count, len(prompt) + 50)


if __name__ == "__main__":
    unittest.main()
