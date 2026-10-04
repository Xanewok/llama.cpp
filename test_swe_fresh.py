"""CPU tests on recorded artifacts: the Harbor trial of the stub proof run (real Harbor 0.23.0 +
Codex 0.160.0 output), two-agent relay logs (real llama and HQ rows), and a real Codex compaction
record shape. Each negative control must fail on the behaviour it guards."""

import copy
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest

import run_swe_fresh as r
import relay
import select_candidates as sel
import stub_responses_server as stub

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
TRIAL = FIX / "stub-trial/job/trial"
ARGS = types.SimpleNamespace(model="local-benchmark", engine="stub", context=98304, compact=88000, cap_s=1200,
                             agents=1, codex_version="0.160.0", provider="local", harbor_version="0.23.0",
                             dataset_path=None, dataset_ref=None, reasoning="xhigh", relay_bind="172.17.0.1")


def calls(name):
    return r.read_calls(FIX / name)


def stub_job(trial=TRIAL, rows=None, **overrides):
    facts = {**r.parse_trial(trial), **overrides}
    stats = r.call_statistics(rows if rows is not None else calls("stub-trial/model-calls.jsonl"))
    return facts, stats


class TrialParsing(unittest.TestCase):
    def test_stub_trial_is_unsolved_not_error(self):
        facts, stats = stub_job()
        self.assertEqual(facts["reward"], 0.0)
        self.assertFalse(facts["solved"])
        self.assertEqual(facts["agent_version"], "0.160.0")
        self.assertEqual(facts["tool_calls"], 1)
        self.assertAlmostEqual(facts["agent_wall_seconds"], 14.45, places=1)
        self.assertEqual(r.classify(facts, stats, 1200), ("unsolved", []))
        self.assertTrue(facts["dataset_ref"].startswith("sha256:"))

    def test_reward_one_is_solved_negative_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "job/trial"
            shutil.copytree(TRIAL.parent, trial.parent)
            path = trial / "result.json"
            document = json.loads(path.read_text())
            document["verifier_result"]["rewards"]["reward"] = 1.0
            path.write_text(json.dumps(document))
            facts, stats = stub_job(trial)
            self.assertTrue(facts["solved"])
            self.assertEqual(r.classify(facts, stats, 1200)[0], "passed")

    def test_agent_timeout_is_a_censored_outcome_not_a_gate(self):
        facts, stats = stub_job(exception_type="AgentTimeoutError", timed_out=True, solved=False)
        self.assertEqual(r.classify(facts, stats, 1200)[0], "timeout")

    def test_other_harbor_exception_invalidates(self):
        facts, stats = stub_job(exception_type="RuntimeError", exception_message="boom")
        status, gates = r.classify(facts, stats, 1200)
        self.assertEqual(status, "invalid")
        self.assertIn("harbor_exception: RuntimeError", gates)

    def test_zero_model_calls_invalidates_even_with_a_reward(self):
        # The failure the whole proof exists to catch: Codex never reached the relay (wrong base URL).
        facts, stats = stub_job(rows=[])
        status, gates = r.classify(facts, stats, 1200)
        self.assertEqual((status, "no_model_calls" in gates), ("invalid", True))

    def test_cap_not_enforced_is_flagged(self):
        facts, stats = stub_job(agent_wall_seconds=3000.0)
        self.assertEqual(r.classify(facts, stats, 1200)[0], "invalid")

    def test_websocket_probe_rows_do_not_count_as_calls(self):
        rows = calls("stub-trial/model-calls.jsonl")
        self.assertTrue(all(c["method"] == "POST" for c in rows))
        probe = {"method": "GET", "path": "/v1/responses", "status": 404, "call": 99}
        stats = r.call_statistics(rows + [probe])
        self.assertEqual(stats["calls"], 2)

    def test_compaction_record_shape_from_a_real_codex_rollout(self):
        # Line shape copied from a real 2026-03-12 Codex rollout; payload text dropped.
        record = {"timestamp": "2026-03-12T20:35:32.424Z", "ordinal": 626, "type": "compacted",
                  "payload": {"message": "", "replacement_history": []}}
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "job/trial"
            shutil.copytree(TRIAL.parent, trial.parent)
            rollout = next((trial / "agent/sessions").rglob("rollout-*.jsonl"))
            self.assertEqual(r.rollout_stats(trial)["compactions"], 0)
            with rollout.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            self.assertEqual(r.rollout_stats(trial)["compactions"], 1)


class CallStatistics(unittest.TestCase):
    def test_llama_rows_match_independent_sums(self):
        rows = calls("two-agent/llama-fix-v4a-agent1.jsonl")
        stats = r.call_statistics(rows)
        timed = [c["timings"] for c in rows]
        self.assertAlmostEqual(stats["decode_tps"], sum(t["predicted_n"] for t in timed)
                               / sum(t["predicted_ms"] / 1000 for t in timed), places=9)
        self.assertAlmostEqual(stats["draft_acceptance"], sum(t["draft_n_accepted"] for t in timed)
                               / sum(t["draft_n"] for t in timed), places=9)
        self.assertEqual(stats["calls"], 30)
        self.assertEqual(stats["peak_context_tokens"], max(c["input_tokens"] + c["output_tokens"] for c in rows))
        self.assertEqual(stats["output_tokens"], sum(c["output_tokens"] for c in rows))
        self.assertAlmostEqual(stats["cache_hit_fraction"],
                               sum(c["cached_tokens"] for c in rows) / sum(c["input_tokens"] for c in rows))

    def test_missing_draft_counters_give_none_not_zero(self):
        rows = copy.deepcopy(calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        for row in rows[:3]:
            row["timings"].pop("draft_n")
        self.assertIsNone(r.call_statistics(rows)["draft_acceptance"])

    def test_hq_rows_have_no_exact_decode_or_acceptance(self):
        stats = r.call_statistics(calls("two-agent/hq-codexh-k5-r1.jsonl"))
        self.assertEqual(stats["calls"], 5)
        self.assertIsNone(stats["decode_tps"])
        self.assertIsNone(stats["draft_acceptance"])
        self.assertEqual(stats["peak_context_tokens"], 14654)

    def test_failed_call_is_reported_and_excluded(self):
        rows = copy.deepcopy(calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        rows[4]["status"] = 500
        rows[4]["event"] = "response.failed"
        stats = r.call_statistics(rows)
        self.assertEqual(stats["failed_calls"], [rows[4]["call"]])
        self.assertEqual(stats["http_500_calls"], [rows[4]["call"]])
        self.assertEqual(r.classify({**r.parse_trial(TRIAL)}, stats, 1200)[0], "invalid")

    def test_runaway_flag_at_32768_tokens(self):
        rows = copy.deepcopy(calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        self.assertEqual(r.call_statistics(rows)["runaway_calls"], [])
        rows[2]["output_tokens"] = 32768
        self.assertEqual(r.call_statistics(rows)["runaway_calls"], [rows[2]["call"]])

    def test_depth_buckets_partition_timed_calls(self):
        stats = r.call_statistics(calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        self.assertEqual(sum(b["calls"] for b in stats["by_depth"].values()), 30)
        self.assertTrue(set(stats["by_depth"]) <= {"0-8192", "8192-32768", "32768-65536", "65536-inf"})

    def test_input_drop_counts_compaction_like_events(self):
        rows = copy.deepcopy(calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        self.assertEqual(r.call_statistics(rows)["input_drops_over_30pct"], 0)
        rows[10]["input_tokens"] = 1000
        self.assertGreaterEqual(r.call_statistics(rows)["input_drops_over_30pct"], 1)


class Metrics(unittest.TestCase):
    def test_delta_and_restart(self):
        before = {"draft_tokens_total": 10.0, "draft_accepted_total": 6.0}
        after = {"draft_tokens_total": 30.0, "draft_accepted_total": 16.0}
        self.assertEqual(r.metrics_delta(before, after), {"draft_tokens_total": 20.0, "draft_accepted_total": 10.0})
        self.assertEqual(r.metrics_delta(after, before), {})  # counter reset after a server restart
        self.assertEqual(r.metrics_delta({}, after), {})


class SummarySchema(unittest.TestCase):
    def jobs(self, rows, delta=None, solved=True):
        facts, stats = stub_job(rows=rows)
        if solved:
            facts.update(reward=1.0, solved=True)
        job = r.build_job("pallets__click-3239", 1, 0, facts, stats, delta or {}, ARGS, 1000.0, 1100.0)
        return job

    def test_campaign_index_fields_are_present(self):
        real = HERE.parent / "qwen38-eval/results/2026-09-28-dual-5060ti/campaign-hq15-codexh-k5-r1-20261003/SUMMARY.json"
        job = self.jobs(calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        summary = r.campaign_summary([job], [("pallets__click-3239", 1)], 100.0, {}, ARGS)
        wanted = {"updated_epoch", "state", "model", "tasks", "solved", "timeouts", "wall_seconds", "output_tokens",
                  "prefill_seconds", "decode_seconds", "peak_context_tokens", "draft_acceptance"}
        if real.exists():
            self.assertEqual(wanted, set(json.loads(real.read_text())))
        self.assertTrue(wanted <= set(summary))
        self.assertTrue(summary["valid"], summary["gates"])
        self.assertEqual(summary["solved"], 1)
        self.assertEqual(job["probe"]["verified_wall_seconds"], job["probe"]["agent_wall_seconds"])
        self.assertAlmostEqual(summary["draft_acceptance"], job["call_stats"]["draft_acceptance"])
        self.assertEqual(summary["acceptance_source"], "per_call_timings")

    def test_unsolved_scores_the_cap(self):
        job = self.jobs(calls("two-agent/llama-fix-v4a-agent1.jsonl"), solved=False)
        self.assertEqual(job["status"], "unsolved")
        self.assertEqual(job["probe"]["verified_wall_seconds"], 1200.0)
        self.assertTrue(job["probe"]["censored"])

    def test_hq_acceptance_comes_from_engine_counters(self):
        delta = {"draft_tokens_total": 400.0, "draft_accepted_total": 240.0, "tokens_predicted_total": 1000.0,
                 "tokens_predicted_seconds_total": 10.0}
        job = self.jobs(calls("two-agent/hq-codexh-k5-r1.jsonl"), delta)
        self.assertEqual(job["probe"]["mtp"]["source"], "engine_counter_delta_exclusive_server")
        self.assertAlmostEqual(job["probe"]["mtp"]["draft_accept_rate"], 0.6)
        self.assertAlmostEqual(job["decode_tps"], 100.0)
        summary = r.campaign_summary([job], [("pallets__click-3239", 1)], 50.0, delta, ARGS)
        self.assertAlmostEqual(summary["draft_acceptance"], 0.6)
        self.assertEqual(summary["acceptance_source"], "engine_counter_delta")

    def test_incomplete_queue_and_invalid_job_make_the_campaign_invalid(self):
        job = self.jobs(calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        self.assertFalse(r.campaign_summary([job], [("pallets__click-3239", 1), ("other", 1)], 1.0, {}, ARGS)["valid"])
        bad = self.jobs([])
        summary = r.campaign_summary([bad], [("pallets__click-3239", 1)], 1.0, {}, ARGS)
        self.assertFalse(summary["valid"])
        self.assertTrue(any("no_model_calls" in g for g in summary["gates"]))


class HarborCommand(unittest.TestCase):
    def command(self, **extra):
        args = types.SimpleNamespace(**{**vars(ARGS), **extra})
        return r.harbor_command("pallets__click-3239", "http://172.17.0.1:5555/v1", Path("/o"), "job", 0.4, args)

    def test_registry_command_matches_the_report(self):
        c = self.command(provider="builtin")
        self.assertEqual(c[:3], ["uvx", "harbor==0.23.0", "run"])
        for flag, value in (("-i", "swe-rebench/pallets__click-3239"), ("-d", "swe-rebench/swe-rebench-leaderboard"),
                            ("--allow-agent-host", "172.17.0.1"), ("-m", "openai/local-benchmark"),
                            ("--agent-timeout-multiplier", "0.400000")):
            self.assertEqual(c[c.index(flag) + 1], value)
        self.assertIn("OPENAI_BASE_URL=http://172.17.0.1:5555/v1", c)
        self.assertIn("version=0.160.0", c)
        config = json.loads(next(a for a in c if a.startswith("config="))[7:])
        self.assertEqual((config["model_context_window"], config["model_auto_compact_token_limit"]), (98304, 88000))
        self.assertNotIn("model_provider", config)

    def test_local_provider_fallback_carries_the_relay_url_and_disables_websockets(self):
        config = json.loads(next(a for a in self.command(provider="local") if a.startswith("config="))[7:])
        self.assertEqual(config["model_provider"], "local")
        provider = config["model_providers"]["local"]
        self.assertEqual(provider["base_url"], "http://172.17.0.1:5555/v1")
        self.assertIs(provider["supports_websockets"], False)
        self.assertEqual(provider["wire_api"], "responses")

    def test_snapshot_uses_bare_task_names(self):
        c = self.command(dataset_path=Path("/snap"))
        self.assertEqual(c[c.index("-i") + 1], "pallets__click-3239")
        self.assertEqual(c[c.index("-p") + 1], "/snap")
        self.assertNotIn("-d", c)

    def test_dataset_ref_pins_the_registry_digest(self):
        c = self.command(dataset_ref="sha256:abc")
        self.assertEqual(c[c.index("-d") + 1], "swe-rebench/swe-rebench-leaderboard@sha256:abc")


class RelayEndToEnd(unittest.TestCase):
    """Real relay + stub server over loopback: the same route the container uses over docker0."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "stub").mkdir()
        (self.tmp / "relay").mkdir()
        self.server = stub.Server(("127.0.0.1", 0), stub.Handler)
        self.server.out, self.server.timings, self.server.turn = self.tmp / "stub", True, 0
        self.server.totals = {"draft": 0, "accepted": 0, "tokens": 0, "decode_s": 0.0}
        self.server.lock = threading.Lock()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp)

    def post(self, port, body):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("POST", "/v1/responses", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        try:
            return response.status, response.read()
        finally:
            connection.close()

    def test_usage_and_timings_are_recorded(self):
        import time
        proxy = relay.UsageProxy(self.server.server_address[1], self.tmp / "relay", 0, "t#r1", time.monotonic() + 30)
        proxy.start()
        tools = [{"type": "function", "name": "exec_command"}]
        status, _ = self.post(proxy.port, {"model": "m", "tools": tools, "input": [{"type": "message"}], "stream": True})
        status2, _ = self.post(proxy.port, {"model": "m", "tools": tools, "stream": True,
                                            "input": [{"type": "function_call_output"}]})
        rows = proxy.stop()
        self.assertEqual((status, status2), (200, 200))
        stats = r.call_statistics(rows)
        self.assertEqual(stats["completed_calls"], 2)
        self.assertEqual(stats["peak_input_tokens"], 9000)
        self.assertAlmostEqual(stats["draft_acceptance"], 0.65)
        first = json.loads((self.tmp / "stub/request-001.json").read_text())
        self.assertEqual(first["tools"], ["exec_command"])  # request bytes reach the server unchanged

    def test_unreachable_upstream_is_an_error_row_not_a_success(self):
        import time
        proxy = relay.UsageProxy(1, self.tmp / "relay", 0, "t#r1", time.monotonic() + 30)
        proxy.start()
        try:
            self.post(proxy.port, {"input": []})
        except (http.client.HTTPException, OSError):
            pass
        rows = proxy.stop()
        stats = r.call_statistics(rows)
        self.assertEqual(stats["completed_calls"], 0)
        self.assertEqual(len(stats["failed_calls"]), 1)


class Selection(unittest.TestCase):
    def rows(self):
        def row(i, repo, lines=10, files=1, chars=100):
            return {"instance_id": f"{repo}-{i}", "repo": repo, "harbor_memory": None, "problem_statement": "x" * chars,
                    "patch": "diff --git a b\n" * files + "+x\n" * lines}
        return [row(1, "a/a"), row(2, "a/a"), row(3, "b/b"), row(4, "c/c", lines=500), row(5, "d/d", files=5),
                row(6, "e/e", chars=9000), row(7, "f/f"), row(8, "g/g")]

    def sizes(self):
        return {f"{repo}-{i}": {"size": 10**9} for i, repo in enumerate("aabcdefg", 1)} | {"g/g-8": {"size": 5 * 10**9}}

    LIMITS = dict(max_patch_lines=150, max_files=3, max_statement=6000, max_image_gb=2.5, max_memory_gb=4)

    def test_filters_one_per_repo_and_is_deterministic(self):
        sizes = {f"{r}-{i}": {"size": 10**9} for i, r in [(1, "a/a"), (2, "a/a"), (3, "b/b"), (4, "c/c"), (5, "d/d"),
                                                          (6, "e/e"), (7, "f/f")]}
        sizes["g/g-8"] = {"size": 5 * 10**9}
        chosen, eligible, repos = sel.pick(self.rows(), sizes, "seed", 10, **self.LIMITS)
        ids = {c["instance_id"] for c in chosen}
        self.assertEqual(len({c["repo"] for c in chosen}), len(chosen))
        # patch too long, too many files, long statement and oversized image are each excluded
        self.assertFalse(ids & {"c/c-4", "d/d-5", "e/e-6", "g/g-8"})
        self.assertEqual((eligible, repos), (4, 3))
        again, _, _ = sel.pick(self.rows(), sizes, "seed", 10, **self.LIMITS)
        self.assertEqual([c["instance_id"] for c in again], [c["instance_id"] for c in chosen])
        other, _, _ = sel.pick(self.rows(), sizes, "other-seed", 10, **self.LIMITS)
        self.assertEqual({c["repo"] for c in other}, {c["repo"] for c in chosen})

    def test_recorded_split_reproduces_the_frozen_calibration_list(self):
        parquet, sizes = HERE / "data/rebench-2026_03.parquet", HERE / "data/sizes-2026_03.json"
        frozen = HERE / "data/calibration-16.json"
        if not (parquet.exists() and sizes.exists() and frozen.exists()):
            self.skipTest("data files absent")
        import pyarrow.parquet as pq
        rows = pq.read_table(parquet).to_pylist()
        document = json.loads(frozen.read_text())
        chosen, _, _ = sel.pick(rows, json.loads(sizes.read_text()), document["seed"], 16, **document["filter"])
        self.assertEqual([c["instance_id"] for c in chosen], [t["instance_id"] for t in document["tasks"]])
        self.assertEqual(len(chosen), 16)
        self.assertEqual(sel.hashlib.sha256(parquet.read_bytes()).hexdigest(), document["parquet_sha256"])


class Manifest(unittest.TestCase):
    def test_manifest_toml_parses_with_required_keys(self):
        import subprocess
        import sys
        import tomllib
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tasks = tmp / "tasks.json"
            tasks.write_text(json.dumps({"tasks": [{"instance_id": "a__a-1"}, {"instance_id": "b__b-2"}]}))
            pins = tmp / "pins.json"
            pins.write_text(json.dumps({"engine": "llama", "tasks_file": str(tasks), "dataset_path": str(tmp),
                                        "files": {str(tasks): "0" * 64}}))
            run = lambda stage, **kw: subprocess.run(
                [sys.executable, str(HERE / "make_manifest.py"), "manifest", "--pins", str(pins), "--stage", stage,
                 "--out", str(HERE / "results/x"), "--xid-since", "2026-10-03 18:57:00", "--canary-task", "a__a-1",
                 "--id", f"swe-fresh-{stage}", "--runs", "2"], capture_output=True, text=True, check=True).stdout
            canary, measure = tomllib.loads(run("canary")), tomllib.loads(run("measure"))
        self.assertEqual(canary["run"]["env"]["ONLY"], "canary")
        self.assertEqual((canary["ab"], canary["retry"], canary["run"]["ports"]), (False, "fresh", [8081]))
        self.assertLessEqual(canary["est_s"] * 10, canary["budget_s"] * 7)
        self.assertEqual(measure["run"]["env"]["REPEAT"], "2")
        self.assertLessEqual(measure["est_s"] * 10, measure["budget_s"] * 7)
        self.assertIn(str(pins), measure["pins"]["files"])
        self.assertTrue(measure["outputs"]["paths"][0].endswith("results/x"))


class PinRoundTrip(unittest.TestCase):
    def test_pin_verify_detects_a_changed_task_file(self):
        import make_manifest as mm
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "snap/a__a-1").mkdir(parents=True)
            (tmp / "snap/can__can-1").mkdir()
            (tmp / "snap/a__a-1/task.toml").write_text("x")
            (tmp / "snap/can__can-1/task.toml").write_text("y")
            tasks = tmp / "tasks.json"
            tasks.write_text(json.dumps({"tasks": [{"instance_id": "a__a-1"}]}))
            engine = tmp / "model.gguf"
            engine.write_text("weights")
            original = mm.engine_files
            mm.engine_files = lambda engine_name, hs: [engine]
            try:
                args = types.SimpleNamespace(engine="llama", tasks_file=tasks, dataset_path=tmp / "snap",
                                             canary_task="can__can-1", pins=tmp / "pins.json")
                mm.pin(args)
                mm.verify(args)
                (tmp / "snap/a__a-1/task.toml").write_text("edited")
                with self.assertRaises(SystemExit):
                    mm.verify(args)
                (tmp / "snap/a__a-1/task.toml").write_text("x")
                mm.verify(args)
                args.canary_task = "missing"
                args.pins = tmp / "pins2.json"
                with self.assertRaises(SystemExit):
                    mm.pin(args)
            finally:
                mm.engine_files = original


# ---------------------------------------------------------------- Opus-judge fixes (2026-10-04)

GPUQ_PATH = "/home/xanewok/.local/bin:/usr/local/bin:/usr/bin:/bin"
CAP = 300.0
AGENT_START = 1000.0  # fake agent_execution.started_at epoch; the cap ends at AGENT_START + CAP


def manifest_toml(stage, runs=2, tasks=("a__a-1", "b__b-2"), engine="llama"):
    import subprocess
    import sys
    import tomllib
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tasks_file = tmp / "tasks.json"
        tasks_file.write_text(json.dumps({"tasks": [{"instance_id": t} for t in tasks]}))
        pins = tmp / "pins.json"
        pins.write_text(json.dumps({"engine": engine, "tasks_file": str(tasks_file), "dataset_path": str(tmp),
                                    "files": {str(tasks_file): "0" * 64}}))
        text = subprocess.run(
            [sys.executable, str(HERE / "make_manifest.py"), "manifest", "--pins", str(pins), "--stage", stage,
             "--out", str(HERE / "results/x"), "--xid-since", "2026-10-03 18:57:00", "--canary-task", "a__a-1",
             "--id", f"swe-fresh-{stage}", "--runs", str(runs)], capture_output=True, text=True, check=True).stdout
    return tomllib.loads(text)


def cut_rows(error="BrokenPipeError: [Errno 32] Broken pipe", started=AGENT_START + CAP - 5, finished=AGENT_START + CAP,
             **extra):
    """Stub rows plus one failed call, shaped like the call Harbor's kill aborts (timeout_inflight.py of the judge)."""
    rows = copy.deepcopy(calls("stub-trial/model-calls.jsonl"))
    cut = dict(rows[-1])
    cut.update(call=99, event=None, error=error, output_tokens=None, started_epoch=started, finished_epoch=finished,
               **extra)
    return rows + [cut]


def timeout_case(rows, **facts_extra):
    facts, _ = stub_job(rows=rows, exception_type="AgentTimeoutError", timed_out=True, solved=False,
                        agent_started_epoch=AGENT_START, agent_wall_seconds=CAP, **facts_extra)
    stats = r.call_statistics(rows, r.inflight_cutoff(facts, CAP))
    return facts, stats


class GpuqManifest(unittest.TestCase):
    def test_every_item_carries_a_path_that_finds_uvx(self):
        # gpuq's own PATH is /usr/local/bin:/usr/bin:/bin; uvx lives in ~/.local/bin (judge F1)
        for stage in ("canary", "measure"):
            path = manifest_toml(stage)["run"]["env"]["PATH"]
            self.assertEqual(path, GPUQ_PATH)

    def test_exit_codes_use_gpuq_classes(self):
        for stage in ("canary", "measure"):
            self.assertEqual(manifest_toml(stage)["exit_codes"],
                             {"3": "hazard", "4": "transient", "5": "deterministic", "6": "integrity", "7": "timeout"})

    def test_disk_floor_is_small_for_the_canary_and_30_for_measure(self):
        self.assertEqual(manifest_toml("canary")["run"]["env"]["MIN_FREE_GB"], "5")
        self.assertEqual(manifest_toml("measure")["run"]["env"]["MIN_FREE_GB"], "30")

    def test_pass_a_is_sixteen_tasks_one_run(self):
        tasks = [f"r{i}__r{i}-1" for i in range(16)]
        item = manifest_toml("measure", runs=1, tasks=tasks)
        self.assertEqual(item["run"]["env"]["REPEAT"], "1")
        self.assertEqual(item["est_s"], 180 + 16 * (240 + 600))
        self.assertLessEqual(item["est_s"] * 10, item["budget_s"] * 7)

    def test_identity_failure_matches_gpuq_integrity_regex(self):
        import make_manifest as mm
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tracked = tmp / "x.txt"
            tracked.write_text("a")
            document = {"engine": "llama", "files": {str(tracked): mm.support().digest(tracked)}}
            (tmp / "pins.json").write_text(json.dumps(document))
            tracked.write_text("b")
            original = mm.engine_files
            mm.engine_files = lambda *a: []
            try:
                with self.assertRaises(SystemExit) as caught:
                    mm.verify(types.SimpleNamespace(pins=tmp / "pins.json"))
            finally:
                mm.engine_files = original
        self.assertRegex(str(caught.exception), r"inputs changed")  # gpuq classify.RULES integrity

    def test_no_unpinned_uvx_call_anywhere(self):
        for name in ("run_swe_fresh.py", "gpu_swe_fresh.sh", "make_manifest.py"):
            for number_, line in enumerate((HERE / name).read_text().splitlines(), 1):
                if "uvx" in line and not line.lstrip().startswith("#"):
                    self.assertTrue("harbor==" in line or "harbor_requirement(" in line, f"{name}:{number_}: {line}")

    def test_harbor_commands_are_pinned_even_without_a_version_argument(self):
        args = types.SimpleNamespace(**{**vars(ARGS), "harbor_version": None})
        self.assertEqual(r.harbor_command("t", "http://x/v1", Path("/o"), "j", 1.0, args)[:3],
                         ["uvx", "harbor==0.23.0", "run"])
        self.assertEqual(r.harbor_version_command(args), ["uvx", "harbor==0.23.0", "--version"])


class TimeoutScoring(unittest.TestCase):
    def test_call_aborted_by_the_cap_kill_scores_timeout_not_invalid(self):
        facts, stats = timeout_case(cut_rows())
        self.assertEqual(r.classify(facts, stats, CAP), ("timeout", []))

    def test_call_started_in_the_last_seconds_is_in_flight_whatever_its_error(self):
        facts, stats = timeout_case(cut_rows(error="RemoteDisconnected: x", started=AGENT_START + CAP - 1, finished=AGENT_START + CAP))
        self.assertEqual(r.classify(facts, stats, CAP)[0], "timeout")

    def test_reward_one_does_not_turn_a_cap_cut_call_into_a_failure(self):
        # The verifier runs after the kill, so a timed-out run can carry reward 1; the cut call is still the cap's.
        facts, stats = timeout_case(cut_rows(), reward=1.0)
        facts["solved"] = True
        self.assertEqual(r.classify(facts, stats, CAP), ("passed", []))

    def test_failed_call_well_before_the_cap_still_invalidates(self):
        facts, stats = timeout_case(cut_rows(error="ConnectionRefusedError: x", started=AGENT_START + 100,
                                             finished=AGENT_START + 101))
        status, gates = r.classify(facts, stats, CAP)
        self.assertEqual(status, "invalid")
        self.assertTrue(any(g.startswith("call_failed") for g in gates))

    def test_disconnect_long_before_the_cap_still_invalidates(self):
        facts, stats = timeout_case(cut_rows(started=AGENT_START + 100, finished=AGENT_START + 101))
        self.assertEqual(r.classify(facts, stats, CAP)[0], "invalid")

    def test_non_timeout_run_never_ignores_a_failed_call(self):
        rows = cut_rows()
        facts, _ = stub_job(rows=rows, agent_started_epoch=AGENT_START)
        stats = r.call_statistics(rows, r.inflight_cutoff(facts, CAP))
        self.assertEqual(r.classify(facts, stats, 1200)[0], "invalid")

    def test_unknown_agent_start_ignores_nothing(self):
        facts, _ = stub_job(rows=cut_rows(), exception_type="AgentTimeoutError", timed_out=True, solved=False,
                            agent_started_epoch=None)
        self.assertIsNone(r.inflight_cutoff(facts, CAP))
        self.assertEqual(r.classify(facts, r.call_statistics(cut_rows(), None), CAP)[0], "invalid")

    def test_parse_trial_reads_the_agent_start_epoch(self):
        self.assertAlmostEqual(r.parse_trial(TRIAL)["agent_started_epoch"], 1791066167.796706, places=3)


class TruncatedCalls(unittest.TestCase):
    def rows(self, **extra):
        rows = copy.deepcopy(calls("stub-trial/model-calls.jsonl"))
        row = dict(rows[-1])
        row.update(call=99, event="response.incomplete", output_tokens=32768, status=200, error=None, **extra)
        return rows + [row]

    def test_max_output_tokens_hit_is_runaway_not_invalid(self):
        rows = self.rows(incomplete_reason="max_output_tokens")
        facts, stats = stub_job(rows=rows)
        self.assertEqual(stats["truncated_calls"], [99])
        self.assertEqual(stats["failed_calls"], [])
        self.assertEqual(r.classify(facts, stats, 1200), ("unsolved", []))
        job = r.build_job("t", 1, 0, facts, stats, {}, ARGS, 1.0, 2.0)
        self.assertIs(job["probe"]["runaway"], True)
        self.assertEqual(job["status"], "unsolved")

    def test_solved_task_with_a_runaway_call_stays_passed(self):
        facts, stats = stub_job(rows=self.rows(incomplete_reason="max_output_tokens"), reward=1.0, solved=True)
        self.assertEqual(r.classify(facts, stats, 1200)[0], "passed")

    def test_server_max_new_tokens_without_a_reason_counts_at_the_runaway_limit(self):
        facts, stats = stub_job(rows=self.rows())
        self.assertEqual(stats["truncated_calls"], [99])

    def test_other_incomplete_reasons_stay_failed_calls(self):
        facts, stats = stub_job(rows=self.rows(incomplete_reason="content_filter"))
        self.assertEqual(stats["truncated_calls"], [])
        self.assertEqual(r.classify(facts, stats, 1200)[0], "invalid")

    def test_clean_job_is_not_flagged_runaway(self):
        facts, stats = stub_job()
        self.assertIs(r.build_job("t", 1, 0, facts, stats, {}, ARGS, 1.0, 2.0)["probe"]["runaway"], False)

    def test_relay_records_the_incomplete_reason(self):
        call = {}
        relay.scan_response({"type": "response.incomplete", "response": {
            "incomplete_details": {"reason": "max_output_tokens"}, "usage": {"input_tokens": 5, "output_tokens": 7}}}, call)
        self.assertEqual((call["event"], call["incomplete_reason"], call["output_tokens"]),
                         ("response.incomplete", "max_output_tokens", 7))


class FakeHarbor:
    """A stand-in child in its own session (as Harbor is) that records which signal reached it."""
    SCRIPT = """#!/bin/bash
job="$1"; ready="$2"; mark="$3"; pidfile="$4"
echo $$ > "$pidfile"
mkdir -p "$job/pallets__click-1__AbCdEfG"
trap 'echo INT > "$mark"; exit 0' INT
trap 'echo TERM > "$mark"; exit 0' TERM
touch "$ready"
while true; do sleep 0.1; done
"""

    def __init__(self, root):
        self.root = Path(root)
        self.script = self.root / "fake_harbor.sh"
        self.script.write_text(self.SCRIPT)
        self.ready, self.mark, self.pidfile = (self.root / n for n in ("ready", "mark", "pid"))

    def command(self, task, base_url, jobs_dir, job_name, multiplier, args):
        return ["bash", str(self.script), str(Path(jobs_dir) / job_name), str(self.ready), str(self.mark), str(self.pidfile)]

    def wait_ready(self, timeout=30):
        until = time.monotonic() + timeout
        while not self.ready.exists() and time.monotonic() < until:
            time.sleep(0.02)
        return self.ready.exists()

    def signal_seen(self, timeout=10):
        until = time.monotonic() + timeout
        while not self.mark.exists() and time.monotonic() < until:
            time.sleep(0.05)
        return self.mark.read_text().strip() if self.mark.exists() else None

    def reap(self):
        """Kill only the child this fixture started (its own pid, which is its group)."""
        import os
        import signal
        if self.pidfile.exists():
            try:
                os.killpg(int(self.pidfile.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


def task_args(tmp, **extra):
    return types.SimpleNamespace(**{**vars(ARGS), "hard_timeout_s": 60, "upstream_port": 1, "upstream_host": "127.0.0.1",
                                    "relay_bind": "127.0.0.1", "metrics_url": None, "min_free_gb": 0, **extra})


class AbnormalStop(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.harbor = FakeHarbor(self.tmp)
        self.campaign = self.tmp / "campaign"
        (self.campaign / "attempts").mkdir(parents=True)
        self.downs = []
        import signal
        # A background job (as gpuq or `cmd &` starts one) inherits SIGINT ignored; the fake child must not.
        self.sigint = signal.signal(signal.SIGINT, signal.default_int_handler)

    def tearDown(self):
        import signal
        signal.signal(signal.SIGINT, self.sigint)
        self.harbor.reap()
        shutil.rmtree(self.tmp)

    def patched(self):
        from unittest import mock
        return (mock.patch.object(r, "harbor_command", self.harbor.command),
                mock.patch.object(r, "compose_down", lambda names, *a, **k: self.downs.append(list(names)), create=True))

    def test_keyboard_interrupt_reaches_harbor_and_compose_down_uses_the_recorded_project(self):
        import os
        import signal
        threading.Thread(target=lambda: self.harbor.wait_ready() and os.kill(os.getpid(), signal.SIGINT), daemon=True).start()
        one, two = self.patched()
        with one, two, self.assertRaises(KeyboardInterrupt):
            r.run_task(0, ("pallets__click-1", 1), task_args(self.tmp), self.campaign)
        self.assertEqual(self.harbor.signal_seen(), "INT")
        self.assertEqual(self.downs, [["pallets__click-1__abcdefg__env"]])
        record = json.loads((self.campaign / "attempts/agent-0-pallets__click-1-r1/harbor.process.json").read_text())
        self.assertEqual(record["pid"], record["pgid"])

    def test_sigterm_is_forwarded_from_a_worker_thread(self):
        import os
        import signal
        done = {}
        previous = r.install_stop_handlers()
        one, two = self.patched()
        try:
            with one, two:
                worker = threading.Thread(target=lambda: done.update(
                    job=r.run_task(0, ("pallets__click-1", 1), task_args(self.tmp), self.campaign)))
                worker.start()
                self.assertTrue(self.harbor.wait_ready())
                os.kill(os.getpid(), signal.SIGTERM)
                worker.join(60)
            self.assertTrue(r.STOP.is_set())
        finally:
            r.restore_stop_handlers(previous)
            r.STOP.clear()
        self.assertEqual(self.harbor.signal_seen(), "INT")
        self.assertEqual(self.downs, [["pallets__click-1__abcdefg__env"]])
        self.assertEqual(done["job"]["status"], "invalid")

    def test_hard_timeout_stops_the_group_and_cleans_the_project(self):
        one, two = self.patched()
        with one, two:
            result = r.run_task(0, ("pallets__click-1", 1), task_args(self.tmp, hard_timeout_s=1.5), self.campaign)
        self.assertEqual(self.harbor.signal_seen(), "INT")
        self.assertEqual(self.downs, [["pallets__click-1__abcdefg__env"]])
        self.assertIn("hard_timeout", result["probe"]["error"])

    def test_compose_down_command_is_exact_and_never_prune(self):
        seen = []
        r.compose_down(["a__b__c__env", "x__y__z__env"], runner=lambda cmd, **kw: seen.append(cmd))
        self.assertEqual(seen, [["docker", "compose", "--project-name", name, "down", "--volumes", "--remove-orphans"]
                                for name in ("a__b__c__env", "x__y__z__env")])

    def test_backstop_cleanup_kills_only_the_recorded_live_process(self):
        import os
        import subprocess
        attempt = self.campaign / "attempts/agent-0-t-r1"
        attempt.mkdir()
        pids = []
        for _ in range(2):
            child = subprocess.Popen(["sleep", "60"], start_new_session=True)
            pids.append(child)
        try:
            good, stale = pids
            (attempt / "compose-projects.json").write_text(json.dumps(["t__abc__env"]))
            (attempt / "harbor.process.json").write_text(json.dumps(
                {"pid": good.pid, "pgid": good.pid, "start_tick": r.proc_start_tick(good.pid)}))
            other = self.campaign / "attempts/agent-0-u-r1"
            other.mkdir()
            (other / "harbor.process.json").write_text(json.dumps(
                {"pid": stale.pid, "pgid": stale.pid, "start_tick": "1"}))  # pid reused by someone else
            r.cleanup_campaign(self.campaign, runner=lambda cmd, **kw: self.downs.append(cmd[3]))
            self.assertEqual(good.wait(10), -9)
            self.assertIsNone(stale.poll())
            self.assertEqual(self.downs, ["t__abc__env"])
        finally:
            for child in pids:
                child.kill()
                child.wait()


class EgressScan(unittest.TestCase):
    def trial_with(self, tmp, *items):
        trial = Path(tmp) / "job/trial"
        shutil.copytree(TRIAL.parent, trial.parent)
        rollout = next((trial / "agent/sessions").rglob("rollout-*.jsonl"))
        with rollout.open("a") as stream:
            for payload in items:
                stream.write(json.dumps({"timestamp": "t", "type": "response_item", "payload": payload}) + "\n")
        return trial

    def test_stub_proof_rollout_is_clean(self):
        import egress_scan
        scan = egress_scan.scan_trial(TRIAL)
        self.assertEqual((scan["flagged"], scan["hits"]), (False, []))
        self.assertEqual(scan["calls_scanned"], 1)
        self.assertFalse(r.build_job("t", 1, 0, *stub_job(), {}, ARGS, 1.0, 2.0)["probe"]["egress_scan"]["flagged"])

    def test_every_listed_pattern_is_caught_in_call_arguments(self):
        import egress_scan
        commands = ["pip install requests", "python -m pip download x", "uv pip install y", "uv add z", "git fetch origin",
                    "git pull", "git clone https://x/y", "git ls-remote origin", "curl -s http://a", "wget http://a",
                    "cat github.com/x", "echo raw.githubusercontent.com", "python -c 'import pypi.org'", "git log --all",
                    "git log --oneline --all | head -5", "git log --oneline -S x --all"]
        with tempfile.TemporaryDirectory() as tmp:
            trial = self.trial_with(tmp, *[{"type": "function_call", "name": "exec_command",
                                            "arguments": json.dumps({"cmd": c})} for c in commands])
            scan = egress_scan.scan_trial(trial)
        self.assertTrue(scan["flagged"])
        self.assertEqual(len(scan["hits"]), len(commands))
        self.assertEqual(scan["calls_scanned"], len(commands) + 1)

    def test_custom_and_shell_call_shapes_are_scanned_but_prose_and_outputs_are_not(self):
        import egress_scan
        with tempfile.TemporaryDirectory() as tmp:
            trial = self.trial_with(
                tmp, {"type": "custom_tool_call", "name": "apply_patch", "input": "curl http://x"},
                {"type": "local_shell_call", "action": {"type": "exec", "command": ["wget", "http://x"]}},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I will not run pip install"}]},
                {"type": "function_call_output", "output": "git fetch github.com"})
            scan = egress_scan.scan_trial(trial)
        self.assertEqual(len(scan["hits"]), 2)

    def test_rescan_writes_flags_into_job_files_and_the_summary_lists_the_task(self):
        import egress_scan
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "c"
            attempt = campaign / "attempts/agent-0-pallets__click-3239-r1"
            trial = self.trial_with(attempt / "harbor", {"type": "function_call", "name": "exec_command",
                                                                 "arguments": json.dumps({"cmd": "git log --all"})})
            facts, stats = stub_job(trial)
            job = r.build_job("pallets__click-3239", 1, 0, facts, stats, {}, ARGS, 1.0, 2.0)
            job["probe"]["egress_scan"] = {"flagged": False, "hits": []}  # as if scanned before the rollout was complete
            for path in (attempt / "job.json", campaign / "job-agent-0-pallets__click-3239-r1.json"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(job))
            flags = egress_scan.rescan_campaign(campaign)
            self.assertEqual([f["task"] for f in flags], ["pallets__click-3239#r1"])
            self.assertTrue(json.loads((campaign / "egress-flags.json").read_text())[0]["hits"])
            rewritten = json.loads((campaign / "job-agent-0-pallets__click-3239-r1.json").read_text())
            self.assertTrue(rewritten["probe"]["egress_scan"]["flagged"])
            summary = r.campaign_summary([rewritten], [("pallets__click-3239", 1)], 1.0, {}, ARGS)
        self.assertEqual(summary["egress_flagged_tasks"], ["pallets__click-3239#r1"])


class DiskStop(unittest.TestCase):
    def test_queue_stops_before_a_task_when_free_space_is_below_the_floor(self):
        free = iter([40, 40, 3, 3, 3])
        ran, stops = [], []
        rows = r.run_queue(list("abcde"), 1, lambda agent, task: ran.append(task) or task, time.monotonic() + 60,
                           min_free_gb=5, free_gb=lambda: next(free), stops=stops)
        self.assertEqual((ran, rows), (["a", "b"], ["a", "b"]))
        self.assertEqual(len(stops), 1)
        self.assertTrue(stops[0].startswith("disk_low: 3 GiB < 5 GiB before c"), stops)

    def test_floor_zero_disables_the_check(self):
        ran = []
        r.run_queue(list("ab"), 1, lambda agent, task: ran.append(task), time.monotonic() + 60,
                    min_free_gb=0, free_gb=lambda: 0)
        self.assertEqual(ran, ["a", "b"])

    def test_verdict_names_the_stop_and_the_runner_exits_with_the_disk_code(self):
        jobs = [SummarySchema.jobs(SummarySchema(), calls("two-agent/llama-fix-v4a-agent1.jsonl"))]
        summary = r.campaign_verdict(jobs, [("pallets__click-3239", 1), ("x", 1)], 1.0, {}, ARGS,
                                     stops=["disk_low: 3 GiB < 5 GiB before x#r1"])
        self.assertFalse(summary["valid"])
        self.assertIn("disk_low: 3 GiB < 5 GiB before x#r1", summary["gates"])
        self.assertEqual(r.exit_code(summary), r.EXIT_DISK_LOW)
        self.assertEqual(r.exit_code({"valid": True, "gates": []}), 0)
        self.assertEqual(r.exit_code({"valid": False, "gates": ["budget"]}), 2)

    def test_shell_stage_maps_the_disk_exit_code_and_passes_the_floor(self):
        text = (HERE / "gpu_swe_fresh.sh").read_text()
        self.assertIn('--min-free-gb "$MIN_FREE_GB"', text)
        self.assertIn('[ "$rc" = 5 ] && fail "$name-disk-low" 5', text)


class Plumbing(unittest.TestCase):
    """Guards the pieces whose removal the judge's mutants left undetected."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.harbor_script = self.tmp / "fake.sh"
        self.harbor_script.write_text('#!/bin/bash\ncp -r "$1"/. "$2/"\n')
        self.campaign = self.tmp / "c"
        (self.campaign / "attempts").mkdir(parents=True)
        self.multipliers = []

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def command(self, task, base_url, jobs_dir, job_name, multiplier, args):
        self.multipliers.append(multiplier)
        target = Path(jobs_dir) / job_name
        target.mkdir(parents=True)
        return ["bash", str(self.harbor_script), str(TRIAL.parent), str(target)]

    def snapshot(self, timeout):
        snap = self.tmp / "snap"
        (snap / "pallets__click-1").mkdir(parents=True, exist_ok=True)
        if timeout is not None:
            (snap / "pallets__click-1/task.toml").write_text(f"[agent]\ntimeout_sec = {timeout}\n")
        else:
            (snap / "pallets__click-1/task.toml").write_text("[agent]\n")
        return snap

    def test_task_toml_timeout_feeds_the_multiplier(self):
        args = task_args(self.tmp, dataset_path=self.snapshot(1500))
        self.assertEqual(r.agent_timeout("pallets__click-1", args), 1500.0)
        self.assertEqual(r.agent_timeout("pallets__click-1", task_args(self.tmp, dataset_path=self.snapshot(None))), 3000.0)

    def test_run_task_passes_cap_over_task_timeout_to_harbor(self):
        from unittest import mock
        args = task_args(self.tmp, dataset_path=self.snapshot(1500), cap_s=1200)
        with mock.patch.object(r, "harbor_command", self.command):
            r.run_task(0, ("pallets__click-1", 1), args, self.campaign)
        self.assertEqual(self.multipliers, [0.8])

    def serve_counters(self):
        import http.server

        class Handler(http.server.BaseHTTPRequestHandler):
            hits = 0

            def do_GET(self):
                type(self).hits += 1
                body = (f"llamacpp:tokens_predicted_total {100 * type(self).hits}\n"
                        f"llamacpp:tokens_predicted_seconds_total {2 * type(self).hits}\n").encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_per_task_metrics_delta_is_the_counter_difference(self):
        from unittest import mock
        args = task_args(self.tmp, metrics_url=self.serve_counters())
        with mock.patch.object(r, "harbor_command", self.command):
            job = r.run_task(0, ("pallets__click-1", 1), args, self.campaign)
        self.assertEqual(job["probe"]["metrics_delta"], {"tokens_predicted_total": 100.0, "tokens_predicted_seconds_total": 2.0})

    def test_concurrent_agents_get_no_per_task_delta(self):
        from unittest import mock
        args = task_args(self.tmp, metrics_url=self.serve_counters(), agents=2)
        with mock.patch.object(r, "harbor_command", self.command):
            job = r.run_task(0, ("pallets__click-1", 1), args, self.campaign)
        self.assertEqual(job["probe"]["metrics_delta"], {})

    def test_web_search_stays_disabled_in_the_codex_config(self):
        for provider in ("builtin", "local"):
            args = types.SimpleNamespace(**{**vars(ARGS), "provider": provider})
            self.assertEqual(r.codex_config(args)["web_search"], "disabled")
            command = r.harbor_command("t", "http://x/v1", Path("/o"), "j", 1.0, args)
            config = json.loads(next(a for a in command if a.startswith("config="))[7:])
            self.assertEqual(config["web_search"], "disabled")

    def jobs(self, **kw):
        good = SummarySchema.jobs(SummarySchema(), calls("two-agent/llama-fix-v4a-agent1.jsonl"))
        good["call_stats"]["completed_calls"] = kw.get("completed", 5)
        good["probe"]["agent_stats"]["tool_calls"] = kw.get("tools", 3)
        good["status"] = kw.get("status", "unsolved")
        return good

    def test_canary_plumbing_gates(self):
        self.assertEqual(r.canary_failures([self.jobs()]), [])
        self.assertEqual(len(r.canary_failures([self.jobs(completed=1)])), 1)
        self.assertEqual(len(r.canary_failures([self.jobs(tools=0)])), 1)
        self.assertEqual(len(r.canary_failures([self.jobs(status="invalid")])), 1)

    def test_verdict_applies_canary_budget_and_counter_gates(self):
        items = [("pallets__click-3239", 1)]
        base = dict(canary=False, over_budget=False, expect_counters=False)
        ok = r.campaign_verdict([self.jobs()], items, 1.0, {}, ARGS, **base)
        self.assertTrue(ok["valid"], ok["gates"])
        bad_canary = r.campaign_verdict([self.jobs(tools=0)], items, 1.0, {}, ARGS, **{**base, "canary": True})
        self.assertFalse(bad_canary["valid"])
        self.assertTrue(any(g.startswith("canary_plumbing_failed") for g in bad_canary["gates"]))
        self.assertTrue(r.campaign_verdict([self.jobs(tools=0)], items, 1.0, {}, ARGS, **base)["valid"])
        late = r.campaign_verdict([self.jobs()], items, 1.0, {}, ARGS, **{**base, "over_budget": True})
        self.assertEqual((late["valid"], "budget" in late["gates"], late["state"]), (False, True, "invalid"))
        blind = r.campaign_verdict([self.jobs()], items, 1.0, {}, ARGS, **{**base, "expect_counters": True})
        self.assertEqual((blind["valid"], "engine_counters_unavailable" in blind["gates"]), (False, True))
        seen = r.campaign_verdict([self.jobs()], items, 1.0, {"tokens_predicted_total": 1.0}, ARGS,
                                  **{**base, "expect_counters": True})
        self.assertTrue(seen["valid"], seen["gates"])

    def test_queue_honours_the_deadline(self):
        ran = []
        r.run_queue(list("abc"), 1, lambda agent, task: ran.append(task), time.monotonic() - 1)
        self.assertEqual(ran, [])
        ran.clear()
        r.run_queue(list(range(50)), 1, lambda agent, task: ran.append(task) or time.sleep(0.05), time.monotonic() + 0.2)
        self.assertTrue(1 <= len(ran) < 50, len(ran))

    def test_queue_stops_when_the_stop_event_is_set(self):
        ran, stop = [], threading.Event()
        r.run_queue(list(range(10)), 1, lambda agent, task: ran.append(task) or stop.set(), time.monotonic() + 60, stop=stop)
        self.assertEqual(ran, [0])

    def test_two_agents_share_one_queue(self):
        ran = []
        rows = r.run_queue(list(range(6)), 2, lambda agent, task: ran.append(task) or task, time.monotonic() + 60)
        self.assertEqual(sorted(rows), list(range(6)))


class RunCommand(unittest.TestCase):
    """r.main() end to end with the queue stubbed: the plumbing from CLI flags to the summary verdict."""

    def run_main(self, *flags, queue_result=None, stops=()):
        import contextlib
        import io
        from unittest import mock

        def fake_queue(items, agents, execute, deadline, **kw):
            self.queue_kwargs = kw
            kw["stops"].extend(stops)
            return queue_result or []

        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "c"
            argv = ["run_swe_fresh.py", "run", "--campaign", str(campaign), "--tasks", "pallets__click-3239",
                    "--upstream-port", "1", *flags]
            with mock.patch("sys.argv", argv), mock.patch.object(r, "run_queue", fake_queue), \
                    mock.patch.object(r, "harbor_version_command", lambda args: ["true"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = r.main()
            return code, json.loads((campaign / "SUMMARY.json").read_text())

    def test_exhausted_budget_is_a_gate(self):
        code, summary = self.run_main("--budget-s", "0")
        self.assertIn("budget", summary["gates"])
        self.assertEqual((summary["valid"], code), (False, 2))

    def test_unreachable_metrics_endpoint_is_a_gate_only_when_one_was_asked_for(self):
        _, summary = self.run_main("--metrics-url", "http://127.0.0.1:1")
        self.assertIn("engine_counters_unavailable", summary["gates"])
        _, summary = self.run_main()
        self.assertNotIn("engine_counters_unavailable", summary["gates"])

    def test_canary_flag_applies_the_plumbing_check(self):
        job = Plumbing.jobs(Plumbing(), tools=0)
        _, summary = self.run_main("--canary", queue_result=[job])
        self.assertTrue(any(g.startswith("canary_plumbing_failed") for g in summary["gates"]))
        _, summary = self.run_main(queue_result=[job])
        self.assertFalse(any(g.startswith("canary_plumbing_failed") for g in summary["gates"]))

    def test_disk_floor_flag_reaches_the_queue_and_a_stop_exits_with_the_disk_code(self):
        code, summary = self.run_main("--min-free-gb", "5", stops=["disk_low: 3 GiB < 5 GiB before x#r1"])
        self.assertEqual(self.queue_kwargs["min_free_gb"], 5)
        self.assertEqual(code, r.EXIT_DISK_LOW)
        self.assertIn("disk_low: 3 GiB < 5 GiB before x#r1", summary["gates"])


class MoreEdges(unittest.TestCase):
    def test_other_exception_is_never_solved_even_with_reward_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "job/trial"
            shutil.copytree(TRIAL.parent, trial.parent)
            document = json.loads((trial / "result.json").read_text())
            document["verifier_result"]["rewards"]["reward"] = 1.0
            document["exception_info"] = {"exception_type": "RuntimeError", "exception_message": "boom"}
            (trial / "result.json").write_text(json.dumps(document))
            self.assertFalse(r.parse_trial(trial)["solved"])
            document["exception_info"]["exception_type"] = "AgentTimeoutError"
            (trial / "result.json").write_text(json.dumps(document))
            facts = r.parse_trial(trial)
            self.assertTrue(facts["solved"] and facts["timed_out"])

    def test_long_call_cut_by_the_kill_is_in_flight_if_it_ends_in_a_disconnect(self):
        cut = dict(started=AGENT_START + 100, finished=AGENT_START + CAP + 2)
        facts, stats = timeout_case(cut_rows(**cut))
        self.assertEqual(r.classify(facts, stats, CAP)[0], "timeout")
        facts, stats = timeout_case(cut_rows(error="ReadTimeout: x", **cut))
        self.assertEqual(r.classify(facts, stats, CAP)[0], "invalid")

    def test_relay_error_row_names_the_exception(self):
        proxy = relay.UsageProxy(1, Path(tempfile.mkdtemp()), 0, "t#r1", time.monotonic() + 30)
        proxy.start()
        try:
            RelayEndToEnd.post(RelayEndToEnd, proxy.port, {"input": []})
        except (http.client.HTTPException, OSError):
            pass
        rows = proxy.stop()
        self.assertTrue(rows[0]["error"].startswith("ConnectionRefusedError"), rows[0].get("error"))

    def test_relay_cuts_a_stream_that_outlives_the_deadline(self):
        import socket
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def endless():
            connection, _ = listener.accept()
            connection.recv(65536)
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            try:
                while True:
                    connection.sendall(b"data: {}\n\n")
                    time.sleep(0.1)
            except OSError:
                pass

        threading.Thread(target=endless, daemon=True).start()
        out = Path(tempfile.mkdtemp())
        proxy = relay.UsageProxy(listener.getsockname()[1], out, 0, "t#r1", time.monotonic() + 1.0)
        proxy.start()
        started = time.monotonic()
        try:
            RelayEndToEnd.post(RelayEndToEnd, proxy.port, {"input": []})
        except (http.client.HTTPException, OSError):
            pass
        rows = proxy.stop()
        listener.close()
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(rows[0]["error"].startswith("TimeoutError"), rows[0].get("error"))


CANARY = FIX / "cal-llama-canary"  # real gpuq canary 2026-10-04 (llama, 300 s cap): rows 49-53, trimmed rollout and result


def canary_case(rows=None, trial=CANARY / "job/trial", cap_s=CAP):
    """The canary through the pipeline run_task uses: facts from the trial, statistics against the cap."""
    facts = r.parse_trial(trial, cap_s)
    rows = calls("cal-llama-canary/model-calls.jsonl") if rows is None else rows
    return facts, r.call_statistics(rows, r.inflight_cutoff(facts, cap_s), r.cap_end_epoch(facts, cap_s))


class RealCanaryCapCut(unittest.TestCase):
    """Call 53 started 16.9 s after the cap and died with BrokenPipe when the container was torn down; the
    verifier had given reward 1 five seconds after Harbor killed the exec client. It scored `invalid: call_failed [53]`."""

    def test_fixture_is_the_observed_shape(self):
        facts, stats = canary_case()
        self.assertTrue(facts["timed_out"] and facts["solved"])
        rows = calls("cal-llama-canary/model-calls.jsonl")
        self.assertEqual([c["call"] for c in rows], [49, 50, 51, 52, 53])
        self.assertTrue(rows[-1]["error"].startswith("BrokenPipeError") and rows[-1].get("event") is None)
        self.assertGreater(rows[-1]["started_epoch"], facts["agent_started_epoch"] + CAP)
        # call-0053.response ends mid-SSE: no terminal event, so the row cannot be a completed call
        call = {}
        for line in (CANARY / "call-0053.response").read_bytes().split(b"\n"):
            relay.ProxyHandler.scan_line(line, call)
        self.assertNotIn("event", call)

    def test_cut_call_is_not_a_failure_even_with_a_solved_verdict(self):
        # old-signature statistics (no cap epoch): the call is failed, and only the in-flight rule can excuse it
        facts = r.parse_trial(CANARY / "job/trial")
        stats = r.call_statistics(calls("cal-llama-canary/model-calls.jsonl"), r.inflight_cutoff(facts, CAP))
        self.assertEqual(stats["failed_calls"], [53])
        status, gates = r.classify(facts, stats, CAP)
        self.assertFalse(any(g.startswith("call_failed") for g in gates), gates)

    def test_canary_scores_timeout_not_invalid(self):
        facts, stats = canary_case()
        self.assertEqual(r.classify(facts, stats, CAP), ("timeout", []))
        self.assertEqual(stats["post_cap_calls"], [51, 52, 53])
        self.assertEqual(stats["post_cap_served_calls"], [51, 52, 53])  # 53 reached the model: served, then cut
        self.assertEqual((stats["calls"], stats["failed_calls"]), (2, []))

    def test_post_cap_calls_leave_the_scored_statistics(self):
        _, stats = canary_case()
        self.assertEqual(stats["completed_calls"], 2)
        self.assertEqual(stats["peak_input_tokens"], max(c["input_tokens"] for c in calls("cal-llama-canary/model-calls.jsonl")[:2]))

    def test_job_records_the_uncredited_reward_and_censors_the_wall(self):
        facts, stats = canary_case()
        args = types.SimpleNamespace(**{**vars(ARGS), "cap_s": CAP})
        job = r.build_job("pallets__click-3239", 1, 0, facts, stats, {}, args, 1.0, 2.0)
        probe = job["probe"]
        self.assertEqual(job["status"], "timeout")
        self.assertEqual((probe["reward"], probe["task_success"], probe["censored"]), (1, False, True))
        self.assertEqual(probe["verified_wall_seconds"], CAP)
        self.assertEqual(probe["post_cap"], {"calls": [51, 52, 53], "served_calls": [51, 52, 53], "tool_calls": 0,
                                             "reward_not_credited": True})
        self.assertEqual(probe["ignored_inflight_calls"], [])
        summary = r.campaign_summary([job], [("pallets__click-3239", 1)], 3.0, {}, args)
        self.assertEqual(summary["post_cap_activity_tasks"], ["pallets__click-3239#r1"])
        self.assertEqual((summary["timeouts"], summary["solved"], summary["invalid_tasks"]), (1, 0, 0))

    def test_solved_timeout_without_post_cap_activity_is_passed_but_censored_at_the_cap(self):
        facts, stats = canary_case(calls("cal-llama-canary/model-calls.jsonl")[:2])
        facts["agent_wall_seconds"] = CAP + 0.46
        args = types.SimpleNamespace(**{**vars(ARGS), "cap_s": CAP})
        probe = r.build_job("t", 1, 0, facts, stats, {}, args, 1.0, 2.0)["probe"]
        self.assertEqual((probe["task_success"], probe["verified_wall_seconds"], probe["censored"]), (True, CAP, True))

    def test_ignored_inflight_calls_are_listed_for_a_solved_timeout_too(self):
        rows = calls("cal-llama-canary/model-calls.jsonl")
        facts = r.parse_trial(CANARY / "job/trial", CAP)
        start = facts["agent_started_epoch"]
        cut = dict(rows[-1], started_epoch=start + CAP - 5, finished_epoch=start + CAP + 2)
        _, stats = canary_case(rows[:2] + [cut])
        args = types.SimpleNamespace(**{**vars(ARGS), "cap_s": CAP})
        job = r.build_job("t", 1, 0, facts, stats, {}, args, 1.0, 2.0)
        self.assertEqual((job["status"], job["probe"]["ignored_inflight_calls"]), ("passed", [53]))

    def test_failure_well_before_the_cap_still_invalidates_negative_control(self):
        rows = calls("cal-llama-canary/model-calls.jsonl")
        facts = r.parse_trial(CANARY / "job/trial", CAP)
        early = dict(rows[-1], started_epoch=facts["agent_started_epoch"] + 100,
                     finished_epoch=facts["agent_started_epoch"] + 101)
        for error in ("BrokenPipeError: [Errno 32] Broken pipe", "ConnectionRefusedError: x"):
            _, stats = canary_case(rows[:-1] + [dict(early, error=error)])
            status, gates = r.classify(facts, stats, CAP)
            self.assertEqual((status, gates), ("invalid", ["call_failed: [53]"]), error)

    def test_cut_call_ending_after_the_cap_scores_timeout_when_the_run_was_solved_before_it(self):
        rows = calls("cal-llama-canary/model-calls.jsonl")
        facts = r.parse_trial(CANARY / "job/trial", CAP)
        start = facts["agent_started_epoch"]
        cut = dict(rows[-1], started_epoch=start + CAP - 5, finished_epoch=start + CAP + 2)
        _, stats = canary_case(rows[:2] + [cut])
        self.assertEqual(stats["inflight_at_cap_calls"], [53])
        self.assertEqual(r.classify(facts, stats, CAP), ("passed", []))  # no activity after the cap: the solve stands
        facts_unsolved = dict(facts, solved=False, reward=0.0)
        self.assertEqual(r.classify(facts_unsolved, stats, CAP), ("timeout", []))


class PostCapEdits(unittest.TestCase):
    """D1: Harbor kills only the host `docker compose exec` client; Codex keeps working while the verifier runs."""

    def trial_with_rollout_call(self, tmp, offset_s):
        trial = Path(tmp) / "job/trial"
        shutil.copytree(CANARY / "job", trial.parent)
        facts = r.parse_trial(trial)
        rollout = next((trial / "agent/sessions").rglob("rollout-*.jsonl"))
        row = json.loads(rollout.read_text().splitlines()[-1])
        row["timestamp"] = datetime.fromtimestamp(facts["agent_started_epoch"] + offset_s, timezone.utc).isoformat()
        with rollout.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return trial

    def pre_cap_rows(self):
        return calls("cal-llama-canary/model-calls.jsonl")[:2]

    def test_solved_run_without_post_cap_activity_keeps_its_solve(self):
        facts, stats = canary_case(self.pre_cap_rows())
        self.assertEqual((facts["post_cap_tool_calls"], stats["post_cap_calls"]), (0, []))
        self.assertEqual(r.classify(facts, stats, CAP), ("passed", []))

    def test_tool_call_after_the_cap_voids_the_solve(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = self.trial_with_rollout_call(tmp, CAP + 3)
            facts, stats = canary_case(self.pre_cap_rows(), trial)
            self.assertEqual(facts["post_cap_tool_calls"], 1)
            self.assertEqual(r.classify(facts, stats, CAP), ("timeout", []))

    def test_tool_call_before_the_cap_does_not_void_the_solve_negative_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = self.trial_with_rollout_call(tmp, CAP - 3)
            facts, stats = canary_case(self.pre_cap_rows(), trial)
            self.assertEqual(facts["post_cap_tool_calls"], 0)
            self.assertEqual(r.classify(facts, stats, CAP), ("passed", []))

    def test_without_the_cap_nothing_is_dated(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = self.trial_with_rollout_call(tmp, CAP + 3)
            self.assertEqual(r.parse_trial(trial)["post_cap_tool_calls"], 0)

    def test_a_model_call_served_after_the_cap_voids_the_solve(self):
        facts, stats = canary_case(calls("cal-llama-canary/model-calls.jsonl")[:3])
        self.assertEqual(stats["post_cap_served_calls"], [51])
        self.assertEqual(r.classify(facts, stats, CAP)[0], "timeout")

    def test_an_unsolved_run_is_unchanged_by_post_cap_calls(self):
        facts, stats = canary_case()
        self.assertEqual(r.classify(dict(facts, solved=False, reward=0.0), stats, CAP), ("timeout", []))

    def test_relay_rejects_a_call_starting_after_the_cap_and_never_forwards_it(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "stub").mkdir()
        (tmp / "relay").mkdir()
        upstream = stub.Server(("127.0.0.1", 0), stub.Handler)
        upstream.out, upstream.timings, upstream.turn = tmp / "stub", True, 0
        upstream.totals = {"draft": 0, "accepted": 0, "tokens": 0, "decode_s": 0.0}
        upstream.lock = threading.Lock()
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        proxy = relay.UsageProxy(upstream.server_address[1], tmp / "relay", 0, "t#r1", time.monotonic() + 30, cap_s=1.0)
        proxy.start()
        body = {"model": "m", "tools": [{"type": "function", "name": "exec_command"}], "stream": True,
                "input": [{"type": "message"}]}
        try:
            first = RelayEndToEnd.post(RelayEndToEnd, proxy.port, body)
            time.sleep(1.2)
            late = RelayEndToEnd.post(RelayEndToEnd, proxy.port, body)
            rows = proxy.stop()
            forwarded = len(list((tmp / "stub").glob("request-*.json")))
        finally:
            upstream.shutdown()
            upstream.server_close()
            shutil.rmtree(tmp)
        self.assertEqual((first[0], late[0]), (200, 403))
        self.assertEqual(forwarded, 1)
        self.assertTrue(rows[1]["post_cap"] and rows[1]["status"] == 403)
        stats = r.call_statistics(rows)
        self.assertEqual((stats["calls"], stats["completed_calls"], stats["failed_calls"]), (1, 1, []))
        self.assertEqual((stats["post_cap_calls"], stats["post_cap_served_calls"]), ([2], []))

    def test_relay_without_a_cap_serves_every_call_negative_control(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "relay").mkdir()
        proxy = relay.UsageProxy(1, tmp / "relay", 0, "t#r1", time.monotonic() + 30)
        proxy.start()
        try:
            for _ in range(2):
                try:
                    RelayEndToEnd.post(RelayEndToEnd, proxy.port, {"input": []})
                except (http.client.HTTPException, OSError):
                    pass
            rows = proxy.stop()
        finally:
            shutil.rmtree(tmp)
        self.assertEqual([c.get("post_cap") for c in rows], [None, None])
        self.assertTrue(all(c["error"].startswith("ConnectionRefusedError") for c in rows))

    def test_runner_hands_the_cap_to_the_relay_and_dates_the_trial_and_calls_against_it(self):
        text = (HERE / "run_swe_fresh.py").read_text()
        self.assertIn("cap_s=args.cap_s)", text)
        self.assertIn("parse_trial(find_trial(out / \"harbor\" / job_name), args.cap_s)", text)
        self.assertIn("call_statistics(calls, inflight_cutoff(facts, args.cap_s), cap_end_epoch(facts, args.cap_s))", text)

    def test_relay_after_the_cap_still_forwards_non_model_requests(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "relay").mkdir()
        proxy = relay.UsageProxy(1, tmp / "relay", 0, "t#r1", time.monotonic() + 30, cap_s=0.3)
        proxy.start()
        try:
            for method, path in (("POST", "/v1/responses"), ("GET", "/v1/models")):
                time.sleep(0.5)
                connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
                try:
                    connection.request(method, path, "{}")
                    connection.getresponse().read()
                except (http.client.HTTPException, OSError):
                    pass
                finally:
                    connection.close()
            rows = proxy.stop()
        finally:
            shutil.rmtree(tmp)
        self.assertEqual([c.get("post_cap") for c in rows], [None, None])


class CanaryProofSurvivesRetry(unittest.TestCase):
    """D4: gpuq's fresh retry renames OUT to OUT.void-a<N> before the stage starts."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "cal-v2"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def canary(self, directory, valid=True, pins="abc"):
        (directory / "canary").mkdir(parents=True)
        (directory / "canary" / "SUMMARY.json").write_text(json.dumps({"valid": valid}))
        (directory / "canary.pins").write_text(pins + "\n")

    def void(self, attempt):
        self.out.rename(self.tmp / f"cal-v2.void-a{attempt}")

    def test_proof_in_out_is_found_in_place(self):
        self.canary(self.out)
        self.assertEqual(r.adopt_canary_proof(self.out, "abc"), self.out)
        self.assertFalse((self.out / "canary-proof").exists())

    def test_proof_moved_by_a_fresh_retry_is_found_and_copied_into_the_new_out(self):
        self.canary(self.out)
        self.void(1)
        self.out.mkdir()
        source = r.adopt_canary_proof(self.out, "abc")
        self.assertEqual(source, self.tmp / "cal-v2.void-a1")
        proof = self.out / "canary-proof"
        self.assertTrue(json.loads((proof / "SUMMARY.json").read_text())["valid"])
        self.assertEqual((proof / "canary.pins").read_text().strip(), "abc")
        self.assertIn("cal-v2.void-a1", (proof / "source.txt").read_text())

    def test_second_retry_still_finds_the_canary_in_the_first_void(self):
        self.canary(self.out)
        self.void(1)
        (self.out / "measure").mkdir(parents=True)
        self.void(2)
        self.out.mkdir()
        self.assertEqual(r.canary_proof(self.out, "abc"), self.tmp / "cal-v2.void-a1")

    def test_newest_valid_proof_wins_and_invalid_or_foreign_ones_are_skipped(self):
        self.canary(self.out, valid=False)
        self.void(1)
        self.canary(self.out, pins="other")
        self.void(2)
        self.canary(self.out, pins="abc")
        self.void(3)
        self.out.mkdir()
        self.assertEqual(r.canary_proof(self.out, "abc"), self.tmp / "cal-v2.void-a3")

    def test_attempt_numbers_sort_numerically(self):
        for attempt in (1, 2, 10):
            self.canary(self.out)
            self.void(attempt)
        self.out.mkdir()
        self.assertEqual(r.canary_proof(self.out, "abc"), self.tmp / "cal-v2.void-a10")

    def test_changed_pins_or_a_failed_canary_find_nothing_negative_control(self):
        self.canary(self.out)
        self.void(1)
        self.out.mkdir()
        self.assertIsNone(r.adopt_canary_proof(self.out, "different"))
        self.assertFalse((self.out / "canary-proof").exists())
        (self.tmp / "cal-v2.void-a1" / "canary" / "SUMMARY.json").write_text(json.dumps({"valid": False}))
        self.assertIsNone(r.adopt_canary_proof(self.out, "abc"))

    def test_no_canary_anywhere_finds_nothing_and_does_not_confuse_a_sibling_out(self):
        self.out.mkdir()
        self.canary(self.tmp / "cal-v2-other")
        (self.tmp / "cal-v2-other").rename(self.tmp / "cal-v2-other.void-a1")
        self.assertIsNone(r.canary_proof(self.out, "abc"))

    def test_cli_exit_code_and_shell_wiring(self):
        self.canary(self.out)
        self.void(1)
        self.out.mkdir()
        run = lambda sha: subprocess.run([sys.executable, str(HERE / "run_swe_fresh.py"), "canary-proof", "--out",
                                          str(self.out), "--pins-sha", sha], capture_output=True, text=True)
        self.assertEqual(run("abc").returncode, 0)
        self.assertEqual(run("zzz").returncode, 1)
        text = (HERE / "gpu_swe_fresh.sh").read_text()
        self.assertIn('canary-proof --out "$OUT" --pins-sha "$PINS_SHA"', text)
        self.assertIn("|| fail canary-proof-missing 6", text)
        self.assertNotIn("""['valid'] is True""", text)


if __name__ == "__main__":
    unittest.main()
