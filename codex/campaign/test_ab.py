"""CPU tests: verification (incl. negative controls), canary proofs on the stub, metrics, summary robustness, orchestrator end to end on the stub, item manifests.
Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -m unittest test_ab -v   (never starts llama-server, never touches a GPU)
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / "repos/gpuq"))

import ab_canary as kn  # noqa: E402
import ab_common as kc  # noqa: E402
import ab_orchestrate as ko  # noqa: E402
import ab_summary as ks  # noqa: E402

ITEMS = {key: kc.EVAL / f"orchestration/gpuq/items/stack3-mask-ab-{key}.toml" for key in kc.MODELS}
ITEM = ITEMS["j5q"]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
for _k in [k for k in ENV if k.startswith("STUB_") or k.startswith("AB_STUB")]:
    ENV.pop(_k)


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def s5_launch():
    found = sorted((kc.WEEKEND / "results").glob("*/campaign-*phase1-S5-*pair1/attempts/*/launch.json"))
    for path in found:
        data = json.loads(path.read_text())
        if "--temp" in data["command"]:
            return data
    return None


def stub_campaign(arm, path, env=None, tasks=None, deadline=None, p=None):
    cmd = [sys.executable, str(HERE / "ab_campaign.py"), "--arm", arm, "--campaign", str(path), "--stub", "--port", str(p or port()),
           "--deadline-epoch", str(deadline or time.time() + 300)] + (["--tasks", tasks] if tasks else [])
    return subprocess.run(cmd, env=dict(ENV, **(env or {})), capture_output=True, text=True, timeout=300)


class Launch(unittest.TestCase):
    @unittest.skipUnless(s5_launch(), "no weekend S5 launch.json")
    def test_real_s5_control_launch_is_arm_a_only(self):
        data = s5_launch()
        fix = lambda argv: [a.replace("/frozen/stack2/", "/frozen/stack2-metafix/") for a in argv]  # the weekend S5 control ran the pre-metafix freeze
        kc.verify_launch(fix(data["command"]), "A", fix(data["executed_command"]))
        with self.assertRaises(kc.Fail):
            kc.verify_launch(data["command"], "A", data["executed_command"])
        with self.assertRaises(kc.Fail):  # B's argv[0] is its own runroot binary
            kc.verify_launch(fix(data["command"]), "B", fix(data["executed_command"]))

    def test_env_per_arm(self):
        kc.verify_env({}, "A")
        kc.verify_env({"GGML_SCHED_INPUT_STAGING": "1", "LD_LIBRARY_PATH": str(kc.ARMS["B"]["bin"])}, "B")
        for env, arm in (({"GGML_SCHED_INPUT_STAGING": "1"}, "A"), ({}, "B"), ({"LD_LIBRARY_PATH": str(kc.ARMS["B"]["bin"])}, "B"),
                         ({"GGML_SCHED_INPUT_STAGING": "1", "LD_LIBRARY_PATH": "/usr/lib"}, "B"),
                         ({"GGML_SCHED_INPUT_STAGING": "1", "LD_LIBRARY_PATH": str(kc.ARMS["B"]["bin"]), "GGML_SCHED_DEBUG": "1"}, "B"),
                         ({"GGML_SCHED_INPUT_STAGING": "1", "LD_LIBRARY_PATH": str(kc.ARMS["B"]["bin"]), "LLAMA_MTP_GRAPH_REUSE": "1"}, "B")):
            with self.assertRaises(kc.Fail, msg=(env, arm)) as caught:
                kc.verify_env(env, arm)
            self.assertEqual(caught.exception.rc, 6)

    def test_nice_must_not_be_inherited(self):
        kc.verify_nice(0)
        for bad in (1, 19, None):
            with self.assertRaises(kc.Fail) as caught:
                kc.verify_nice(bad)
            self.assertEqual(caught.exception.rc, 6)

    def test_executed_binary_is_never_named_llama_server(self):
        for arm, spec in kc.ARMS.items():
            self.assertNotEqual(Path(spec["exe"]).name, "llama-server")
            cmd = kn.server_argv(arm, kc.MODEL, 8081)
            kc.verify_launch(cmd, arm, [str(spec["exe"]), *cmd[1:]])
            with self.assertRaises(kc.Fail):
                kc.verify_launch(cmd, arm, [str(spec["runner_binary"]), *cmd[1:]])
            with self.assertRaises(kc.Fail):
                kc.verify_launch(cmd, arm, [str(kc.ARMS["A" if arm == "B" else "B"]["exe"]), *cmd[1:]])

    def mutate(self, **change):
        cmd = [a.replace("/frozen/stack2/", "/frozen/stack2-metafix/") for a in s5_launch()["command"]]
        for flag, value in change.items():
            flag = "--" + flag.replace("_", "-")
            if value is None:
                i = cmd.index(flag)
                del cmd[i:i + 2]
            else:
                cmd[cmd.index(flag) + 1] = value
        return cmd

    @unittest.skipUnless(s5_launch(), "no weekend S5 launch.json")
    def test_each_wrong_flag_is_rejected(self):
        for change in ({"spec_draft_n_max": "7"}, {"spec_draft_p_min": "0.6"}, {"spec_draft_n_min": "1"}, {"tensor_split": "69,31"},
                       {"spec_draft_vocab": None}, {"spec_draft_sampling": "greedy"}, {"temp": "0.7"}):
            with self.assertRaises(kc.Fail, msg=change) as caught:
                kc.verify_launch(self.mutate(**change), "A")
            self.assertEqual(caught.exception.rc, 6)
        duplicated = self.mutate() + ["--spec-draft-p-min", "0.6"]
        with self.assertRaises(kc.Fail):
            kc.verify_launch(duplicated, "A")
        with self.assertRaises(kc.Fail):
            kc.verify_launch(self.mutate(), "A", ["/usr/bin/llama-server"])
        with self.assertRaises(kc.Fail):
            kc.verify_launch(self.mutate() + ["--no-spec-draft-backend-sampling"], "A")

    def test_k_above_seven_is_never_allowed(self):
        self.assertTrue(all(a["k"] <= 7 for a in kc.ARMS.values()))

    def test_props_and_log_echo(self):
        kc.verify_props({"total_slots": 1, "default_generation_settings": {"n_ctx": 102400}})
        with self.assertRaises(kc.Fail):
            kc.verify_props({"total_slots": 2, "default_generation_settings": {"n_ctx": 102400}})
        with self.assertRaises(kc.Fail):
            kc.verify_props({"total_slots": 1, "default_generation_settings": {"n_ctx": 65536}})
        self.assertEqual(kc.verify_server_log("nothing here", "B"), 0)
        self.assertEqual(kc.verify_server_log("x n_max=5, n_min=0, p_min=0.00 y", "B"), 1)
        with self.assertRaises(kc.Fail):
            kc.verify_server_log("x n_max=7, n_min=0, p_min=0.00 y", "B")


class Counters(unittest.TestCase):
    def test_draft_length_check_table(self):
        check = lambda arm, steps, tokens: kc.draft_length_check(arm, steps, tokens)[0]
        for arm in "AB":
            self.assertEqual(check(arm, 1000, 5000), "ok")
            self.assertEqual(check(arm, 1000, 4000), "fail")
            self.assertEqual(check(arm, 1000, 5100), "fail")
            self.assertEqual(check(arm, 1000, 7000), "fail")  # k7 server
            self.assertEqual(check(arm, 100, 500), "insufficient")
            self.assertEqual(check(arm, None, None), "insufficient")

    def test_real_s5_samples_satisfy_arm_a(self):
        seen = 0
        for f in (kc.WEEKEND / "results").glob("*/campaign-*phase1-S5-*/job-*.json"):
            if f.name.endswith(".sha256"):
                continue
            m = (json.loads(f.read_text()).get("probe") or {}).get("metrics_delta") or {}
            if m.get("drafts", 0) >= kc.MIN_DRAFTS:
                self.assertEqual(kc.draft_length_check("A", m["drafts"], m["draft_tokens"])[0], "ok", f)
                seen += 1
        if not seen:
            self.skipTest("no weekend S5 jobs")

    def test_derived_rounds_match_server_truth(self):
        p = port()
        args = ["-c", "102400", "-np", "1", "--spec-draft-n-max", "7", "--spec-draft-p-min", "0.6", "--port", str(p)]
        server = subprocess.Popen([str(HERE / "stub_srv.py"), *args], stdout=subprocess.DEVNULL, env=ENV)
        try:
            base = f"http://127.0.0.1:{p}"
            for _ in range(100):
                try:
                    urllib.request.urlopen(base + "/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            for seed in range(5):
                req = urllib.request.Request(base + "/stub/generate", data=json.dumps({"n": 4000, "seed": seed}).encode())
                urllib.request.urlopen(req, timeout=30)
            truth = json.loads(urllib.request.urlopen(base + "/stub/truth").read())
            sys.path.insert(0, str(kc.EVAL))
            from scripts.instruments import emit
            m = emit.metrics_delta({k: 0 for k in ("prompt_tokens_total", "prompt_tokens_cached_total", "prompt_seconds_total", "tokens_predicted_total",
                                                  "tokens_predicted_seconds_total", "draft_tokens_total", "draft_accepted_total", "drafts_total")},
                                   emit.scrape_metrics(base))
            d = kc.derive(m)
            self.assertGreater(truth["plain"], 100)
            self.assertEqual(d["plain"], truth["plain"])
            self.assertEqual(d["rounds"], truth["rounds"])
        finally:
            server.terminate()
            server.wait(timeout=10)


class Summary(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kr-test-", dir=str(HERE / "runs") if (HERE / "runs").is_dir() else None))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def campaign(self, idx, arm, **kw):
        path = self.tmp / f"campaign-{idx}-{arm}"
        p = stub_campaign(arm, path, **kw)
        self.assertIn(p.returncode, (0, 1), p.stderr)
        return path

    def test_full_set_and_runaway_flag(self):
        self.campaign(1, "A")
        self.campaign(2, "B", env={"STUB_RUNAWAY_TASK": "rust/react"})
        self.campaign(3, "B")
        self.campaign(4, "A")
        rows, text = ks.summarize(self.tmp)
        by = {r["idx"]: r for r in rows}
        self.assertTrue(all(r["eligible"] for r in rows))
        self.assertEqual([t["task"] for t in by[2]["runaway_tasks"]], ["rust/react"])
        self.assertEqual(by[2]["runaways"], 1)
        self.assertNotEqual(by[2]["decode_tps"], by[2]["decode_tps_excl"])
        self.assertEqual(by[1]["decode_tps"], by[1]["decode_tps_excl"])
        self.assertIn("#1:", text)
        self.assertIn("#4:", text)
        comp = json.loads((self.tmp / "summary.json").read_text())["comparison"]
        self.assertEqual(len(comp["decode_tps"]["A"]["values"]), 2)
        self.assertEqual(len(comp["decode_tps"]["B"]["values"]), 2)

    def test_cap_cut_failed_and_damaged_inputs_do_not_crash(self):
        partial = self.campaign(1, "B", env={"STUB_TASK_SLEEP": "0.4"}, deadline=time.time() + 2)
        self.assertEqual(json.loads((partial / "state.json").read_text())["status"], "stopped")
        failed = self.campaign(2, "A", env={"STUB_FAIL_TASK": "rust/forth"})
        damaged = self.campaign(3, "A")
        (damaged / "job-solve-rust-xorcism.json").write_text("{not json")
        no_probe = self.campaign(4, "B")
        job = no_probe / "job-solve-rust-react.json"
        data = json.loads(job.read_text())
        data["probe"]["metrics_delta"] = None
        job.write_text(json.dumps(data))
        no_log = self.campaign(5, "B")
        for log in no_log.glob("attempts/*/server.log"):
            log.unlink()
        (self.tmp / "campaign-6-A").mkdir()
        (self.tmp / "campaign-7-B").mkdir()
        (self.tmp / "campaign-7-B" / "SUMMARY.json").write_text("[1, 2")
        rows, text = ks.summarize(self.tmp)
        by = {r["idx"]: r for r in rows}
        self.assertFalse(by[1]["eligible"])
        self.assertLess(by[1]["jobs"], 8)
        self.assertFalse(by[2]["eligible"])
        self.assertEqual(by[2]["failed"][0]["task"], "rust/forth")
        self.assertFalse(by[3]["eligible"])
        self.assertTrue(by[3]["errors"])
        self.assertFalse(by[4]["eligible"])
        self.assertEqual(by[5]["runaways"], 0)
        self.assertEqual(by[6]["jobs"], 0)
        self.assertIsNone(by[6]["decode_tps"])
        self.assertIn("n/a", text)
        json.loads((self.tmp / "summary.json").read_text())

    def test_empty_dir_and_single_arm(self):
        ks.summarize(self.tmp)
        self.campaign(1, "A")
        rows, text = ks.summarize(self.tmp)
        self.assertIn("n/a", text)
        self.assertIsNone(json.loads((self.tmp / "summary.json").read_text())["comparison"]["decode_tps"]["B"]["mean"])


class Orchestrator(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ab-test-", dir=str(HERE / "runs") if (HERE / "runs").is_dir() else None))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_orch(self, *extra, env=None, mode="run", stub=True):
        cmd = [sys.executable, str(HERE / "ab_orchestrate.py"), mode, "--out", str(self.tmp / "out"), "--port", str(port()), *(["--stub"] if stub else []), *extra]
        return subprocess.run(cmd, env=dict(ENV, AB_SCRATCH=str(self.tmp), **(env or {})), capture_output=True, text=True, timeout=900)

    def classify(self, log, rc, runtime=500, events="measure_start"):
        from gpuq.classify import classify
        from gpuq.manifest import parse
        from gpuq.config import load
        m = parse(ITEM, load())
        return classify(dict(manifest=m, rc=rc, runtime=runtime, events=events, log=log))[0]

    def test_full_plan_and_selftest(self):
        p = self.run_orch()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        for label in ("B without input staging", "B executed from the wrong binary", "B on the wrong root", "A ignores k", "positive control"):
            self.assertIn(label, p.stdout)
        rows = json.loads((self.tmp / "out/summary.json").read_text())["campaigns"]
        self.assertEqual([r["arm"] for r in rows], list("ABBA"))
        self.assertTrue(all(r["eligible"] for r in rows))
        self.assertEqual(self.classify(p.stdout, p.returncode), "ok")

    def abort_case(self, env, arm_idx, text, via):
        p = self.run_orch("--no-selftest", env=env)
        self.assertEqual(p.returncode, 6, p.stdout + p.stderr)
        self.assertIn(text, p.stdout)
        out = (self.tmp / "out/summary.md").read_text()
        self.assertIn(f"| {arm_idx} | B |", out)
        self.assertNotIn(f"| {arm_idx + 1} |", out)  # nothing ran after the bad arm
        self.assertEqual(self.classify(p.stdout, p.returncode), "integrity", via)

    def test_b_without_staging_aborts_before_any_measurement_is_trusted(self):
        self.abort_case({"STUB_B_NO_STAGING": "1"}, 2, "(env)", "staging")

    def test_b_run_from_the_wrong_binary_aborts(self):
        self.abort_case({"STUB_WRONG_EXE": "1"}, 2, "executed binary", "exe")

    def test_b_on_the_wrong_root_aborts(self):
        self.abort_case({"STUB_WRONG_ROOT": "1"}, 2, "manifest engine", "root")

    def test_k_ignored_aborts_via_counters(self):
        p = self.run_orch("--no-selftest", env={"STUB_FORCE_NMAX": "4", "STUB_LIE": "1"})
        self.assertEqual(p.returncode, 6, p.stdout)
        self.assertIn("(counters)", p.stdout)

    def test_selftest_catches_a_verifier_that_passes_everything(self):
        p = subprocess.run([sys.executable, "-c", (
            "import sys; sys.path.insert(0, %r); import ab_orchestrate as ko, ab_common as kc\n"
            "ko.verify_campaign = lambda *a, **k: {}\n"
            "ctx = ko.Ctx(%r, stub=True)\n"
            "try:\n    ko.negative_control(ctx)\nexcept kc.Fail as e:\n    print('FAIL', e.rc, e); sys.exit(0)\n"
            "sys.exit(1)") % (str(HERE), str(self.tmp / "nc"))], env=dict(ENV, AB_SCRATCH=str(self.tmp)), capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("was NOT detected", p.stdout)

    def test_stub_nice_inheritance_is_detected(self):
        # the tests themselves run niced, so the stub's recorded server nice is > 0 and the opt-in check must reject it
        p = self.run_orch("--no-selftest", "--plan", "A", env={"AB_STUB_CHECK_NICE": "1"})
        if os.getpriority(os.PRIO_PROCESS, 0) == 0:
            self.skipTest("test process is not niced")
        self.assertEqual(p.returncode, 6, p.stdout)
        self.assertIn("(nice)", p.stdout)

    def test_xid_in_journal_is_hazard(self):
        journal = self.tmp / "journal.txt"
        journal.write_text("Oct 04 03:00:00 host kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.\n")
        p = self.run_orch("--no-selftest", env={"AB_JOURNAL_FILE": str(journal)})
        self.assertEqual(p.returncode, 3, p.stdout)
        self.assertEqual(self.classify(p.stdout, p.returncode), "hazard")
        journal.write_text("Oct 04 03:00:00 host kernel: all quiet\n")
        self.assertEqual(self.run_orch("--no-selftest", "--plan", "A", env={"AB_JOURNAL_FILE": str(journal)}).returncode, 0)

    def test_xid_regex_is_word_bounded(self):
        self.assertTrue(kc.xid_clean(journal_file=self.write("kernel: NVRM: Xid Xidnotreal xidfoo")) is False)
        self.assertTrue(kc.xid_clean(journal_file=self.write("kernel: pxidx hxid fallen off the roadmap")))

    def write(self, text):
        path = self.tmp / "j.txt"
        path.write_text(text + "\n")
        return str(path)

    def test_no_time_at_all_is_not_a_crash(self):
        p = self.run_orch("--no-selftest", "--budget-s", "100")
        self.assertEqual(p.returncode, 5, p.stdout)
        self.assertTrue((self.tmp / "out/summary.md").exists())

    def test_cap_cut_campaign_is_partial_and_run_continues(self):
        env = {"AB_CAP_S": "8", "AB_CLEANUP_S": "3", "AB_MIN_START_S": "4", "STUB_TASK_SLEEP": "1.2"}
        p = self.run_orch("--no-selftest", "--plan", "AB", "--budget-s", "400", env=env)
        self.assertEqual(p.returncode, 5, p.stdout + p.stderr)
        rows = json.loads((self.tmp / "out/summary.json").read_text())["campaigns"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not r["eligible"] and r["state"] == "stopped" for r in rows))
        self.assertIn("capped, partial", p.stdout)
        self.assertIn("[excluded]", (self.tmp / "out/summary.md").read_text())

    def canary(self, env=None):
        return self.run_orch(mode="canary", env=env)

    def test_canary_stub_passes_and_proves_everything(self):
        p = self.canary()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("canary ok", p.stdout)
        self.assertEqual(self.classify(p.stdout, p.returncode, events=""), "ok")
        proof = json.loads((self.tmp / "out/canary-proofs.json").read_text())
        self.assertGreaterEqual(proof["B"]["prompt_tokens"], 33800)
        self.assertEqual(proof["A"]["prompt_sha"], proof["B"]["prompt_sha"])
        self.assertGreaterEqual(proof["B"]["r2"]["timings"]["cache_n"], 32768)
        self.assertLessEqual(proof["B"]["mask"]["max_kib"], kn.MAX_SUFFIX_KIB)
        self.assertGreaterEqual(proof["B"]["staging"]["input_fence_staged"], 1)
        self.assertEqual(proof["B"]["staging"]["input_fence_sync"], 0)
        self.assertEqual(proof["identity"]["tokens"], kn.N_PREDICT)
        launch_a = json.loads((self.tmp / "out/boot-A/launch.json").read_text())
        launch_b = json.loads((self.tmp / "out/boot-B/launch.json").read_text())
        self.assertNotIn("--verbosity", launch_a["command"])
        self.assertEqual(launch_b["command"][-2:], ["--verbosity", "5"])
        self.assertEqual(launch_a["executed_env"]["GGML_SCHED_INPUT_STAGING"], None)
        self.assertEqual(launch_b["executed_env"]["GGML_SCHED_INPUT_STAGING"], "1")

    def canary_fails(self, env, rc, text, klass):
        p = self.canary(env)
        self.assertEqual(p.returncode, rc, p.stdout[-1500:] + p.stderr[-500:])
        self.assertIn(text, p.stdout)
        self.assertEqual(self.classify(p.stdout, p.returncode, events=""), klass)

    def test_canary_rejects_full_mask_uploads(self):
        self.canary_fails({"STUB_B_FULL_MASK": "1"}, 6, "path-engaged proof failed", "integrity")

    def test_canary_rejects_missing_staging_counters(self):
        self.canary_fails({"STUB_NO_STAGED": "1"}, 6, "staging-engaged proof failed", "integrity")

    def test_canary_rejects_token_divergence(self):
        self.canary_fails({"STUB_DIVERGE": "1"}, 6, "cross-arm identity failed: tokens differ at 50", "integrity")

    def test_canary_rejects_draft_counter_divergence(self):
        self.canary_fails({"STUB_DRAFT_DIVERGE": "1"}, 6, "cross-arm identity failed: draft_n", "integrity")

    def test_canary_rejects_b_without_staging_env(self):
        self.canary_fails({"STUB_B_NO_STAGING": "1"}, 6, "staging", "integrity")

    def test_canary_short_prompt_is_insufficient_data_not_integrity(self):
        p = self.canary({"AB_PROMPT_TOKENS": "5000"})
        self.assertEqual(p.returncode, 5, p.stdout[-800:])
        self.assertEqual(self.classify(p.stdout, p.returncode, events=""), "insufficient_data")

    def test_canary_wall_cap_is_the_only_rc7(self):
        p = self.canary({"AB_CANARY_CAP_S": "20"})
        self.assertEqual(p.returncode, 7, p.stdout[-800:])
        self.assertIn("canary hit its wall cap", p.stdout)
        self.assertEqual(self.classify(p.stdout, p.returncode, events=""), "timeout")

    def test_canary_boot_failure_is_transient(self):
        self.canary_fails({"AB_BOOT_CAP_S": "0"}, 4, "not ready inside", "transient")

    def test_gate_functions_unit(self):
        log = self.tmp / "s.log"
        line = lambda size: f"## SPLIT #3: Meta [Meta(CUDA0,CUDA1)] # 2 inputs: [attn_inp_kq_mask ({size:>4})] [inp_tokens (  0K)]\n"
        log.write_text(line("9M") + "marker\n")
        mark = log.stat().st_size
        log.write_text(log.read_text() + line("0K") + line("1K") + line("3K") + "other [attn_inp_kq_mask (99M)] not a split\n")
        self.assertEqual(kn.mask_uploads(log), [9216, 0, 1, 3])
        self.assertEqual(kn.mask_uploads(log, mark), [0, 1, 3])
        self.assertEqual(kn.check_compact_masks([0, 1, 3, 3])["max_kib"], 3)
        for bad in ([0, 64], [1024 * 9, 1]):
            with self.assertRaises(kc.Fail) as caught:
                kn.check_compact_masks(bad)
            self.assertEqual(caught.exception.rc, 6)
        with self.assertRaises(kc.Fail) as caught:
            kn.check_compact_masks([1])
        self.assertEqual(caught.exception.rc, 5)
        self.assertEqual(kn.check_staging({"input_fence_staged": 10, "input_fence_sync": 0})["input_fence_staged"], 10)
        self.assertEqual(kn.check_staging({"input_fence_staged": 10, "input_fence_sync": 10})["input_fence_sync"], 10)
        for counts, rc in (({"input_fence_sync": 9}, 6), ({"input_fence_staged": 2, "input_fence_sync": 3}, 6), ({}, 5)):
            with self.assertRaises(kc.Fail) as caught:
                kn.check_staging(counts)
            self.assertEqual(caught.exception.rc, rc)

    def test_term_stops_own_child_and_frees_its_port(self):
        out, p = self.tmp / "out", port()
        cmd = [sys.executable, str(HERE / "ab_orchestrate.py"), "run", "--stub", "--no-selftest", "--out", str(out), "--port", str(p), "--plan", "A"]
        proc = subprocess.Popen(cmd, env=dict(ENV, STUB_TASK_SLEEP="5"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=0.5)
                break
            except OSError:
                time.sleep(0.2)
        else:
            self.fail("stub server never came up")
        proc.terminate()
        output, _ = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 130, output)
        time.sleep(1.0)
        with self.assertRaises(OSError):
            urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=0.5)


class RealRoots(unittest.TestCase):
    """The frozen roots as they are on disk (no GPU, no server): the checks the real run starts with."""

    def test_frozen_checks_pass_for_both_arms(self):
        for arm in kc.ARMS:
            ko.frozen_checks(arm)

    def test_reported_hashes(self):
        b = kc.ARMS["B"]["root"]
        self.assertEqual(kc.sha(b / "SOURCE"), "cb2327c2a42e52d3afdae18e7da5a7f175e42a16056fbf12757db7208a3fce08")
        self.assertEqual(kc.sha(b / "bin/SHA256SUMS"), "e98a3e498fa551959a417907e4d76b34f21c69a2eb6eb57a57b7b982565077d5")

    def test_tampered_pinned_sums_are_rejected(self):
        name = str(kc.ARMS["B"]["root"] / "SHA256SUMS")
        saved = dict(kc.PINNED_ROOT_SUMS)
        try:
            ko.PINNED_ROOT_SUMS[name] = "0" * 64 + " SOURCE\n"
            with self.assertRaises(kc.Fail) as caught:
                ko.frozen_checks("B")
            self.assertEqual(caught.exception.rc, 6)
        finally:
            ko.PINNED_ROOT_SUMS.clear()
            ko.PINNED_ROOT_SUMS.update(saved)

    def test_runroot_is_the_frozen_b_binary_not_a_copy(self):
        spec = kc.ARMS["B"]
        self.assertTrue(Path(spec["runner_binary"]).is_symlink())
        self.assertEqual(Path(spec["runner_binary"]).resolve(), Path(spec["exe"]).resolve())
        self.assertEqual(Path(spec["exe"]).name, "round-trace")

    def test_sizes(self):
        for key, spec in kc.MODELS.items():
            self.assertEqual(spec["path"].stat().st_size, spec["bytes"], key)


class Item(unittest.TestCase):
    def test_gpuq_check_and_rules(self):
        for key, item in ITEMS.items():
            p = subprocess.run([str(Path.home() / "repos/gpuq/bin/gpuq"), "submit", "--check", str(item)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("valid", p.stdout)
            self.assertNotIn("warning", p.stderr)

    def test_contract(self):
        for key, item in ITEMS.items():
            m = tomllib.loads(item.read_text())
            self.assertEqual(m["id"], f"stack3-mask-ab-{key}")
            self.assertEqual(m["run"]["env"]["AB_ITEM"], key)
            self.assertEqual(m["canary"]["env"]["AB_ITEM"], key)
            self.assertIn("{attempt}", m["outputs"]["paths"][0])
            self.assertIn("{attempt}", m["run"]["env"]["OUT"])
            self.assertIn("{attempt}", m["canary"]["env"]["OUT"])
            for stage in ("run", "preflight", "canary"):
                self.assertNotIn("CUDA_VISIBLE_DEVICES", m[stage].get("env", {}))
            self.assertEqual(m["exit_codes"]["7"], "timeout")
            self.assertEqual(m["exit_codes"]["3"], "hazard")
            self.assertEqual(m["run"]["env"]["XID_SINCE"], "2026-10-03 18:57:00")
            self.assertEqual(m["canary"]["env"]["XID_SINCE"], "2026-10-03 18:57:00")
            sig = {s["class"]: s["regex"] for s in m["signatures"]}
            self.assertEqual(sig["hazard"], r"\bXid\b|fallen off the bus")
            self.assertLessEqual(m["est_s"], m["budget_s"])
            self.assertLessEqual(m["budget_s"], 3.0 * 3600)
            self.assertEqual(m["budget_s"], int(m["run"]["env"]["BUDGET_S"]) + m["canary"]["timeout_s"] + 150)
            self.assertEqual(m["run"]["ports"], [8081])
            self.assertTrue(m["ab"])

    def test_pins_exist_and_cover_inputs(self):
        for key, item in ITEMS.items():
            pins = set(tomllib.loads(item.read_text())["pins"]["files"])
            for path in pins:
                self.assertTrue(Path(path).exists(), path)
            a, b = kc.ARMS["A"], kc.ARMS["B"]
            for must in (kc.MODELS[key]["path"], kc.IDS, a["exe"], a["runner_binary"], a["bin"] / "libggml-cuda.so", a["bin"] / "SHA256SUMS",
                         b["exe"], b["root"] / "SHA256SUMS", b["root"] / "SOURCE", b["bin"] / "SHA256SUMS", b["bin"] / "libggml-cuda.so",
                         b["bin"] / "libllama.so", b["runner_root"] / "RUNROOT.SHA256SUMS", b["runner_root"] / "gguf-py/gguf/constants.py",
                         kc.RUNNER, kc.ASSETS / "tasks.json", HERE / "run.sh", HERE / "make_runroot.sh", *HERE.glob("*.py")):
                if Path(must).name == "gen_item.py":
                    continue
                self.assertIn(str(must), pins, must)
            for name in ("hillclimb.py", "agentic_campaign.py", "swift_longtail.py", "agentic_probe.py"):
                self.assertIn(str(kc.EVAL / "scripts/instruments" / name), pins)
            other = kc.MODELS["iq4" if key == "j5q" else "j5q"]["path"]
            self.assertNotIn(str(other), pins)

    def test_items_are_current(self):
        p = subprocess.run([sys.executable, str(HERE / "gen_item.py"), "--check"], capture_output=True, text=True, env=ENV)
        self.assertEqual(p.returncode, 0, "run gen_item.py; an item lags its pinned inputs: " + p.stderr[-300:])

    def test_budget_arithmetic(self):
        import gen_item
        for key, item in ITEMS.items():
            m = tomllib.loads(item.read_text())
            b = gen_item.budget(key)
            self.assertGreaterEqual(b["run_s"], 4 * gen_item.WALL_S[key] * 1.25 + 150)
            self.assertEqual(m["budget_s"], b["budget_s"])
            self.assertGreaterEqual(int(m["run"]["env"]["BUDGET_S"]) - 60, ko.MIN_START_S)  # the first campaign always starts
            # four campaigns at the slowest measured wall must fit inside the run budget without touching the margin
            self.assertLessEqual(4 * gen_item.WALL_S[key] + ko.CLEANUP_S, int(m["run"]["env"]["BUDGET_S"]))
            self.assertEqual(m["canary"]["timeout_s"], ko.CANARY_CAP_S + 50)
            self.assertLessEqual(m["canary"]["timeout_s"], 900)

    def test_no_forbidden_process_names_or_commands(self):
        for path in HERE.glob("*.py"):
            if path.name == "test_ab.py":
                continue
            self.assertNotRegex(path.read_text(), r"pgrep\s+-f|pkill\s+-f|pkill |killall")
        text = (HERE / "run.sh").read_text()
        self.assertNotRegex(text, r"pgrep|pkill|killall")
        run = [line for line in text.splitlines() if "ab_orchestrate.py\" run" in line or "ab_orchestrate.py\" canary" in line]
        self.assertEqual(len(run), 2)
        self.assertTrue(all("nice" not in line for line in run), "the server would inherit nice")

    def test_real_main_path_builds_the_manifest_per_arm_and_model(self):
        code = r"""
import sys, tempfile, shutil
sys.path.insert(0, sys.argv[1])
import ab_campaign as kc
import ab_common as common
from pathlib import Path
arm = sys.argv[2]
spec = common.ARMS[arm]
class A: pass
args = A(); args.max_attempts = 0; args.probe_timeout = 0; args.arm = arm
h, s, jobs = kc.configure(arm)
kc.install(h, args)
assert h.subprocess.check_output(["git", "-C", str(spec["runner_root"]), "rev-parse", "HEAD"], text=True).strip() == spec["commit"]
s.sha256 = lambda path: s.MODEL["sha256"]
s.agentic.AgenticRunner.run = lambda self: None
tmp = Path(tempfile.mkdtemp(prefix="ab-main-", dir=sys.argv[3])); camp = tmp / "c"
sys.argv = ["x", "--campaign", str(camp), "--assets", str(kc.ASSETS), "--deadline-epoch", "0"]
s.main()
manifest = h.read_sealed(camp / "manifest.json")
assert manifest["server_extra"] == kc.EXTRA, manifest["server_extra"]
assert [j["configs"][0]["maximum"] for j in manifest["queue"]] == [5] * 8
assert [j["configs"][0]["p_min"] for j in manifest["queue"]] == [0] * 8
assert manifest["engine_commit"] == spec["commit"] and manifest["binary"] == str(spec["runner_binary"])
assert str(Path(spec["runner_binary"]).resolve()) in manifest["binary_pins"]
assert [m["id"] for m in manifest["models"] if m["id"] == common.MODEL_ID], manifest["models"]
assert Path([m for m in manifest["models"] if m["id"] == common.MODEL_ID][0]["path"]) == common.MODEL
shutil.rmtree(tmp)
print("manifest", arm, common.ITEM_KEY, "ok")
"""
        tmp = HERE / "runs"
        tmp.mkdir(exist_ok=True)
        for key in kc.MODELS:
            for arm in "AB":  # one arm per process: the runner modules hold global state
                p = subprocess.run([sys.executable, "-c", code, str(HERE), arm, str(tmp)], capture_output=True, text=True,
                                   env=dict(ENV, AB_ITEM=key), timeout=300)
                self.assertEqual(p.returncode, 0, p.stdout[-500:] + p.stderr[-1500:])
                self.assertIn(f"manifest {arm} {key} ok", p.stdout)


if __name__ == "__main__":
    unittest.main()
