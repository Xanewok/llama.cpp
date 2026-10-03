#!/usr/bin/env python3
"""CPU-only tests for the sanitizer stages of scope_gpu.py. Run: python3 codex/test_sanitize.py

Real fixtures: codex/fixtures/memcheck-real-*.log (compute-sanitizer memcheck, sm_120 and sm_75).
SYNTHETIC blocks below follow the documented formats of the CUDA 13.2 Compute Sanitizer manual
(/usr/local/cuda-13.2/compute-sanitizer/docs); no racecheck/synccheck/memcheck-error output exists locally.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scope_gpu as g

FIX = Path(__file__).resolve().parent / "fixtures"
REAL = (FIX / "memcheck-real-sm120-20261003.log").read_text()
REAL75 = (FIX / "memcheck-real-sm75-20261001.log").read_text()

# SYNTHETIC: manual's racecheck analysis report, plus the sample test-backend-ops tail.
RACE_ERR = """========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at flash_attn_ext_f16<(int)256, (int)256, (int)8, (int)8>(...)+0x90 in fattn-mma-f16.cuh:1500
=========     and Read access at flash_attn_ext_f16<(int)256, (int)256, (int)8, (int)8>(...)+0x100 in fattn-mma-f16.cuh:1510 [508 hazards]
=========
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
"""
RACE_WARN = "========= WARNING: Race reported between Write access at k()+0xf0 in a.cu:40\n=========     and Read access at k()+0x280 in a.cu:46 [4 hazards]\n"
TAIL = "  73/73 tests passed\n  Backend CUDA0: OK\n"
# SYNTHETIC: manual's synccheck sample.
SYNC_ERR = ("========= Barrier error detected. Divergent thread(s) in warp\n=========     at flash_attn_ext_f16<(int)256>(...)+0xf0 in x.cuh:79\n"
            "=========     by thread (37,0,0) in block (0,0,0)\n=========\n========= ERROR SUMMARY: 1 error\n")
# SYNTHETIC: manual's memcheck sample, kernel name as the repro expects it.
MEM_ERR = ("========= COMPUTE-SANITIZER\n========= Invalid __global__ read of size 4 bytes\n"
           "=========     at 0x1a0 in /x/fattn-common.cuh:696:void flash_attn_mask_to_KV_max<(int)8>(const __half2 *, int *, int, long, long)\n"
           "=========     by thread (3,0,0) in block (0,1,0)\n=========     Address 0x7f1200100000 is out of bounds\n"
           "=========\n  1/1 tests passed\n========= ERROR SUMMARY: 1 error\n")


def case_vars(hsk, nh, nr, kv, nb, sinks, kvtype="f16"):
    return (f"hsk={hsk},hsv={hsk},nh={nh},nr23=[{nr[0]},{nr[1]}],kv={kv},nb={nb},mask=1,sinks={sinks},max_bias=0.000000,"
            f"logit_softcap=0.000000,prec=f32,type_K={kvtype},type_V={kvtype},permute=[0,1,2,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0")


class Parser(unittest.TestCase):
    def test_real_logs(self):
        for text, total in ((REAL, 73), (REAL75, 73)):
            p = g.parse_sanitizer(text)
            self.assertEqual((p["errors"], p["tests_passed"], p["tests_total"], p["blocks"]), (0, total, total, []))

    def test_synthetic_racecheck(self):
        p = g.parse_sanitizer(RACE_ERR + TAIL)
        self.assertEqual(p["hazards"], (1, 1, 0))
        self.assertEqual(g.gate_sanitizer("racecheck", p, 99, 73)[0], "hazard")
        self.assertIn("flash_attn_ext_f16", p["kernels"])
        self.assertEqual(len(g.parse_sanitizer(RACE_WARN)["blocks"]), 1)

    def test_synthetic_synccheck_and_memcheck(self):
        p = g.parse_sanitizer(SYNC_ERR + TAIL)
        self.assertEqual(g.gate_sanitizer("synccheck", p, 99, 73)[0], "hazard")
        self.assertEqual(p["blocks"][0]["header"], "Barrier error detected. Divergent thread(s) in warp")
        self.assertIn("flash_attn_mask_to_KV_max", g.parse_sanitizer(MEM_ERR)["kernels"])

    def test_ansi_and_multi_process(self):
        text = "\x1b[1;32m========= ERROR SUMMARY: 0 errors\x1b[0m\n========= ERROR SUMMARY: 2 errors\n"
        self.assertEqual(g.parse_sanitizer(text)["errors"], 2)

    def test_incomplete_and_wrong_counts(self):
        p = g.parse_sanitizer(REAL)
        self.assertEqual(g.gate_sanitizer("racecheck", p, 0, 73)[0], "incomplete")
        self.assertEqual(g.gate_sanitizer("memcheck", p, None, 73)[0], "incomplete")
        self.assertEqual(g.gate_sanitizer("memcheck", p, 0, 74)[0], "incomplete")
        self.assertEqual(g.gate_sanitizer("memcheck", p, 3, 73)[0], "fail")
        self.assertEqual(g.gate_sanitizer("memcheck", g.parse_sanitizer(REAL.replace("OK", "FAIL")), 0, 73)[0], "fail")
        self.assertEqual(g.gate_sanitizer("memcheck", g.parse_sanitizer(REAL[:5000]), 0, 73)[0], "incomplete")


class Repro(unittest.TestCase):
    CASES = {"mask_partial_tile": case_vars(256, 2, (1, 2), 32768, 3, 0), "mask_full_tile": case_vars(256, 2, (1, 2), 32768, 8, 0),
             "mask_one_sequence": case_vars(256, 2, (1, 1), 32768, 3, 0), "sinks_gqa6_padded": case_vars(256, 16, (6, 1), 512, 32, 1),
             "sinks_gqa8_no_padding": case_vars(256, 16, (8, 1), 512, 32, 1)}

    def test_each_selector_matches_only_its_case(self):
        self.assertEqual(set(self.CASES), set(g.REPRO_CASES))
        for name, spec in g.REPRO_CASES.items():
            hits = [n for n, v in self.CASES.items() if re.search(spec["case"], v)]
            self.assertEqual(hits, [name])

    def test_selectors_miss_the_shared_selection(self):
        for v in self.CASES.values():
            self.assertFalse(re.search(g.NEW_CASES, v))
        real = [re.search(r"FLASH_ATTN_EXT\((.*)\): ", l).group(1) for l in REAL.splitlines() if "FLASH_ATTN_EXT(" in l]
        for spec in g.REPRO_CASES.values():
            self.assertFalse(any(re.search(spec["case"], c) for c in real))

    def test_canary_selects_one_real_case(self):
        real = [re.search(r"FLASH_ATTN_EXT\((.*)\): ", l).group(1) for l in REAL.splitlines() if "FLASH_ATTN_EXT(" in l]
        self.assertEqual(sum(bool(re.search(g.CANARY_CASE, c)) for c in real), g.CANARY_CASE_COUNT)
        self.assertEqual(sum(bool(re.search(g.NEW_CASES, c)) for c in real), g.NEW_CASES_COUNT)

    def test_verdicts(self):
        defect, control = g.REPRO_CASES["mask_partial_tile"], g.REPRO_CASES["mask_full_tile"]
        hit = g.parse_sanitizer(MEM_ERR)
        self.assertEqual(g.repro_verdict(defect, hit, 99)[0], "reproduced")
        self.assertEqual(g.repro_verdict(control, hit, 99)[0], "fail")
        clean = g.parse_sanitizer("  1/1 tests passed\n========= ERROR SUMMARY: 0 errors\n")
        self.assertEqual(g.repro_verdict(defect, clean, 0)[0], "not_reproduced")
        self.assertEqual(g.repro_verdict(control, clean, 0)[0], "pass")
        self.assertEqual(g.repro_verdict(defect, hit, None)[0], "incomplete")
        other = g.parse_sanitizer(MEM_ERR.replace("flash_attn_mask_to_KV_max", "something_else"))
        self.assertEqual(g.repro_verdict(defect, other, 99)[0], "fail")

    def test_injection_is_strict(self):
        self.assertRaises(AssertionError, g.repro_tests, "int main() {}")
        src = g.SINKS_OLD + g.REPRO_ANCHOR
        out = g.repro_tests(src)
        self.assertEqual(out.count("sanitizer repro"), 2)
        self.assertRaises(AssertionError, g.repro_tests, out)


class Plan(unittest.TestCase):
    def test_budget(self):
        for stage in ("san", "repro", "synccheck", "racecheck", "build"):
            self.assertLessEqual(sum(e for _, e in g.sanitize_plan(stage, False, True)), 0.7 * g.SAN_BUDGET_S)
        self.assertLessEqual(sum(e for n, e in g.sanitize_plan("san", True, False)), 300)


FAKE = """#!/usr/bin/env python3
import sys, time
argv = sys.argv
tool, pat = argv[argv.index("--tool") + 1], argv[argv.index("-p") + 1]
if "SLOW" in pat:
    time.sleep(30)
real = open(%r).read()
if "kv=32768,nb=3" in pat and "nr23=\\[1,2\\]" in pat:
    sys.stdout.write(%r)
elif "kv=32768" in pat or "kv=512,nb=32" in pat:
    sys.stdout.write("  1/1 tests passed\\n========= ERROR SUMMARY: 0 errors\\n")
else:
    if "nb=8,mask=1,sinks=0" in pat and "q8_0" in pat:
        real = real.replace("73/73", "1/1")
    if tool == "racecheck":
        real = real.replace("ERROR SUMMARY: 0 errors", "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)")
    sys.stdout.write(real)
sys.exit(0)
""" % (str(FIX / "memcheck-real-sm120-20261003.log"), MEM_ERR)


class Flow(unittest.TestCase):
    """Drives Run.san_run / Run.sanitize with a fake compute-sanitizer; no GPU tool is executed."""

    def setUp(self):
        import hashlib, json, shutil, tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        self.tmp = Path(tempfile.mkdtemp(prefix="san-test-", dir=g.ROOT / "codex"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        tool = self.tmp / "toolkit/bin"
        tool.mkdir(parents=True)
        (tool / "compute-sanitizer").write_text(FAKE)
        (tool / "compute-sanitizer").chmod(0o755)
        self.frozen = self.tmp / "frozen"
        for arm in ("up", "scope", "repro"):
            d = self.frozen / arm
            d.mkdir(parents=True)
            (d / "test-backend-ops").write_text("fake")
            (d / "SHA256SUMS.json").write_text(json.dumps({"test-backend-ops": hashlib.sha256(b"fake").hexdigest()}))
        self.args = SimpleNamespace(budget=g.SAN_BUDGET_S, out=str(self.tmp / "out"), gpus="0", arch=120, toolkit=str(self.tmp / "toolkit"),
                                    frozen_dir=str(self.frozen), stage="san", canary=False, local=True, port=0, model=None,
                                    e2e_only=False, seed_frozen="")
        self.run_ = g.Run(self.args)
        for name, fn in (("guard", lambda: None), ("xid", lambda: None), ("ensure_frozen", lambda: None),
                         ("frozen_ok", lambda *a, **k: True), ("source_inputs", lambda: (None, ""))):
            setattr(self.run_, name, fn)
        patcher = patch.object(g, "repro_tests", lambda text: text)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_full_flow_reports_defects_and_stays_clean(self):
        self.run_.sanitize()
        v = self.run_.summary["sanitize"]
        self.assertEqual(v["repro-mask_partial_tile"], "reproduced")
        self.assertEqual(v["repro-sinks_gqa6_padded"], "not_reproduced")
        self.assertEqual(v["repro-mask_full_tile"], "pass")
        for tool in ("racecheck", "synccheck"):
            for arm in ("up", "scope"):
                self.assertEqual(v[f"{tool}-{arm}"], "pass")
        for tag in v:
            self.assertTrue((self.run_.out / f"{tag}.log").stat().st_size > 0)

    def test_canary_runs_one_case_per_tool(self):
        self.run_.args.canary = True
        self.run_.sanitize()
        self.assertEqual(sorted(self.run_.summary["sanitize"]),
                         ["racecheck-scope-canary", "repro-mask_partial_tile", "synccheck-scope-canary"])

    def test_timeout_is_incomplete_and_log_survives(self):
        rec = self.run_.san_run("racecheck", "scope", "slow", "SLOW", 1, cases=1)
        self.assertEqual((rec["verdict"], rec["rc"]), ("incomplete", None))
        self.assertTrue((self.run_.out / "slow.log").exists())


if __name__ == "__main__":
    unittest.main()
