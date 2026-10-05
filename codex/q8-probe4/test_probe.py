#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import probe
import prepare_driver


class ProbeTest(unittest.TestCase):
    def recorded_with_metrics(self):
        # The old driver reports only failed ERRs; fill missing values only for gate tests.
        raw = (probe.HERE / "fixtures/canary-test.sql").read_text()
        return "\n".join("[FLASH_ATTN_EXT] ERR = 0.0001 > " + str(5e-4 * max(1, probe.shape_key(probe.sql_rows(line)[0]["params"])[1] / 20000)) + " " + line
                         if "INSERT INTO" in line and "ERR =" not in line else line for line in raw.splitlines())

    def test_f16_failure_is_reported_and_q8_failure_gates(self):
        raw = self.recorded_with_metrics()
        rows = probe.validate(raw, "test", "797465a5c", expected=probe.RECORDED_CASES)
        self.assertEqual(len(rows), 48)
        failed = [r for r in rows if r["passed"] == "0"]
        self.assertEqual([(r["type"], r["nmse"], r["tolerance"]) for r in failed], [("f16", 0.002350528, 0.0022528)])
        lines = raw.splitlines()
        q8 = next(i for i, line in enumerate(lines) if "name=q8-fa:q8_0:" in line)
        lines[q8] = lines[q8].replace("ERR = 0.0001", "ERR = 0.1").replace("'1', '1', ''", "'1', '0', 'test failed'")
        with self.assertRaises(probe.Gate) as error:
            probe.validate("\n".join(lines), "test", "797465a5c", expected=probe.RECORDED_CASES)
        self.assertEqual(error.exception.code, 8)

    def test_metrics_fail_closed(self):
        raw = self.recorded_with_metrics()
        for mutant in (raw.replace("ERR = 0.0001", "ERR = nan"), raw.replace(" > 0.0005", " > 0.1"),
                       raw.replace("'1', '1', ''", "'1', '0', 'compare failed'")):
            with self.assertRaises(probe.Gate) as error:
                probe.validate(mutant, "test", "797465a5c", expected=probe.RECORDED_CASES)
            self.assertEqual(error.exception.code, 6)

    def test_f16_integrity_requires_same_gpu_and_both_arms(self):
        rows = probe.validate(self.recorded_with_metrics(), "test", "797465a5c", expected=probe.RECORDED_CASES)
        controls = [dict(r, arm=a, gpu=0, kind="test") for a in probe.FROZEN for r in rows]
        probe.f16_integrity(controls)
        for missing in ([r for r in controls if r["arm"] == "up"], []):
            with self.assertRaises(probe.Gate):
                probe.f16_integrity(missing)
        next(r for r in controls if r["type"] == "f16" and r["arm"] == "patch")["nmse"] += 2e-6
        with self.assertRaises(probe.Gate) as error:
            probe.f16_integrity(controls)
        self.assertEqual(error.exception.code, 6)

    def test_driver_instrumentation(self):
        source = (probe.ROOT / "tests/test-backend-ops.cpp").read_text()
        driver = prepare_driver.instrument(source)
        self.assertIn("gen(probe_seed(tensor) ^ (uint32_t) start)", driver)
        self.assertIn("std::mt19937 rd(probe_seed(tensor));", driver)
        self.assertIn("ERR = %.17g > %.17g", driver)
        with self.assertRaises(AssertionError):
            prepare_driver.instrument(source.replace("ggml_backend_load_all();", ""))

    def test_perf_cap_exceeds_driver_minimum(self):
        source = (probe.ROOT / "tests/test-backend-ops.cpp").read_text()
        self.assertIn("total_time_us < 250*1000", prepare_driver.instrument(source))
        cases = len(probe.matrix())
        self.assertGreater(probe.STAGE_S["perf"], cases * 0.25)

    def test_test_exit_one_requires_row_validation(self):
        with tempfile.TemporaryDirectory(dir=probe.ROOT / "build-tmp") as tmp:
            args = [sys.executable, "-c", "print('saved'); raise SystemExit(1)"]
            raw, _ = probe.command(args, Path(tmp), "test", time.monotonic() + 2, allow_test_failure=True)
            self.assertEqual(raw.strip(), "saved")
            with self.assertRaises(probe.Gate):
                probe.validate(raw, "test", "797465a5c", expected=probe.RECORDED_CASES)
            with self.assertRaises(probe.Gate):
                probe.command(args, Path(tmp), "perf", time.monotonic() + 2)

    def test_real_artifacts(self):
        probe.selftest()

    def test_case_serialization(self):
        lines = (probe.ROOT / "codex/q8-general-case-schema.txt").read_text().splitlines()
        self.assertEqual({probe.shape_key(line) for line in lines}, probe.matrix())
        controls = (probe.ROOT / "codex/q8-general-controls-schema.txt").read_text().splitlines()
        self.assertEqual({probe.shape_key(line) for line in controls}, {(t, 8192, nb, g, d) for t in probe.TYPES for nb, g, d in probe.CONTROLS})

    def test_devices_fail_closed(self):
        real = (probe.HERE / "fixtures/real-fa-perf.log").read_text()
        for mutant in (real.replace("RTX 5060 Ti", "other GPU"), real + "\nDevice 1: RTX 5060 Ti, compute capability 12.0,", ""):
            with self.assertRaises(probe.Gate):
                probe.check_device(mutant)

    def test_timeout_stops_exact_pid_and_saves_raw(self):
        with tempfile.TemporaryDirectory(dir=probe.ROOT / "build-tmp") as tmp:
            out = Path(tmp)
            foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
            try:
                with self.assertRaises(probe.Gate) as error:
                    probe.command([sys.executable, "-c", "import time; print('raw', flush=True); time.sleep(5)"], out, "deadline", time.monotonic() + 0.15)
                self.assertEqual(error.exception.code, 7)
                self.assertEqual((out / "deadline.sql").read_text().strip(), "raw")
                pid = int((out / "deadline.pid").read_text())
                self.assertFalse(Path(f"/proc/{pid}").exists())
                self.assertIsNone(foreign.poll())
            finally:
                foreign.terminate()
                foreign.wait()

    def test_no_gpu_mode_refuses_before_gpu_command(self):
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            with tempfile.TemporaryDirectory(dir=probe.ROOT / "build-tmp") as tmp:
                with self.assertRaises(probe.Gate) as error:
                    probe.run("canary", Path(tmp), 1)
                self.assertEqual(error.exception.code, 4)
        finally:
            if original is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original

    def test_build_commit_must_differ_from_source_only_under_codex(self):
        with tempfile.TemporaryDirectory(dir=probe.ROOT / "build-tmp") as tmp:
            def git(*args):
                return subprocess.run(["git", "-C", tmp, "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                      check=True, capture_output=True, text=True).stdout.strip()

            def commit(path, text):
                (Path(tmp) / path).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / path).write_text(text)
                git("add", "-A")
                git("commit", "-qm", path)
                return git("rev-parse", "HEAD")

            git("init", "-q")
            source = commit("src/a.c", "1")
            ok = commit("codex/r.txt", "2")
            bad = commit("src/a.c", "3")
            probe.build_identity({"up": source}, Path(tmp), {"up": ok[:9]})
            for build in (source[:9], bad[:9]):
                with self.assertRaises(probe.Gate) as error:
                    probe.build_identity({"up": source}, Path(tmp), {"up": build})
                self.assertEqual(error.exception.code, 6)
            git("checkout", "-q", "--orphan", "other")
            other = commit("codex/o.txt", "4")
            with self.assertRaises(probe.Gate):
                probe.build_identity({"up": source}, Path(tmp), {"up": other[:9]})


if __name__ == "__main__":
    unittest.main()
