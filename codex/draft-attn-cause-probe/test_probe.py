#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import probe


class ProbeTest(unittest.TestCase):
    def test_recorded_numeric_errors(self):
        rows = probe.metrics((probe.HERE / "fixtures/recorded-q8.sql").read_text())
        failures = [r for r in rows if r["passed"] == "0"]
        self.assertEqual(len(failures), 2)
        self.assertEqual([r["err"] for r in failures], [0.00086876831123901721, 0.0024069909944978208])
        self.assertEqual(probe.sql_rows((probe.HERE / "fixtures/recorded-draft.sql").read_text())[4]["passed"], "0")

    def diagnostic_fixture(self):
        raw = (probe.HERE / "fixtures/recorded-q8.sql").read_text()
        lines = []
        for line in raw.splitlines():
            rows = probe.sql_rows(line)
            if not rows or rows[0]["op"] != "FLASH_ATTN_EXT":
                continue
            match = probe.re.search(r"name=q8-fa:(f16|q8_0):(8192|32768|90112):q1:g(4|6):d256,", rows[0]["params"])
            if not match:
                continue
            typ, kv, gqa = match.groups()
            kv, gqa = int(kv), int(gqa)
            if gqa == 4 and (kv != 90112 or typ != "f16"):
                continue
            line = line.replace(match[0], f"name={probe.case_name(typ, kv, 1, gqa)},")
            if typ == "f16":
                line = "ORACLE = 0.000001 CPU_ERR = 0.0024 INPUT = 123456789abcdef0 " + line
            lines.append(line)
        return "\n".join(lines)

    def test_diagnostic_gates_keep_reference_failures(self):
        raw = self.diagnostic_fixture()
        rows = probe.diagnostic_rows(raw, "0bb874d20")
        self.assertEqual(len(rows), 7)
        self.assertEqual(sum(r["reference_tolerance_failed"] for r in rows), 1)
        for mutant in (raw.replace("ORACLE = 0.000001", "ORACLE = 0.1"),
                       raw.replace("CPU_ERR = 0.0024", "CPU_ERR = nan"),
                       raw.replace(" > 0.0022528000000000001", " > 0.1"),
                       raw.replace("'0bb874d20'", "'000000000'"),
                       raw + "\n" + raw.splitlines()[0], ""):
            with self.assertRaises(probe.Gate):
                probe.diagnostic_rows(mutant, "0bb874d20")
        paired = [dict(r, gpu=0, seed=0, arm="up") for r in rows]
        paired += [dict(r, gpu=0, seed=0, arm="patch") for r in rows]
        probe.paired(paired)
        paired[-1]["input_hash"] = "wrong"
        # The final q8 record has no f16 hash gate.
        paired[-2]["input_hash"] = "wrong"
        with self.assertRaises(probe.Gate):
            probe.paired(paired)

    def test_real_artifacts(self):
        probe.selftest()

    def test_case_serialization(self):
        lines = (probe.ROOT / "codex/fa-case-schema-results.txt").read_text().splitlines()
        self.assertEqual({probe.shape_key(line) for line in lines}, {(t, k, 1, 6) for t in probe.TYPES for k in probe.DEPTHS})
        controls = (probe.ROOT / "codex/fa-controls-schema-results.txt").read_text().splitlines()
        self.assertEqual({probe.shape_key(line) for line in controls}, {("f16", 8192, nb, gqa) for nb, gqa in probe.CONTROLS})

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
