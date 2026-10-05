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
        with tempfile.TemporaryDirectory() as tmp:
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
