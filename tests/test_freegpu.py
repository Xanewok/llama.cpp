"""freegpu.sh is the gate every measurement sits behind, and it reads nvidia-smi, which
prints one line per card. These pin the multi-GPU arithmetic and the fail-closed contract."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEGPU = ROOT / "serving" / "freegpu.sh"


class FreeGpuTest(unittest.TestCase):
    def run_with(self, smi_body: str) -> subprocess.CompletedProcess:
        """Run freegpu.sh against a stub nvidia-smi. sleep is stubbed too, or the
        held-memory path burns its full 60 s retry budget in the suite."""
        with tempfile.TemporaryDirectory() as td:
            bindir = Path(td) / "bin"
            bindir.mkdir()
            (bindir / "nvidia-smi").write_text("#!/usr/bin/env bash\n" + smi_body)
            (bindir / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
            # freegpu kills discovered engine PIDs; unit tests must never discover host processes.
            (bindir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
            for f in bindir.iterdir():
                f.chmod(0o755)
            env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", WORK=td)
            return subprocess.run([str(FREEGPU)], capture_output=True, text=True,
                                  env=env, timeout=120)

    def test_two_idle_cards_sum_below_the_threshold(self):
        p = self.run_with('printf "5\\n7\\n"\n')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("12 MiB", p.stdout)

    def test_one_busy_card_holds_the_gate_shut(self):
        """The single-GPU form errored on a two-line value and failed closed by accident;
        this must fail closed because the memory is actually held."""
        p = self.run_with('printf "5\\n8000\\n"\n')
        self.assertEqual(p.returncode, 1)
        self.assertIn("GPU STILL HELD", p.stderr)

    def test_a_broken_nvidia_smi_is_not_read_as_free(self):
        p = self.run_with("exit 1\n")
        self.assertEqual(p.returncode, 1)
        self.assertNotIn("gpu free", p.stdout)

    def test_single_card_still_passes(self):
        p = self.run_with('printf "3\\n"\n')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("3 MiB", p.stdout)
