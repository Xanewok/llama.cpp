"""The tunnel supervisor, driven with a stub ssh on PATH.

The forward drops roughly every 80-90 minutes under load and ended two legs that way.
The supervisor's contract is narrow and both halves matter: restart a forward that served
and then dropped, and give up on one that never binds -- a spin on a dead box would hold
the port and burn CPU for as long as the session lives.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOX = ROOT / "scripts" / "infra" / "box.sh"


class SupervisorTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.bin = Path(tmp.name)
        self.marks = self.bin / "attempts"

    def stub_ssh(self, body: str) -> None:
        path = self.bin / "ssh"
        path.write_text(f'#!/usr/bin/env bash\necho x >> "{self.marks}"\n{body}\n')
        path.chmod(0o755)

    def supervise(self, timeout: float, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(BOX), "tunnel-supervise", "h", "22", "18999", "8000"],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", **env},
        )

    def attempts(self) -> int:
        return len(self.marks.read_text().splitlines()) if self.marks.exists() else 0

    def test_gives_up_when_the_forward_never_binds(self) -> None:
        # ExitOnForwardFailure makes ssh exit at once on a taken port or a gone box.
        self.stub_ssh("exit 255")
        result = self.supervise(timeout=30, TUNNEL_MAX_FAILS="3", TUNNEL_UP_SECONDS="10")
        self.assertEqual(result.returncode, 1)
        self.assertIn("giving up", result.stderr)
        self.assertEqual(self.attempts(), 3)

    def test_a_forward_that_stayed_up_is_restarted_not_counted(self) -> None:
        # The transient this exists for: ssh served, then dropped. Three such drops must
        # not trip a give-up threshold of three, or an hourly drop would end the run.
        self.stub_ssh('sleep 1; exit 255')
        with self.assertRaises(subprocess.TimeoutExpired):
            self.supervise(timeout=12, TUNNEL_MAX_FAILS="3", TUNNEL_UP_SECONDS="0")
        self.assertGreater(self.attempts(), 3)

    def test_it_reconnects_rather_than_exiting_on_the_first_drop(self) -> None:
        self.stub_ssh("exit 255")
        with self.assertRaises(subprocess.TimeoutExpired):
            self.supervise(timeout=8, TUNNEL_MAX_FAILS="99", TUNNEL_UP_SECONDS="10")
        self.assertGreater(self.attempts(), 1)


class TunnelWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.box = BOX.read_text()

    def test_tunnel_launches_the_supervisor_not_a_bare_ssh(self) -> None:
        launch = self.box[self.box.index("tunnel)"):self.box.index("tunnel-supervise)")]
        self.assertIn('"$0" tunnel-supervise', launch)
        self.assertNotIn("nohup ssh", launch)

    def test_supervisors_are_killed_before_the_ssh_they_own(self) -> None:
        # The reverse order lets a live supervisor respawn the forward between the two
        # kills, leaving the port held by a tunnel pointing at the previous box.
        launch = self.box[self.box.index("tunnel)"):self.box.index("tunnel-supervise)")]
        self.assertLess(
            launch.index('pgrep -f -- "tunnel-supervise'),
            launch.index('pgrep -f -- "$spec"'),
        )


if __name__ == "__main__":
    unittest.main()
