"""`box.sh rent` must not leave a billing box behind when it is killed.

Between `vastai create instance` and the id being printed, `rent` polls for up to
RENT_CAP*20s. In that window the caller has no id, so its teardown trap has nothing to
destroy -- a kill there used to leave a box billing until the reaper's TTL caught it.
This drives the real script against a stub `vastai`, so the assertion is about the
shell's control flow rather than a mock's expectations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOX = ROOT / "scripts" / "infra" / "box.sh"
INSTANCE_ID = "4242424"

STUB = r"""#!/usr/bin/env bash
# Stands in for the vast.ai CLI. Records every destroy so the test can assert on it.
case "$1 $2" in
  "search offers")
    cat <<'JSON'
[{"id": 999, "machine_id": 111, "dph_total": 0.10, "gpu_max_power": 400,
  "reliability2": 0.999, "inet_down": 1000, "cpu_cores_effective": 32,
  "cuda_max_good": 13.5, "verification": "verified"}]
JSON
    ;;
  "create instance") echo '{"new_contract": __ID__}' ;;
  "show instance")   echo '{"actual_status": "running", "status_msg": "",
                          "ssh_host": "stub.invalid", "ssh_port": 2222}' ;;
  "destroy instance") echo "$3" >> "$DESTROY_LOG"; echo "destroyed $3" ;;
  "logs "*|"logs") : ;;
  *) : ;;
esac
""".replace("__ID__", INSTANCE_ID)


class RentOrphanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        bindir = d / "bin"
        bindir.mkdir()
        stub = bindir / "vastai"
        stub.write_text(STUB)
        stub.chmod(0o755)
        # `endpoint` shells out to ssh; a stub that always fails keeps rent in its poll
        # loop, which is exactly the window under test.
        ssh = bindir / "ssh"
        ssh.write_text("#!/usr/bin/env bash\nexit 1\n")
        ssh.chmod(0o755)
        # rent's poll loop sleeps 20s a tick. Shorten it so the window under test is
        # seconds wide rather than minutes; the control flow is identical.
        slp = bindir / "sleep"
        slp.write_text("#!/usr/bin/env bash\nexec /bin/sleep 0.2\n")
        slp.chmod(0o755)
        self.destroy_log = d / "destroys.txt"
        self.env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DESTROY_LOG": str(self.destroy_log),
            "STATE": str(d / "instances.log"),
            "QUARANTINE": str(d / "quarantine.log"),
            # Generous, so the poll loop is still running when the test kills it.
            "RENT_CAP": "600",
            "RENT_WAIT": "600",
            "RENT_TRIES": "1",
        }
        self.addCleanup(self.tmp.cleanup)

    def _rent(self):
        return subprocess.Popen(
            [str(BOX), "rent", "test-ttl4", "RTX 3090", "1", "80"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=self.env, start_new_session=True,
        )

    def _wait_until_created(self, proc, timeout=30):
        """The box exists once rent has recorded it; that is the window's start."""
        state = Path(self.env["STATE"])
        deadline = time.time() + timeout
        while time.time() < deadline:
            if state.exists() and INSTANCE_ID in state.read_text():
                return True
            if proc.poll() is not None:
                self.fail(f"rent exited early: {proc.communicate()}")
            time.sleep(0.2)
        self.fail("rent never created an instance")

    def _destroyed(self):
        return self.destroy_log.read_text().split() if self.destroy_log.exists() else []

    def test_sigterm_during_the_poll_destroys_the_box(self):
        proc = self._rent()
        self._wait_until_created(proc)
        proc.terminate()
        proc.communicate(timeout=60)
        self.assertIn(INSTANCE_ID, self._destroyed(),
                      "rent was killed after creating a box and did not destroy it")

    def test_sigint_during_the_poll_destroys_the_box(self):
        proc = self._rent()
        self._wait_until_created(proc)
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.communicate(timeout=60)
        self.assertIn(INSTANCE_ID, self._destroyed())

    def test_a_box_handed_to_the_caller_is_not_destroyed(self):
        """The trap must disown on success, or rent would destroy what it just returned."""
        ssh = Path(self.env["PATH"].split(":")[0]) / "ssh"
        ssh.write_text("#!/usr/bin/env bash\nexit 0\n")   # endpoint now succeeds
        proc = self._rent()
        out, err = proc.communicate(timeout=120)
        self.assertEqual(proc.returncode, 0, err)
        self.assertEqual(out.strip(), INSTANCE_ID)
        self.assertEqual(self._destroyed(), [],
                         "rent destroyed the box it had just handed to the caller")
