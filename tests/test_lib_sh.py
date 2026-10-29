"""The shell layer's wait primitives, driven through bash.

Only the two functions with real logic are tested. `say` is a wrapper and a test of it
would assert that echo echoes; the value here is that `proc_state` distinguishes three
outcomes, because collapsing "ssh failed" into "process is gone" is what recorded a
context rung as refused and put the wrong ceiling in a results table.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "infra" / "lib.sh"


def run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail\n. "{LIB}"\n{script}'],
        capture_output=True, text=True, timeout=60,
    )


class ProcStateTest(unittest.TestCase):
    """A fake runner stands in for ssh, so the three outcomes are reachable offline."""

    def state(self, runner_body: str) -> int:
        return run(
            f"runner () {{ {runner_body}; }}\nproc_state runner someproc\necho \"rc=$?\""
        ).stdout.strip().splitlines()[-1].removeprefix("rc=")

    def test_running_is_zero(self) -> None:
        self.assertEqual(self.state('echo UP'), "0")

    def test_gone_is_one(self) -> None:
        self.assertEqual(self.state('echo DOWN'), "1")

    def test_ssh_failure_is_unknown_not_gone(self) -> None:
        # ssh exits 255 for any client-side failure. Reading that as "the process died"
        # is the bug: a transient drop then ends a wait loop early.
        self.assertEqual(self.state('return 255'), "2")

    def test_unrecognised_output_is_unknown(self) -> None:
        self.assertEqual(self.state('echo something-else'), "2")

    def test_empty_output_is_unknown(self) -> None:
        self.assertEqual(self.state('true'), "2")


class AwaitServerTest(unittest.TestCase):
    def await_with(self, curl_body: str, runner_body: str, expect: str = "") -> str:
        script = f"""
        curl () {{ {curl_body}; }}
        runner () {{ {runner_body}; }}
        await_server 65535 2 0 runner someproc {expect}
        echo "rc=$?"
        """
        return run(script).stdout.strip().splitlines()[-1].removeprefix("rc=")

    def test_healthy_without_identity_check(self) -> None:
        self.assertEqual(self.await_with('echo ok', 'echo UP'), "0")

    def test_confirmed_dead_returns_one(self) -> None:
        self.assertEqual(self.await_with('true', 'echo DOWN'), "1")

    def test_ssh_flake_keeps_waiting_then_times_out(self) -> None:
        # Must exhaust its tries rather than return 1 on the first failed probe.
        self.assertEqual(self.await_with('true', 'return 255'), "1")

    def test_stale_server_serving_another_model_is_rejected(self) -> None:
        # A server that outlived its kill answers /health from its previous config.
        out = run(f"""
        curl () {{ echo ok; }}
        served_model () {{ echo "/models/OLD-QUANT.gguf"; }}
        runner () {{ echo UP; }}
        await_server 65535 1 0 runner someproc NEW-QUANT
        echo "rc=$?"
        """).stdout.strip().splitlines()[-1]
        self.assertEqual(out, "rc=3")

    def test_matching_model_is_accepted(self) -> None:
        out = run(f"""
        curl () {{ echo ok; }}
        served_model () {{ echo "/models/NEW-QUANT.gguf"; }}
        runner () {{ echo UP; }}
        await_server 65535 1 0 runner someproc NEW-QUANT
        echo "rc=$?"
        """).stdout.strip().splitlines()[-1]
        self.assertEqual(out, "rc=0")


class TunnelPatternTest(unittest.TestCase):
    """box.sh's tunnel gate must match a real ssh command line.

    The first version matched "ssh -N -L <port>:", which cannot occur: SSH_OPTS are
    interpolated between `ssh` and `-N`. As a stale-tunnel killer that failed silently;
    promoted to a success gate it failed every launch.
    """

    ARGV = ("ssh -i /home/u/.ssh/id_ed25519_vast -o IdentitiesOnly=yes "
            "-o StrictHostKeyChecking=no -N -L 127.0.0.1:8010:127.0.0.1:8000 "
            "-o ExitOnForwardFailure=yes -p 55192 root@ssh9.vast.ai")

    def test_the_old_pattern_could_never_match(self) -> None:
        self.assertNotIn("ssh -N -L 8010:", self.ARGV)

    def test_the_forward_spec_matches(self) -> None:
        self.assertIn("-L 127.0.0.1:8010:127.0.0.1:8000", self.ARGV)

    def test_box_sh_uses_the_forward_spec(self) -> None:
        box = (ROOT / "scripts" / "infra" / "box.sh").read_text()
        self.assertIn('-L 127.0.0.1:$lport:127.0.0.1:$rport', box)
        self.assertNotIn('pgrep -f "ssh -N -L', box)

    def test_box_sh_gates_on_port_ownership(self) -> None:
        # A bare connect test cannot tell our forward from a squatter, and checking our
        # ssh is alive races its own startup. Both were tried against a real squatted
        # port and both passed when they should have failed.
        box = (ROOT / "scripts" / "infra" / "box.sh").read_text()
        self.assertIn("ss -ltnpH", box)
        self.assertIn("/proc/$pid/cmdline", box)
        self.assertNotIn("/dev/tcp/127.0.0.1/$lport", box)

    def test_forward_binds_one_explicit_address(self) -> None:
        # "-L $lport:" binds 127.0.0.1 and ::1 both, so a squatter holding only IPv4 lets
        # ssh bind IPv6 and look healthy while clients still reach the squatter.
        box = (ROOT / "scripts" / "infra" / "box.sh").read_text()
        self.assertIn('spec="-L 127.0.0.1:$lport:127.0.0.1:$rport"', box)


class RentPriceCapTest(unittest.TestCase):
    """The retry loop walks *up* the price-sorted offer list.

    Two machines that took a rental and then refused the ssh key pushed one arm from the
    $0.115 market floor to $0.412/hr, and nothing in the loop noticed it had left the band
    the arm was budgeted for.
    """

    BOX = (ROOT / "scripts" / "infra" / "box.sh").read_text()

    def test_the_cap_is_applied_before_the_first_offer_is_taken(self) -> None:
        self.assertIn("dphmax = float(os.environ['DPHMAX'])", self.BOX)
        self.assertIn("if (o.get('dph_total') or 1e9) <= dphmax", self.BOX)

    def test_the_cap_has_a_default_so_it_binds_unset(self) -> None:
        # An env-only cap protects nothing: every existing caller predates it.
        self.assertIn("DPHMAX=${DPHMAX:-0.22}", self.BOX)

    def test_no_offer_under_the_cap_is_a_refusal_not_an_escalation(self) -> None:
        self.assertIn("no offer for $gpu x$n under", self.BOX)


class RentPatienceTest(unittest.TestCase):
    """RENT_CAP must outlast a cold host's image pull.

    At 60 ticks (20 min) a 3090 was destroyed mid-`Pulling fs layer`, after paying for the
    rental and most of a ~5 GB cudnn-devel download, and the retry drew another cold machine.
    """

    BOX = (ROOT / "scripts" / "infra" / "box.sh").read_text()

    def test_the_wall_clock_budget_covers_a_cold_pull(self) -> None:
        self.assertIn("${RENT_CAP:-90}", self.BOX)

    def test_reachability_probes_stay_separately_bounded(self) -> None:
        # One budget for both would let a slow pull spend the probe allowance and
        # condemn the machine as having refused the key.
        self.assertIn("${RENT_WAIT:-15}", self.BOX)
        self.assertIn('if [ "$state" = running ]; then', self.BOX)


class SkipListTest(unittest.TestCase):
    """Machines that refuse the ssh key must stay excluded across runs, but not forever.

    28852 and 27960 each took a rental and refused the key twice in one day, because
    `skip` lived only for the run that discovered them. A permanent denylist is the wrong
    answer -- the fault is usually transient -- so quarantine entries age out.
    """

    BOX = ROOT / "scripts" / "infra" / "box.sh"

    def skip_list(self, quarantine: str, exclude: str = "", ttl: int = 14400) -> str:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "q.log"
            f.write_text(quarantine)
            out = subprocess.run(
                ["bash", str(self.BOX), "skiplist"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "QUARANTINE": str(f),
                     "QUARANTINE_TTL": str(ttl), "EXCLUDE_MACHINES": exclude},
            )
            # Not `2>/dev/null`: box.sh dies at `cmd=$1` under `set -u` when sourced, and
            # a silenced failure returns "" -- which two of these tests assert.
            self.assertEqual(out.returncode, 0, out.stderr)
            return out.stdout.strip()

    def fresh(self, *machines: int) -> str:
        now = int(time.time())
        return "".join(f"{now - 60} {m}\n" for m in machines)

    def stale(self, *machines: int) -> str:
        old = int(time.time()) - 20000        # older than the 4h default
        return "".join(f"{old} {m}\n" for m in machines)

    def test_a_recent_failure_is_skipped(self) -> None:
        self.assertEqual(self.skip_list(self.fresh(28852)), "28852")

    def test_an_expired_failure_comes_back(self) -> None:
        self.assertEqual(self.skip_list(self.stale(28852)), "")

    def test_the_permanent_set_and_the_quarantine_merge(self) -> None:
        self.assertEqual(
            self.skip_list(self.fresh(28852), exclude="28070,28077"),
            "28070,28077,28852",
        )

    def test_a_machine_in_both_appears_once(self) -> None:
        self.assertEqual(self.skip_list(self.fresh(28070), exclude="28070"), "28070")

    def test_no_quarantine_file_is_empty_not_an_error(self) -> None:
        self.assertEqual(self.skip_list(""), "")

    def test_a_repeat_offender_is_listed_once(self) -> None:
        self.assertEqual(self.skip_list(self.fresh(28852) * 3), "28852")


class PowerCapTest(unittest.TestCase):
    """A throttled card must not take a throughput leg.

    Run 10 drew a 3090 capped at 200 W to pair against a 390 W baseline. Decode is what the
    leg measures, so the rows measure the cap; they are void.
    """

    BOX = (ROOT / "scripts" / "infra" / "box.sh").read_text()
    DRIVER = (ROOT / "orchestration" / "session2" / "polyglot-cost.sh").read_text()

    def test_the_offer_filter_reads_the_cap(self) -> None:
        self.assertIn("powermin = float(os.environ['POWERMIN'])", self.BOX)
        self.assertIn("and (o.get('gpu_max_power') or 0) >= powermin", self.BOX)

    def test_it_defaults_off_so_fit_only_arms_are_unaffected(self) -> None:
        # A context-ceiling ladder does not care about wattage, and a floor it did not ask
        # for would shrink its pool for nothing.
        self.assertIn("POWERMIN=${POWERMIN:-0}", self.BOX)

    def test_the_speed_leg_sets_a_floor(self) -> None:
        self.assertIn("export POWERMIN=${POWERMIN:-340}", self.DRIVER)


class StrictModesRepairTest(unittest.TestCase):
    """Some hosts deliver the key with ownership OpenSSH StrictModes rejects.

    Measured: `vastai logs` on such a box shows `Authentication refused: bad ownership or
    modes for file /root/.ssh/authorized_keys` beside `Failed publickey` for our exact
    fingerprint, while a healthy box logs `Accepted key ... found at
    /root/.ssh/authorized_keys:1`. An onstart that only chmods still fails; chown fixes it.
    """

    BOX = (ROOT / "scripts" / "infra" / "box.sh").read_text()

    def test_the_instance_repairs_ownership_not_just_mode(self) -> None:
        self.assertIn("chown -R root:root /root/.ssh", self.BOX)
        self.assertIn("chmod 600 /root/.ssh/authorized_keys", self.BOX)

    def test_the_repair_is_bounded(self) -> None:
        # The file appears after onstart starts, so a single pass can miss it; an unbounded
        # loop would outlive the rent window and keep a core busy for the whole leg.
        self.assertIn("for i in $(seq 1 60); do chown", self.BOX)

    def test_a_failed_draw_records_why(self) -> None:
        # "never accepted the key" cost five rentals because the reason lives only in the
        # container's sshd log, which the destroy threw away.
        self.assertIn('vastai logs "$id"', self.BOX)
        self.assertIn("Failed publickey|Accepted key", self.BOX)


class SyntaxTest(unittest.TestCase):
    # Recursive, and floored: a flat glob silently stops covering shell the moment a file
    # moves into a subdirectory, and the test keeps passing. Raise the floor when adding
    # shell; lower it only alongside a deletion.
    SHELL_DIRS = ("scripts", "serving", "orchestration")
    SHELL_FLOOR = 58

    def test_every_shell_file_parses(self) -> None:
        files = sorted(p for d in self.SHELL_DIRS for p in (ROOT / d).rglob("*.sh"))
        self.assertGreaterEqual(len(files), self.SHELL_FLOOR)
        for path in files:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertEqual(
                    subprocess.run(["bash", "-n", str(path)],
                                   capture_output=True, text=True).returncode,
                    0, f"{path} failed bash -n",
                )



class NoArgvTest(unittest.TestCase):
    """Every driver rents a box within seconds of starting, so argv must be refused before
    the spend. `./serve --help` rented one before this guard existed."""

    DRIVERS = [ROOT / "orchestration" / "serve-box.sh", *sorted(
        (ROOT / "orchestration" / "session2").glob("*.sh"))]

    def test_help_prints_the_header_and_exits_clean(self):
        for path in self.DRIVERS:
            with self.subTest(driver=path.name):
                p = subprocess.run([str(path), "--help"], capture_output=True,
                                   text=True, timeout=60)
                self.assertEqual(p.returncode, 0)
                self.assertGreater(len(p.stdout.strip().splitlines()), 3)
                self.assertNotIn("#", p.stdout.splitlines()[0])

    def test_any_positional_argument_is_refused(self):
        for path in self.DRIVERS:
            with self.subTest(driver=path.name):
                p = subprocess.run([str(path), "bogus"], capture_output=True,
                                   text=True, timeout=60)
                self.assertEqual(p.returncode, 2)
                self.assertIn("takes no arguments", p.stderr)

    def test_every_renting_driver_installs_the_guard(self):
        """A new driver that forgets no_argv is the failure this class exists to prevent."""
        for path in self.DRIVERS:
            with self.subTest(driver=path.name):
                self.assertIn('no_argv "$@"', path.read_text())


if __name__ == "__main__":
    unittest.main()
