"""`box.sh rent` must not pick a host whose bandwidth price dominates the leg cost.

Offers are sorted by `-o dph`, which sees only the hourly rate. Instance 49058864 billed
$0.059 of GPU time and $0.711 of download -- 90 percent of the leg -- on a host whose
hourly rate looked competitive, while the same pool held bandwidth at a fifteenth the
price. This drives the real script against a stub `vastai` and asserts on the offer id
that reaches `create instance`.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOX = ROOT / "scripts" / "infra" / "box.sh"

# Ordered as `-o dph` would return them: the cheap-by-the-hour offer is also the one that
# charges fifteen times as much per GB, which is the trap the cap exists to catch.
STUB = r"""#!/usr/bin/env bash
case "$1 $2" in
  "search offers")
    cat <<'JSON'
[{"id": 111, "machine_id": 11, "dph_total": 0.16, "inet_down_cost": 0.0390,
  "gpu_max_power": 400, "reliability2": 0.999, "inet_down": 1000,
  "cpu_cores_effective": 32, "cuda_max_good": 13.5, "verification": "verified"},
 {"id": 222, "machine_id": 22, "dph_total": 0.17, "inet_down_cost": 0.0026,
  "gpu_max_power": 400, "reliability2": 0.999, "inet_down": 1000,
  "cpu_cores_effective": 32, "cuda_max_good": 13.5, "verification": "verified"}]
JSON
    ;;
  "create instance") echo "$3" >> "$OFFER_LOG"; echo '{"new_contract": 4242424}' ;;
  "show instance")   echo '{"actual_status": "running", "status_msg": "",
                          "ssh_host": "stub.invalid", "ssh_port": 2222}' ;;
  "destroy instance") echo "destroyed $3" ;;
  *) : ;;
esac
"""


class OfferCostTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        bindir = d / "bin"
        bindir.mkdir()
        for name, body in (("vastai", STUB),
                           ("ssh", "#!/usr/bin/env bash\nexit 1\n"),
                           ("sleep", "#!/usr/bin/env bash\nexec /bin/sleep 0.05\n")):
            f = bindir / name
            f.write_text(body)
            f.chmod(0o755)
        self.offer_log = d / "offers.txt"
        self.env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "OFFER_LOG": str(self.offer_log),
            "STATE": str(d / "instances.log"),
            "QUARANTINE": str(d / "quarantine.log"),
            "RENT_CAP": "1",
            "RENT_WAIT": "1",
            "RENT_TRIES": "1",
        }
        self.addCleanup(self.tmp.cleanup)

    def _rent(self, **extra):
        subprocess.run([str(BOX), "rent", "test-ttl4", "RTX 3090", "1", "80"],
                       capture_output=True, text=True, timeout=120,
                       env={**self.env, **extra})
        return self.offer_log.read_text().split() if self.offer_log.exists() else []

    def test_the_default_cap_skips_the_expensive_bandwidth_offer(self):
        self.assertEqual(["222"], self._rent(),
                         "rent took the $0.0390/GB offer over the $0.0026/GB one")

    def test_a_zero_cap_disables_the_filter(self):
        self.assertEqual(["111"], self._rent(DLCOSTMAX="0"),
                         "DLCOSTMAX=0 must not filter on bandwidth price")


if __name__ == "__main__":
    unittest.main()
