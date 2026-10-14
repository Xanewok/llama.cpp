"""Emit codex/stack11-s2/window/bw-stack11-s2.toml (run again after any change of either tree's source diff or of the window's files).

usage: mk_window_toml.py > codex/stack11-s2/window/bw-stack11-s2.toml
Times (G): two CUDA builds ~1800 s each (ccache warm from stack11-base), np2 smoke ~600 s, unit + host-mapped ~300 s, ops on both GPUs
(4 x 2 runs) ~1500 s, five health boots ~1500 s, two freezes ~120 s, render and check of six items ~600 s, meta_rows on both builds ~600 s, 12 placement boots ~1800 s; budget_s a D cap.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_pins

ROOT = check_pins.ROOT
HERE = check_pins.HERE
UUIDS = 'GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f'
SCRIPT = HERE / 'bw-stack11-s2.sh'
EST = 3600 + 600 + 300 + 1500 + 1500 + 120 + 600 + 600 + 1800
BUDGET = 28000
OUT = ROOT / 'codex/window/stack11-s2b-a{attempt}'


def head():
    title = ("stack11-s2 CUDA window: build stack11-base + 87/88/91/90 with and without K7/K10 (frozen/stack11-s2 from wt/stack11-s2-k, "
             "frozen/stack11-s2-nok from wt/stack11-s2), np2 joint-replay smoke, CPU-device unit tests, test-host-mapped and touched ops on both "
             "GPUs vs frozen/stack11-base, served-flag health boots, freeze both, render/check the A/B items (NOT submitted), pkg 90 placement proof on the frozen builds vs frozen/stack11-base; Mac CPU gates required")
    return f'''id = "bw-stack11-s2b"
title = "{title}"
kind = "filler"
priority = 22
host_sensitive = false
preemptible = false
# G: builds 3600 + np2 smoke 600 + unit/host-mapped 300 + ops 1500 + health boots 1500 + freeze 120 + render 600 + meta rows 600 + placement 1800 = {EST} s; D: budget cap.
est_s = {EST}
budget_s = {BUDGET}
ab = false
retry = "never"
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = [{{ class = "deterministic", regex = "bw-s2: FAIL" }}]

[run]
cmd = ["/usr/bin/bash", "{SCRIPT}", "{OUT}"]
cwd = "{ROOT}"
env = {{ PATH = "/home/xanewok/.local/bin:/usr/local/bin:/usr/bin:/bin", CUDA_VISIBLE_DEVICES = "{UUIDS}", GIT_NO_LAZY_FETCH = "1", GPUQ_BUILD_WINDOW = "1", GPUQ_ACTIVE = "1", PORT = "{{port}}", TMPDIR = "/mnt/scratch/tmp-build" }}
ports = "dynamic"

[preflight]
cmd = ["/usr/bin/bash", "{SCRIPT}", "--preflight"]
timeout_s = 300

[selftest]
cmd = ["/usr/bin/bash", "{SCRIPT}", "--selftest"]
timeout_s = 120

[outputs]
paths = ["{OUT}/summary.json", "{OUT}/payload.log"]

[fix]
editable = []
protected = ["frozen", "budget", "stages", "controls", "correctness"]

[pins]
files = [
'''


def main():
    sys.stdout.write(head() + ''.join(f'  {json.dumps(p)},\n' for p in check_pins.required()) + ']\n')


if __name__ == '__main__':
    main()
