"""Emit codex/stack11/window/bw-stack11-base.toml (run again after any change of the tree's source diff or of the window's files).

usage: mk_window_toml.py > codex/stack11/window/bw-stack11-base.toml
Times (G): est_s is stack10b's window (7200 s M-ish: build, component gates, full suite on both GPUs, K11a) plus the stack11 additions
(FA T1-T7 and touched ops on both GPUs ~1500 s, GDN and MMVQ ~600 s, four health boots ~1200 s, KLD dump build ~120 s, FA q8_0
timing A/B 5+5 runs on one GPU ~300 s); budget_s as stack10b's (60000 s, D cap).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_pins

ROOT = check_pins.ROOT
HERE = check_pins.HERE
UUIDS = 'GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f'
SCRIPT = HERE / 'bw-stack11-base.sh'
EST = 7200 + 1500 + 600 + 1200 + 120 + 300
BUDGET = 60000
OUT = ROOT / 'codex/window/stack11-base-a{attempt}'


def head():
    title = ("stack11-base CUDA window: build, np2 joint-replay smoke, every component's CUDA gates (FA T1-T7, MMVQ, MMQ tiles, GDN joint, "
             "K4 K6 K7 K10 K11b, meta subgraph), fafix SASS, identity and FA q8_0 timing vs frozen/stack10, touched ops and the full test-backend-ops on both "
             "GPUs, K11a sync count, served-flag health boots of the four swift15 profiles, freeze as stack11-base, build kld_dump pair, "
             "render/check the A/B and KLD items (NOT submitted); Mac CPU gates required")
    return f'''id = "bw-stack11-base"
title = "{title}"
kind = "filler"
priority = 19
host_sensitive = false
preemptible = false
# G: stack10b window 7200 + FA/touched ops 1500 + GDN/MMVQ 600 + health boots 1200 + KLD build 120 + FA timing 300 = {EST} s; D: budget as stack10b's.
est_s = {EST}
budget_s = {BUDGET}
ab = false
retry = "never"
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = [{{ class = "deterministic", regex = "bw-stack11: FAIL" }}]

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
