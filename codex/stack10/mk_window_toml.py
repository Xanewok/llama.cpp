"""Emit codex/stack10/bw-stack10.toml (run again after any change of the tree's source diff or of the window's files).

usage: mk_window_toml.py > codex/stack10/bw-stack10.toml
       mk_window_toml.py tail > codex/stack10/bw-stack10-tail.toml   (STACK10_TAIL_FROM: reuse attempt a1's stages 40-65, run 66 63 70 71)
Tail times (G): full-suite 2 x 433 s (a1's CUDA0 wall) + self-test runs + flake reruns about 600 s, K11a about 230 s, freeze and render
about 90 s, precheck and controls about 30 s = 2000 s; budget twice that (D).
Times: est_s is the sum of the component windows' estimates (G; builds are counted once per component, so an upper bound),
budget_s twice that (D cap).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_pins

ROOT = check_pins.ROOT
SUM_EST = 4200 + 1800 + 2400 + 3600 + 2400 + 1800 + 5400 + 3000 + 3600 + 2400 + 4200
UUIDS = 'GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f'
SCRIPT = ROOT / 'codex/stack10/bw-stack10.sh'
TAIL_EST = 2000


def head(tail):
    est = TAIL_EST if tail else SUM_EST
    ident = 'bw-stack10-tail' if tail else 'bw-stack10'
    out = f'codex/window/stack10-{"tail-" if tail else ""}a{{attempt}}'
    title = ('stack10 tail window: reuse attempt a1 stages 40-65 after an integrity proof; full test-backend-ops on both GPUs, K11a sync count, '
             'freeze, render/check six A/B items' if tail else
             "stack10 combined CUDA window: build, every component's CUDA gates, fafix SASS, full test-backend-ops on both GPUs, freeze, "
             'render/check six A/B items; Mac CPU gates required')
    comment = ('# G: full-suite 2 x 433 s + reruns 600 + K11a 230 + freeze/render 90 + precheck 30 = 2000 s; D: budget twice that.' if tail else
               f'# G: sum of the component windows p55 4200 + k7 1800 + k4 2400 + k6 3600 + k2 2400 + k10 1800 + k9 5400 + k8 3000 + k11a 3600 + k11b 2400 + fafix 4200 = {SUM_EST} s\n'
               '# (one CUDA build is shared, so this overcounts by about ten builds); D: budget twice that.')
    tail_env = ' STACK10_TAIL_FROM = "stack10-a1",' if tail else ''
    return f'''id = "{ident}"
title = "{title}"
kind = "filler"
priority = 19
host_sensitive = false
preemptible = false
{comment}
est_s = {est}
budget_s = {2 * est}
ab = false
retry = "never"
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = [{{ class = "deterministic", regex = "bw-stack10: FAIL" }}]

[run]
cmd = ["/usr/bin/bash", "{SCRIPT}", "{ROOT / out}"]
cwd = "{ROOT}"
env = {{ PATH = "/home/xanewok/.local/bin:/usr/local/bin:/usr/bin:/bin", CUDA_VISIBLE_DEVICES = "{UUIDS}", GIT_NO_LAZY_FETCH = "1", GPUQ_BUILD_WINDOW = "1",{tail_env} GPUQ_ACTIVE = "1", PORT = "{{port}}" }}
ports = "dynamic"

[preflight]
cmd = ["/usr/bin/bash", "{SCRIPT}", "--preflight"]
timeout_s = {300 if tail else 120}

[outputs]
paths = ["{ROOT / out}/summary.json", "{ROOT / out}/payload.log"]

[fix]
editable = []
protected = ["frozen", "budget", "stages", "controls", "correctness"]

[pins]
files = [
'''


def main():
    tail = sys.argv[1:] == ['tail']
    sys.stdout.write(head(tail) + ''.join(f'  {json.dumps(p)},\n' for p in check_pins.required(tail)) + ']\n')


if __name__ == '__main__':
    main()
