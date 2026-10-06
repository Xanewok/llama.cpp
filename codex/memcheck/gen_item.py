#!/usr/bin/env python3
"""Writes item.toml; the pin list is the live file set of both frozen builds, so run it after every freeze."""
import os, sys

W = '/home/xanewok/repos/qwen38-eval-worktrees'
HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.realpath(f'{HERE}/../frozen/graph-capture-spike')
STACK6 = f'{W}/frozen/stack6'
GPUS = 'GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f'
RES = '/home/xanewok/repos/qwen38-eval/results/2026-10-06-graph-capture-memcheck'
RAW = '/mnt/scratch/work/graph-capture-spike/memcheck'
CAP_S = 1500  # run stage: memcheck slows kernels ~100x, a 4B boot plus 32 tokens under it is the long pole
CANARY_S = 840  # gpuq caps canary.timeout_s at 900; a control that hits its cap is 'not applicable', not a finding

def pins():
    out = [f'{HERE}/memcheck.py', f'{HERE}/gen_item.py']
    out += sorted(f'{HERE}/fixtures/{f}' for f in os.listdir(f'{HERE}/fixtures'))
    for root in (STACK6, SPIKE):
        out += [f'{root}/SOURCE', f'{root}/RECIPE.txt']
        b = f'{root}/build/bin'
        out += sorted(f'{b}/{f}' for f in os.listdir(b))
    return out

def main():
    mc = f'{HERE}/memcheck.py'
    env = lambda sub, budget: (f'{{ CUDA_VISIBLE_DEVICES = "{GPUS}", GPUQ_ACTIVE = "1", BUDGET_S = "{budget}", '
                               f'OUT = "{RES}/{sub}a{{attempt}}", RAW = "{RAW}/{sub}a{{attempt}}" }}')
    t = f'''id = "graph-capture-memcheck"
title = "Both GPUs: compute-sanitizer memcheck of the in-graph AllReduce node (GGML_META_AR_NODE=1), stack6 control first"
# Stage caps: canary {CANARY_S} s, run {CAP_S}+60 s; budget_s = preflight 120 + canary 900 + run 1560 + margin.
priority = 90
kind = "measure"
budget_s = 2900
est_s = 1800
ab = false
retry = "fresh"
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = [
  {{ class = "integrity", regex = "probe: integrity" }},
  {{ class = "timeout", regex = "probe: budget" }},
]

[run]
cmd = ["python3", "{mc}", "--run"]
cwd = "{HERE}"
env = {env('', CAP_S + 60)}
ports = []

[preflight]
cmd = ["python3", "{mc}", "--preflight"]
timeout_s = 120

[canary]
cmd = ["python3", "{mc}", "--canary"]
env = {env('canary-', CANARY_S)}
timeout_s = {CANARY_S + 60}

[outputs]
paths = ["{RES}/a{{attempt}}"]

[fix]
editable = []
protected = ["probe", "frozen", "budget"]

[pins]
files = [
''' + ''.join(f'  "{p}",\n' for p in pins()) + ']\n'
    open(f'{HERE}/item.toml', 'w').write(t)
    print(f'item.toml: {len(pins())} pins')

if __name__ == '__main__':
    sys.exit(main())
