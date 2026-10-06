#!/usr/bin/env python3
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'memcheck/memcheck.py'
ROOT = HERE / 'frozen/gdn-getter-stack6'
RECORDED = HERE / 'memcheck/fixtures'
FIXTURES = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')
files = [SCRIPT, ROOT / 'SOURCE', ROOT / 'RECIPE.txt', FIXTURES / 'xid-lines.txt']
files += sorted((ROOT / 'build/bin').glob('*'))
files += [RECORDED / (n + '.log') for n in ['mc-old', 'mc-idx', 'rc-idx', 'sc-idx']]
q = json.dumps
run_out = HERE / 'gpu-results/memcheck/full-a{attempt}'
canary_out = HERE / 'gpu-results/memcheck/canary-a{attempt}'
lines = [
    'id = "gdn-getter-stack6-memcheck"',
    'title = "GDN getter: memcheck, racecheck, synccheck including multi-sequence indexed reads"',
    '# D: full stage caps 240 + 180 + 180 + 180 = 780 s <= 0.7 x 1200 s; canary caps total 210 s.',
    '# harness.mkitem supports server experiments only; this kernel-only manifest follows the prior sanitizer item.',
    'priority = 90', 'kind = "measure"', 'budget_s = 1800', 'est_s = 900',
    'ab = false', 'retry = "fresh"',
    'exit_codes = { "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }',
    'signatures = [{ class = "integrity", regex = ' + q(r'\bintegrity\b') + ' }, { class = "timeout", regex = ' + q(r'\bbudget\b') + ' }]',
    '[run]', 'cmd = ' + q(['python3', str(SCRIPT), '--stage', 'run', '--out', str(run_out), '--budget', '1200']),
    'cwd = ' + q(str(HERE)), 'env = { GPUQ_ACTIVE = "1", CUDA_VISIBLE_DEVICES = "GPU-ee9a3956-297c-604b-b4c1-f07930d27e46", PYTHONDONTWRITEBYTECODE = "1" }', 'ports = []',
    '[preflight]', 'cmd = ' + q(['python3', str(SCRIPT), '--stage', 'preflight']), 'timeout_s = 120',
    '[canary]', 'cmd = ' + q(['python3', str(SCRIPT), '--stage', 'canary', '--out', str(canary_out), '--budget', '450']),
    'env = { GPUQ_ACTIVE = "1", CUDA_VISIBLE_DEVICES = "GPU-ee9a3956-297c-604b-b4c1-f07930d27e46", PYTHONDONTWRITEBYTECODE = "1" }', 'timeout_s = 450',
    '[outputs]', 'paths = ' + q([str(run_out)]),
    '[fix]', 'editable = []', 'protected = ["frozen", "coverage", "correctness", "budget", "memcheck_first"]',
    '[pins]', 'files = [', *['  ' + q(str(p)) + ',' for p in files], ']', ''
]
(HERE / 'memcheck/item.toml').write_text('\n'.join(lines))
