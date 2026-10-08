#!/usr/bin/env python3
"""Summarize test-backend-ops perf logs of the shared-mask attention cases: s10_k2_perf.py OUT.json ARM=LOG..."""
import json
import re
import statistics
import sys
from collections import defaultdict

LINE = re.compile(r'FLASH_ATTN_EXT\((shared_mask=1[^)]*)\):\s+\d+ runs - +([0-9.]+) us/run')

out, *pairs = sys.argv[1:]
runs = defaultdict(lambda: defaultdict(list))
for pair in pairs:
    arm, path = pair.split('=', 1)
    with open(path, encoding='utf-8', errors='replace') as f:
        found = LINE.findall(f.read())
    assert len(found) == 4, f'{path}: expected 4 shared-mask cases, found {len(found)}'
    for case, us in found:
        runs[case][arm].append(float(us))

result = {}
for case, arms in sorted(runs.items()):
    row = {arm: dict(us=v, mean_us=statistics.mean(v), min_us=min(v)) for arm, v in arms.items()}
    if 'share0' in arms and 'share1' in arms:
        row['saving_us_mean'] = row['share0']['mean_us'] - row['share1']['mean_us']
        row['saving_us_min'] = row['share0']['min_us'] - row['share1']['min_us']
    result[case] = row
with open(out, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
for case, row in result.items():
    print(case, {k: round(v, 2) for k, v in row.items() if k.startswith('saving')})
