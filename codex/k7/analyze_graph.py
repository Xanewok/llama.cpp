#!/usr/bin/env python3
"""Count q/k RMS_NORM -> SCALE pairs in a GGML_SCHED_DEBUG=2 node dump (view nodes are not printed)."""
import re
import sys

LINE = re.compile(r'node #\s*(\d+) \(\s*(\w+)\): (.{20}) \(.*?use=(\d+),c=(\d):(.*)$')
ops = []
for raw in open(sys.argv[1], errors='replace'):
    m = LINE.search(raw)
    if m:
        idx, op, name, use, comp, srcs = m.groups()
        # the first source name is a fixed 20-character field after one space
        ops.append((int(idx), op, name.strip(), int(use), srcs[1:21].strip()))

pairs = q_rms = k_rms = 0
for i, (idx, op, name, use, src) in enumerate(ops):
    if op != 'RMS_NORM' or not (src.startswith('q_conv') or src.startswith('k_conv')):
        continue
    q_rms += src.startswith('q_conv')
    k_rms += src.startswith('k_conv')
    if (src.startswith('q_conv') and i + 3 < len(ops)
            and [o[1] for o in ops[i:i + 4]] == ['RMS_NORM', 'SCALE', 'RMS_NORM', 'SCALE']
            and ops[i][3] == 1 and ops[i + 2][3] == 1 and ops[i + 2][4].startswith('k_conv')):
        pairs += 1
print(f'nodes={len(ops)} q_conv_rms={q_rms} k_conv_rms={k_rms} adjacent_pairs={pairs}')
sys.exit(0 if q_rms > 0 and q_rms == k_rms == pairs else 1)
