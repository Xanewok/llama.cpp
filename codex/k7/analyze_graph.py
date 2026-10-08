#!/usr/bin/env python3
"""Count q/k RMS_NORM -> SCALE chain pairs in a GGML_SCHED_DEBUG=2 node dump (view nodes are not printed)."""
import re
import sys
from collections import Counter

NODE = re.compile(r'node #\s*(\d+) \(\s*(\w+)\):.*?use=(\d+),c=(\d):[\d.]* D\s+(\S+)')
ops = [(int(m[1]), m[2], int(m[3]), m[5]) for m in NODE.finditer(open(sys.argv[1], errors='replace').read())]

pairs = q_rms = k_rms = 0
gaps = Counter()
for i, (idx, op, use, src) in enumerate(ops):
    if op != 'RMS_NORM' or not src.startswith(('q_conv', 'k_conv')):
        continue
    q_rms += src.startswith('q_conv')
    k_rms += src.startswith('k_conv')
    if src.startswith('q_conv') and i + 3 < len(ops) and [o[1] for o in ops[i:i + 4]] == ['RMS_NORM', 'SCALE', 'RMS_NORM', 'SCALE'] \
            and use == 1 and ops[i + 2][2] == 1 and ops[i + 2][3].startswith('k_conv') and ops[i + 1][0] == idx + 1:
        pairs += 1
        gaps[ops[i + 2][0] - ops[i + 1][0] - 1] += 1  # no-op nodes between the chains
print(f'nodes={len(ops)} q_conv_rms={q_rms} k_conv_rms={k_rms} adjacent_pairs={pairs} noop_nodes_between={dict(gaps)}')
sys.exit(0 if q_rms > 0 and q_rms == k_rms == pairs else 1)
