#!/usr/bin/env python3
"""GPU0 idle between consecutive draft decodes (last kernel end of block i -> first kernel start of block i+1),
from an attrib-pack nsys sqlite. Rounds are V..V (np2 draft blocks are labelled C by the pack matcher).
Blocks 2..6 of a 7-block round are D1..D5; boundaries D1->D2..D4->D5 are what a draft chain removes.
Usage: bgap.py ATTRIB_PACK_DIR SQLITE HEAD_TYPE LABEL"""
import sys, statistics, json
sys.path.insert(0, sys.argv[1])
import attrib_pack as ap

path, head_type, label = sys.argv[2], sys.argv[3], sys.argv[4]
tr = ap.Trace(path, need_memcpy=False, need_runtime=True)
blocks = ap.llama_blocks(tr, head_type)
vi = [i for i, b in enumerate(blocks) if b['kind'] == 'V']
cut = tr.t0 + 750e6
res = {}
for d in tr.devs[:2]:
    gaps = {i: [] for i in range(1, 6)}   # boundary after block i (C->D1 is 1, D4->D5 is 5)
    span = []
    for a, b in zip(vi, vi[1:]):
        sel = blocks[a:b]
        if len(sel) != 7 or sel[0]['host_s'] < cut:
            continue
        ends = [max(k[1] for k in x['dev'][d]) for x in sel]
        starts = [min(k[0] for k in x['dev'][d]) for x in sel]
        for i in range(1, 6):
            gaps[i].append((starts[i + 1] - ends[i]) / 1e3)
        span.append((ends[6] - starts[2]) / 1e3)
    res[d] = {'n': len(span), 'gap_us_median': {f'{i}->{i+1}': round(statistics.median(v), 1) for i, v in gaps.items() if v},
              'D1..D5_span_us': round(statistics.median(span), 1) if span else None}
print(json.dumps({label: res}))
