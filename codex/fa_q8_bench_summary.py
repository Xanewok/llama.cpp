"""Summarize fa-q8-bench output. usage: fa_q8_bench_summary.py OUT.json LOG [LOG...]  (one log per device)

Per tile count: the worst ik/cv ratio over the shapes with that many query tiles, per device. Cap suggestion: the largest T such
that every shape with tiles <= T has ratio <= 1 + SLACK on every device (the ik arm is then no slower than convert + f16 for
all batch sizes up to T tiles). Exit 1 when a row is missing, the ik arm did not engage where it must, or the cv arm engaged.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SLACK = 0.02


def rows(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('bench '):
            d = dict(kv.split('=', 1) for kv in line.split()[1:])
            out.append(dict(dev=d['dev'], dk=int(d['dk']), gqa=int(d['gqa']), seqs=int(d['seqs']), kv=int(d['kv']), nb=int(d['nb']),
                            tiles=int(d['tiles']), ik=float(d['ik_us']), cv=float(d['cv_us']), ratio=float(d['ratio']),
                            eng_ik=[int(x) for x in d['engaged_ik'].split(',')], eng_cv=[int(x) for x in d['engaged_cv'].split(',')],
                            buf_ik=float(d['buf_ik_mib']), buf_cv=float(d['buf_cv_mib']), same=int(d['same'])))
    return out


def main():
    out, logs = Path(sys.argv[1]), sys.argv[2:]
    allrows = [r for log in logs for r in rows(log)]
    bad = []
    if not allrows:
        bad.append('no bench rows')
    devs = sorted({r['dev'] for r in allrows})
    per_dev = {d: [r for r in allrows if r['dev'] == d] for d in devs}
    if len({len(v) for v in per_dev.values()}) > 1:
        bad.append(f'row counts differ per device: { {d: len(v) for d, v in per_dev.items()} }')
    for r in allrows:
        mq = r['gqa'] >= 2
        if mq and not all(e == 1 for e in r['eng_ik']):
            bad.append(f'ik arm not engaged: {r}')
        if any(r['eng_cv']):
            bad.append(f'cv arm engaged: {r}')
        if not mq and any(r['eng_ik']):
            bad.append(f'gqa 1 engaged: {r}')
    worst = defaultdict(dict)
    for r in allrows:
        if r['gqa'] >= 2:
            worst[r['tiles']][r['dev']] = max(worst[r['tiles']].get(r['dev'], 0), r['ratio'])
    cap, ok = 0, True
    for t in sorted(worst):
        ok = ok and all(v <= 1 + SLACK for v in worst[t].values()) and len(worst[t]) == len(devs)
        if ok:
            cap = t
    differing = [r for r in allrows if not r['same']]
    scratch = [(r['dk'], r['gqa'], r['seqs'], r['kv'], r['buf_cv'] - r['buf_ik']) for r in allrows if r['gqa'] >= 2 and r['nb'] == 128]
    summary = dict(rows=len(allrows), devices=devs, worst_ratio_by_tiles={str(t): worst[t] for t in sorted(worst)},
                   suggested_cap_tiles=cap, slack=SLACK, rows_not_byte_identical=len(differing), violations=bad[:20],
                   scratch_saved_mib_nb128=scratch)
    out.write_text(json.dumps(summary, indent=1) + '\n')
    for t in sorted(worst):
        print(f'tiles={t:3d} worst ik/cv per device: {worst[t]}')
    print(f'suggested cap (tiles, no slower than convert+f16 within {SLACK:.0%} on all devices): {cap}')
    print(f'rows={len(allrows)} byte-different ik vs cv: {len(differing)}')
    for b in bad[:20]:
        print('VIOLATION', b)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
