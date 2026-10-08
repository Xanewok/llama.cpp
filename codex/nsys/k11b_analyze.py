#!/usr/bin/env python3
"""K11b staging timeline per arm from exported cuda-pass traces (np2 round shape VCCCCCC).

usage: k11b_analyze.py OUTDIR BASE_ARM ARM...      (OUTDIR/<arm>/decode-8k.sqlite)
Metrics are means over complete rounds, in us relative to the GPU0 end of the previous block, per boundary i (after block i;
boundary 0 = V->C0, boundary 6 = D5->V): GPU1-GPU0 first-kernel lag, GPU1 wall of the next block, host memcpy gaps before each
cudaMemcpyBatchAsync, staging span, H2D chain per device. Verdict lines compare each arm with BASE_ARM on boundaries 0 and 6.
Same segmentation and trim as the X2 analysis (attrib-pack an.rounds_for)."""
import bisect
import collections
import statistics
import sys

sys.path.insert(0, '/home/xanewok/repos/attrib-pack')
sys.path.insert(0, '/home/xanewok/repos/attrib-pack/runs/hqgap8-analysis')
import attrib_pack as ap  # noqa: E402
import an  # noqa: E402

SHAPE = 'VCCCCCC'
BIG = (0, 6)


def main_tid(tr):
    c = collections.Counter(tid for s, e, nm, cid, tid in tr.runtime if nm in ap.LAUNCH)
    return c.most_common(1)[0][0]


def analyze(path):
    tr = ap.Trace(path, need_memcpy=True, need_runtime=True)
    rounds, _, _, _, _ = an.rounds_for(tr, SHAPE)
    main = main_tid(tr)
    d0, d1 = tr.devs[:2]
    calls = sorted((s, e, nm) for s, e, nm, cid, tid in tr.runtime if tid == main)
    starts = [c[0] for c in calls]
    h2d = {d: sorted((m[0], m[1], m[3]) for m in tr.memcpy if m[2] == d and m[4] == 1) for d in (d0, d1)}
    hs = {d: [x[0] for x in h2d[d]] for d in h2d}
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    period = [(r['host_e'] - r['host_s']) / 1e6 for r in rounds]
    for r in rounds:
        seq = [b for _, b in r['phases']] + [r['next']]
        ext = {d: [(min(k[0] for k in b['dev'][d]), max(k[1] for k in b['dev'][d])) for b in seq] for d in (d0, d1)}
        for i in range(7):
            a = ext[d0][i][1]
            z = ext[d0][i + 1][0]
            w = calls[bisect.bisect_left(starts, a - 5e4):bisect.bisect_left(starts, z + 5e3)]
            mb = [k for k, (s, e, nm) in enumerate(w) if nm == 'cudaMemcpyBatchAsync' and s > a]
            if len(mb) < 2:
                continue
            m = acc[i]
            qs = [k for k in range(mb[0]) if w[k][2] == 'cudaEventQuery']
            if not qs:
                continue
            q0 = qs[-1]
            m['memcpy gap before MB0 (host)'].append((w[mb[0]][0] - w[mb[0] - 1][1]) / 1e3)
            m['memcpy gap before MB1 (host)'].append((w[mb[1]][0] - w[mb[1] - 1][1]) / 1e3)
            m['staging span (first query -> MB1 end)'].append((w[mb[1]][1] - w[q0][0]) / 1e3)
            m['GPU0 idle gap'].append((z - a) / 1e3)
            m['GPU1-GPU0 first-kernel lag'].append((ext[d1][i + 1][0] - ext[d0][i + 1][0]) / 1e3)
            m['GPU1 wall of next block'].append((ext[d1][i + 1][1] - ext[d1][i + 1][0]) / 1e3)
            first = min(ext[d0][i + 1][0], ext[d1][i + 1][0])
            last = max(ext[d0][i + 1][1], ext[d1][i + 1][1])
            m['block span (first kernel, any GPU -> last end)'].append((last - first) / 1e3)
            m['previous GPU0 end -> block end'].append((last - a) / 1e3)
            for d, nm in ((d0, 'dev0'), (d1, 'dev1')):
                xs = h2d[d][bisect.bisect_left(hs[d], a):bisect.bisect_left(hs[d], ext[d][i + 1][0])]
                if xs:
                    m[nm + ' H2D chain start'].append((xs[0][0] - a) / 1e3)
                    m[nm + ' H2D chain end'].append((max(x[1] for x in xs) - a) / 1e3)
    return acc, period


def main():
    out, base, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    res = {}
    for arm in [base] + [a for a in arms if a != base]:
        res[arm] = analyze(f'{out}/{arm}/decode-8k.sqlite')
        acc, period = res[arm]
        print(f'== arm {arm}: rounds {len(period)}, mean V->V period {statistics.fmean(period):.3f} ms (informational, 1 boot)')
        for i in sorted(acc):
            n = len(acc[i]['GPU0 idle gap'])
            print(f'  boundary after block {i} (rounds {n})')
            for k, v in acc[i].items():
                print(f'    {k:42s} {statistics.fmean(v):9.1f}')
    print('== K11b verdict (D = within-trace, 1 boot per arm; falsifier: lag stays > 40 us or the block span does not shrink by >= 40 us; GPU1 wall alone need not move, the lag removed from the start is removed from the end)')
    for arm in res:
        if arm == base:
            continue
        for i in BIG:
            try:
                lag = statistics.fmean(res[arm][0][i]['GPU1-GPU0 first-kernel lag'])
                lag0 = statistics.fmean(res[base][0][i]['GPU1-GPU0 first-kernel lag'])
                wall = statistics.fmean(res[arm][0][i]['GPU1 wall of next block'])
                wall0 = statistics.fmean(res[base][0][i]['GPU1 wall of next block'])
                span = statistics.fmean(res[arm][0][i]['block span (first kernel, any GPU -> last end)'])
                span0 = statistics.fmean(res[base][0][i]['block span (first kernel, any GPU -> last end)'])
                full = statistics.fmean(res[arm][0][i]['previous GPU0 end -> block end'])
                full0 = statistics.fmean(res[base][0][i]['previous GPU0 end -> block end'])
            except statistics.StatisticsError:
                print(f'K11B-VERDICT {arm} boundary {i}: no data')
                continue
            print(f'K11B-VERDICT {arm} boundary {i}: lag {lag:.1f} us (base {lag0:.1f}, <40: {"yes" if lag < 40 else "no"}); '
                  f'GPU1 wall {wall:.1f} (delta {wall - wall0:+.1f}); block span {span:.1f} (delta {span - span0:+.1f}, <= -40: '
                  f'{"yes" if span - span0 <= -40 else "no"}); previous end -> block end {full:.1f} (delta {full - full0:+.1f})')


if __name__ == '__main__':
    main()
