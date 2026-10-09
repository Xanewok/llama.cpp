"""k9fix_summary.py OUT: perf-<dev>-<arm>-<n>.out files of bw-k9-fix.sh -> TSV on stdout, verdict lines on stderr.

One row per (dev, type, m1, n): median over runs of the per-run median us/graph for each arm (base = stock MMQ, k9old = K9 as
of 1039f834f, fix = this tree with PDL on, off = this tree with GGML_CUDA_MMQ_PDL=0), the A/A spread over the base runs and
the deltas to base. Verdicts (brief, 2026-10-09): the fix is expected within the A/A spread on >= 90 % of rows; the old K9 must
still be slower than the A/A spread on >= 90 % of rows, or the bench has lost the a3 regression and the fix verdict is void.
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ('base', 'k9old', 'fix', 'off')
NAME = re.compile(r'^perf-(CUDA\d)-(base|k9old|fix|off)-\d+\.out$')
LINE = re.compile(r'^perf (\S+) m1=(\d+) n=(\d+) min_us=([\d.]+) median_us=([\d.]+)$')


def main():
    out = Path(sys.argv[1])
    cells = defaultdict(list)
    for path in sorted(out.glob('perf-*.out')):
        m = NAME.match(path.name)
        if not m:
            continue
        for line in path.read_text().splitlines():
            r = LINE.match(line)
            assert r, (path, line)
            cells[(m.group(1), r.group(1), int(r.group(2)), int(r.group(3)), m.group(2))].append(float(r.group(5)))
    keys = sorted({k[:4] for k in cells})
    assert keys, 'no perf rows'
    print('dev\ttype\tm1\tn\tbase_us\tk9old_us\tfix_us\toff_us\taa_spread_us\tfix_minus_base\tk9old_minus_base\toff_minus_base')
    fix_in = old_out = 0
    fix_d, old_d = [], []
    for dev, typ, m1, n in keys:
        v = {a: cells[(dev, typ, m1, n, a)] for a in ARMS}
        assert len(v['base']) >= 2 and all(v[a] for a in ARMS), (dev, typ, m1, n)
        b, o, f, x = (statistics.median(v[a]) for a in ARMS)
        aa = max(v['base']) - min(v['base'])
        fix_in += f - b <= aa
        old_out += o - b > aa
        fix_d.append(f - b)
        old_d.append(o - b)
        print(f'{dev}\t{typ}\t{m1}\t{n}\t{b:.2f}\t{o:.2f}\t{f:.2f}\t{x:.2f}\t{aa:.2f}\t{f - b:+.2f}\t{o - b:+.2f}\t{x - b:+.2f}')
    rows = len(keys)
    control = 'ok' if old_out >= 0.9*rows else 'VOID'
    expected = 'met' if fix_in >= 0.9*rows else 'NOT met'
    print(f'control k9old-base beyond A/A: {old_out}/{rows} rows, mean {statistics.mean(old_d):+.2f} us/graph ({control})',
          file=sys.stderr)
    print(f'fix-base within A/A: {fix_in}/{rows} rows, mean {statistics.mean(fix_d):+.2f} us/graph, '
          f'median {statistics.median(fix_d):+.2f} (expected >= 90 %: {expected})', file=sys.stderr)


main()
