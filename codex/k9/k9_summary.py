"""k9_summary.py OUT: perf-<dev>-<arm>-<n>.out files of bw-k9.sh -> TSV on stdout.

One row per (dev, type, m1, n): median over runs of the per-run median us/graph for each arm (base = stock MMQ, k9 = PDL on,
off = k9 binary with GGML_CUDA_MMQ_PDL=0), the A/A spread between the two base runs, and the k9-base and off-base deltas.
A delta is only a finding when it exceeds the A/A spread.
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r'^perf-(CUDA\d)-(base|k9|off)-\d+\.out$')
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
    print('dev\ttype\tm1\tn\tbase_us\tk9_us\toff_us\taa_spread_us\tk9_minus_base\toff_minus_base')
    for dev, typ, m1, n in keys:
        base = cells[(dev, typ, m1, n, 'base')]
        k9 = cells[(dev, typ, m1, n, 'k9')]
        off = cells[(dev, typ, m1, n, 'off')]
        assert len(base) >= 2 and k9 and off, (dev, typ, m1, n)
        b, a, o = statistics.median(base), statistics.median(k9), statistics.median(off)
        print(f'{dev}\t{typ}\t{m1}\t{n}\t{b:.2f}\t{a:.2f}\t{o:.2f}\t{max(base) - min(base):.2f}\t{a - b:+.2f}\t{o - b:+.2f}')


main()
