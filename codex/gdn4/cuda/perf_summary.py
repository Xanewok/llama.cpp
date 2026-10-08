"""perf_summary.py OUT: median us/run per (gpu, case, arm) from perfA/perfB logs; delta cand vs base and each arm's own spread.

The spread (max - min over the rounds, as % of the median) of the same arm is the noise floor a delta must clear.
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r'^\s*(GATED_DELTA_NET\w*\(.*\)):\s+\d+ runs\s+-\s+([\d.]+) us/run')
NAME = re.compile(r'perf[AB]-gpu(\d)-(base|cand)-\d+\.log$')


def main():
    out = Path(sys.argv[1])
    runs = defaultdict(list)
    for log in sorted(out.glob('perf[AB]-gpu*-*.log')):
        m = NAME.match(log.name)
        assert m, log
        for line in log.read_text(errors='replace').splitlines():
            if (hit := LINE.match(re.sub(r'\x1b\[[0-9;]*m', '', line))):
                runs[(m[1], hit[1], m[2])].append(float(hit[2]))
    cases = sorted({(g, c) for g, c, _ in runs})
    assert cases, 'no perf lines'
    rounds = {len(v) for v in runs.values()}
    assert len(rounds) <= 2, f'unequal round counts {rounds}'
    spread = lambda v: 100 * (max(v) - min(v)) / statistics.median(v)
    print('gpu\tcase\tbase_us\tcand_us\tdelta_pct\tbase_spread_pct\tcand_spread_pct\tn_base\tn_cand')
    for g, c in cases:
        b, k = runs.get((g, c, 'base')), runs.get((g, c, 'cand'))
        assert b and k, f'case without both arms: gpu{g} {c}'
        mb, mk = statistics.median(b), statistics.median(k)
        print(f'{g}\t{c}\t{mb:.2f}\t{mk:.2f}\t{100 * (mk / mb - 1):+.1f}\t{spread(b):.1f}\t{spread(k):.1f}\t{len(b)}\t{len(k)}')


main()
