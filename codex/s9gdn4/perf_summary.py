"""perf_summary.py OUT: median us/run per (gpu, case, arm) from perfA/perfB logs; delta cand vs base and each arm's own spread.

The spread (max - min over the rounds, as % of the median) of the same arm is the noise floor a delta must clear.
perfC-gpu0-oldh-*.log: the old perf harness with the candidate kernel; its fusion rows timed ggml_sum, so at 6 tokens the new
harness rows (cand) must be larger (control table at the end, ratio new/old).
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r'^\s*(GATED_DELTA_NET\w*\(.*\)):\s+\d+ runs\s+-\s+([\d.]+) us/run')
NAME = re.compile(r'perf([ABC])-gpu(\d)-(base|cand|oldh)-\d+\.log$')


def main():
    out = Path(sys.argv[1])
    runs = defaultdict(list)
    for log in sorted(out.glob('perf[ABC]-gpu*-*.log')):
        m = NAME.match(log.name)
        assert m, log
        got = 0
        for line in log.read_text(errors='replace').splitlines():
            if (hit := LINE.match(re.sub(r'\x1b\[[0-9;]*m', '', line))):
                runs[(m[2], hit[1], m[3])].append(float(hit[2]))
                got += 1
        assert got, f'no perf rows in {log}'
    cases = sorted({(g, c) for g, c, a in runs if a != 'oldh'})
    assert cases, 'no perf lines'
    spread = lambda v: 100 * (max(v) - min(v)) / statistics.median(v)
    print('gpu\tcase\tbase_us\tcand_us\tdelta_pct\tbase_spread_pct\tcand_spread_pct\tn_base\tn_cand')
    for g, c in cases:
        b, k = runs.get((g, c, 'base')), runs.get((g, c, 'cand'))
        assert b and k, f'case without both arms: gpu{g} {c}'
        mb, mk = statistics.median(b), statistics.median(k)
        print(f'{g}\t{c}\t{mb:.2f}\t{mk:.2f}\t{100 * (mk / mb - 1):+.1f}\t{spread(b):.1f}\t{spread(k):.1f}\t{len(b)}\t{len(k)}')
    print('\nCONTROL (gpu0, fusion rows): old harness timed ggml_sum; new harness times gdn+cpy')
    print('case\told_harness_us\tnew_harness_us\tnew_over_old')
    for g, c, a in sorted(runs):
        if a != 'oldh':
            continue
        k = runs.get((g, c, 'cand'))
        assert k, f'no cand rows for {c}'
        mo, mk = statistics.median(runs[(g, c, a)]), statistics.median(k)
        print(f'{c}\t{mo:.2f}\t{mk:.2f}\t{mk / mo:.2f}')


main()
