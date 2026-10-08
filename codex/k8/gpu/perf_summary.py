"""perf_summary.py OUT canary|run: test-backend-ops --output sql files of k8-gpu.sh perf -> TSV on stdout.

Run mode: one row per (gpu, hot|cold, type, m, k, n) with the median time_us per setting (default tile, L1 policy auto-l1, K8 policy auto,
auto-drop), the deltas against the default tile and against the L1 policy, and the spread between the default runs (A/A noise).
auto-drop skips every SMALL_I GEMM (a no-GEMM floor where K8 or L1 launch, not a null control) and equals default elsewhere.
"""
import math
import re
import sqlite3
import statistics
import sys
from pathlib import Path

TAG = re.compile(r'^p-g(\d)-(hot|cold)-(?:r\d-)?(default|auto-drop|auto-l1|auto)-\d+\.out$')
CASE = re.compile(r'type_a=([^,]+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),bs=\[1,1\],nr=\[1,1\]')


def rows(path):
    content = path.read_text()
    statements = re.findall(r'CREATE TABLE IF NOT EXISTS test_backend_ops \([\s\S]*?\);|INSERT INTO test_backend_ops [^\n]*;', content)
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.executescript('\n'.join(statements))
    out = [dict(r) for r in db.execute('SELECT * FROM test_backend_ops WHERE op_name = "MUL_MAT"')]
    db.close()
    for r in out:
        assert r['supported'] and r['passed'], (path, r['op_params'])
        assert math.isfinite(r['time_us']) and r['time_us'] > 0 and r['n_runs'] >= 1, (path, r['op_params'])
    return out


def med(s, key):
    return statistics.median(s[key]) if s.get(key) else float('nan')


def main():
    out, mode = Path(sys.argv[1]), sys.argv[2]
    if mode == 'canary':
        got = rows(out / 'canary.out')
        assert len(got) == 1, len(got)
        return
    samples = {}
    for path in sorted(out.glob('p-g*.out')):
        m = TAG.match(path.name)
        assert m, path.name
        gpu, temp, setting = m.groups()
        for r in rows(path):
            t, mm, n, k = CASE.search(r['op_params']).groups()
            samples.setdefault((gpu, temp, t, int(mm), int(k), int(n)), {}).setdefault(setting, []).append(r['time_us'])
    print('gpu\ttemp\ttype\tm\tk\tn\tdefault_us\tautol1_us\tauto_us\tautodrop_us\tauto_minus_default_us\tautol1_minus_default_us\t'
          'auto_minus_autol1_us\taa_default_spread_us\tauto_vs_default_pct')
    for key in sorted(samples):
        s = samples[key]
        if not s.get('default') or not s.get('auto'):
            continue
        d, l1, a, x = med(s, 'default'), med(s, 'auto-l1'), med(s, 'auto'), med(s, 'auto-drop')
        spread = max(s['default']) - min(s['default']) if len(s['default']) > 1 else float('nan')
        print('\t'.join([*map(str, key), f'{d:.2f}', f'{l1:.2f}', f'{a:.2f}', f'{x:.2f}', f'{a - d:+.2f}', f'{l1 - d:+.2f}', f'{a - l1:+.2f}',
                         f'{spread:.2f}', f'{100 * (a - d) / d:+.1f}']))


if __name__ == '__main__':
    main()
