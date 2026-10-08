"""perf_summary.py OUT canary|run: test-backend-ops --output sql files of l1-gpu.sh perf -> TSV on stdout.

Run mode: one row per (gpu, hot|cold, type, m, k, n) with the median time_us per setting (default tiles, auto policy, auto-drop null
control), the auto-default delta, the spread between the two default runs (A/A noise) and the auto-drop-default delta.
"""
import math
import re
import sqlite3
import statistics
import sys
from pathlib import Path

TAG = re.compile(r'^p-g(\d)-(hot|cold)-(?:r\d-)?(default|auto-drop|auto)-\d+\.out$')
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
    print('gpu\ttemp\ttype\tm\tk\tn\tdefault_us\tauto_us\tautodrop_us\tauto_minus_default_us\taa_default_spread_us\tautodrop_minus_default_us')
    for key in sorted(samples):
        s = samples[key]
        d, a = s.get('default', []), s.get('auto', [])
        if not d or not a:
            continue
        md, ma = statistics.median(d), statistics.median(a)
        x = s.get('auto-drop')
        mx = statistics.median(x) if x else float('nan')
        spread = max(d) - min(d) if len(d) > 1 else float('nan')
        print('\t'.join([*map(str, key), f'{md:.2f}', f'{ma:.2f}', f'{mx:.2f}', f'{ma - md:+.2f}', f'{spread:.2f}', f'{mx - md:+.2f}']))


if __name__ == '__main__':
    main()
