"""Parse test-backend-ops console logs.

  check_ops.py passed LOG MIN_CASES   every case OK, N/N tests passed with N >= MIN_CASES, no FAIL line
  check_ops.py failed LOG PATTERN     negative control: some case line matching PATTERN is FAIL
  check_ops.py perf LOG MIN_ROWS      at least MIN_ROWS "us/run" rows, every one supported
"""
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
SUMMARY = re.compile(r'^\s+(\d+)/(\d+) tests passed\s*$')
PERF = re.compile(r'^\s+(GATED_DELTA_NET\w*)\((.*)\):\s+(\d+) runs\s+-\s+([\d.]+) us/run')


def lines(path):
    with open(path, errors='replace') as f:
        return [ANSI.sub('', l.rstrip('\n')) for l in f]


def main():
    mode, log = sys.argv[1], sys.argv[2]
    ls = lines(log)
    if mode == 'passed':
        need = int(sys.argv[3])
        sums = [SUMMARY.match(l) for l in ls]
        sums = [m for m in sums if m]
        assert sums, f'{log}: no "N/N tests passed" line'
        for m in sums:
            assert m[1] == m[2], f'{log}: {m[1]}/{m[2]} tests passed'
            assert int(m[2]) >= need, f'{log}: only {m[2]} cases, need {need}'
        bad = [l for l in ls if l.rstrip().endswith('FAIL') or 'not supported' in l]
        assert not bad, f'{log}: {bad[:3]}'
        print(f'{log}: {sums[0][2]} cases passed')
    elif mode == 'failed':
        pat = re.compile(sys.argv[3])
        hit = [l for l in ls if l.rstrip().endswith('FAIL') and pat.search(l)]
        assert hit, f'{log}: no FAIL line matches {sys.argv[3]!r}'
        print(f'{log}: {len(hit)} failing cases match; first: {hit[0][:160]}')
    elif mode == 'perf':
        need = int(sys.argv[3])
        rows = [PERF.match(l) for l in ls]
        rows = [m for m in rows if m]
        assert len(rows) >= need, f'{log}: {len(rows)} perf rows, need {need}'
        assert not any('not supported' in l for l in ls), f'{log}: unsupported case'
        print(f'{log}: {len(rows)} perf rows')
    else:
        sys.exit(f'unknown mode {mode}')


try:
    main()
except AssertionError as e:
    print(f'check_ops: FAIL {e}')
    sys.exit(1)
