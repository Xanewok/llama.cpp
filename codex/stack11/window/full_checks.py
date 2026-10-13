"""Judge `test-backend-ops test --output sql` of the whole suite on one backend; flake handling after codex/p55_checks.py.

Contract: every supported `test` row of the backend carries this build's commit; no `FAIL` line (the components' byte-identity
gates print them); the PASS marker of every selected component is present; for p55 the quant x n=5..8 MUL_MAT coverage of the
recorded fixture holds. A failing row is never a verdict by itself: it goes to the retry list as `scope<TAB>op<TAB>params`.
  scope `in`  : a case the selected components' launch configs reach (p55: changed quants any n, any type n=5..8; K8: quantized
                type_a with n > 8 on MUL_MAT / MUL_MAT_ID). Excused only if the frozen/stack10 binary fails it too, 3 of 3.
  scope `out` : everything else (inputs are random per run, so near-threshold NMSE cases flake). Excused if it passes 3 fresh reruns,
                or fails 3 of 3 on the frozen/stack10 binary (a pre-existing failure, not a regression).
More than MAX_EXCUSED failing rows on one backend fails outright.

usage: full_checks.py --fixtures
       full_checks.py check SQL BACKEND COMMIT COMPONENTS   (writes SQL.retry; prints counts)
       full_checks.py --regex PARAMS
       full_checks.py --rerun SQL OP BACKEND COMMIT PARAMS  (rc 0: rows exist, all passed; rc 2: rows exist, some failed; rc 1: none)
       full_checks.py ops-check SQL BACKEND COMMIT SCOPE OP,OP... [SELFTEST_SQL...]
                            one op-filtered run: every OP has a passing row (a MARKER_ONLY op: its PASS marker in SQL or a
                            SELFTEST_SQL), failing rows go to SQL.retry with SCOPE (in: only a frozen/stack10 3/3 failure excuses)
"""
from pathlib import Path
import re
import sys

INSERT = 'INSERT INTO test_backend_ops'
CHANGED_QUANTS = ('q4_K', 'q5_K', 'q6_K', 'iq4_xs', 'q8_0')
UNQUANTIZED = ('f32', 'f16', 'bf16')
MAX_EXCUSED = 24
# Marker lines printed by the byte-identity cases of each component (tests/test-backend-ops.cpp).
MARKERS = {
    'base': 'ADD_SOFTPLUS_MUL: same-backend byte identity and changing-input replay PASS',
    'k7': 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS',
    'k4': 'SSM_CONV_RS: byte identity with the gathered path and changing-id replay PASS',
    'k6a': 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS',
    'k6b': 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS',
}
# Ops that are only a backend self-test, no test case: their PASS marker (cases >= 1) stands for a passing row.
MARKER_ONLY = {'RMS_NORM_SCALE_PAIR': MARKERS['k7']}
# Ops each selected component adds or changes cases for: at least one supported passing row must exist.
OPS = {'base': ('FLASH_ATTN_EXT', 'GATED_DELTA_NET', 'CONCAT'), 'p55': ('MUL_MAT', 'MUL_MAT_ID'), 'k4': ('SSM_CONV_RS',),
       'k10': ('TOP_K',), 'k8': ('MUL_MAT', 'MUL_MAT_ID')}


def parse_rows(raw):
    """A failing case prints `[OP] ERR = x > y ` on the same line before its INSERT, so the INSERT is searched for, not anchored."""
    for line in raw.splitlines():
        at = line.find(INSERT)
        if at < 0:
            continue
        body = line[at:].split(' VALUES (', 1)[1].rsplit(');', 1)[0]
        row = [v.replace("''", "'") for v in re.findall(r"'((?:[^']|'')*)'", body)]
        if len(row) != 16:
            raise ValueError('SQL field count changed')
        yield row


def quant_n(params):
    quant = re.search(r'type_a=(\w+)|mmvq-general:(\w+):', params)
    n = re.search(r'\bn=(\d+)|:n(\d+):', params)
    if quant is None or n is None:
        return None
    return quant.group(1) or quant.group(2), int(n.group(1) or n.group(2))


def in_scope(op, params, comps):
    if op not in ('MUL_MAT', 'MUL_MAT_ID'):
        return False
    qn = quant_n(params)
    if qn is None:
        return bool({'p55', 'k8'} & comps)
    quant, n = qn
    if 'p55' in comps and (quant in CHANGED_QUANTS or 5 <= n <= 8):
        return True
    return 'k8' in comps and quant not in UNQUANTIZED and n > 8


def check(raw, backend, commit, comps):
    """Returns (supported row count, retry list of (scope, op, params))."""
    rows, retry = [], []
    for row in parse_rows(raw):
        if row[2] != backend or row[5] != 'test' or not int(row[6]):
            continue
        if row[1] != commit[:9]:
            raise ValueError(f'wrong commit {row[1]} in a {backend} row')
        rows.append(row)
        if not (int(row[7]) and not row[8]):
            retry.append(('in' if in_scope(row[3], row[4], comps) else 'out', row[3], row[4]))
    if not rows:
        raise ValueError('no supported rows')
    fails = [l.strip() for l in raw.splitlines() if re.match(r'\s*FAIL ', l)]
    if fails:
        raise ValueError(f'{len(fails)} FAIL lines, first: {fails[0]}')
    for comp in sorted(comps):
        if comp in MARKERS and MARKERS[comp] not in raw:
            raise ValueError(f'{comp}: marker line missing: {MARKERS[comp]}')
        for op in OPS.get(comp, ()):
            if not any(r[3] == op and int(r[7]) and not r[8] for r in rows):
                raise ValueError(f'{comp}: no passing {op} row')
    if len(retry) > MAX_EXCUSED:
        raise ValueError(f'{len(retry)} failing rows (limit {MAX_EXCUSED})')
    if 'p55' in comps:
        mm = [r for r in rows if r[3] == 'MUL_MAT']
        for quant in CHANGED_QUANTS:
            for columns in (5, 6, 7, 8):
                if not any(re.search(r'\b' + quant + r'\b', r[4]) and re.search(rf'(?:\bn={columns}\b|:n{columns}:)', r[4]) for r in mm):
                    raise ValueError(f'p55: missing quant/column coverage: {quant} n={columns}')
    return len(rows), retry


def ops_check(raw, backend, commit, scope, ops, selftests=''):
    """An op-filtered run: rows of other ops are an error (the filter is exact), every op of OPS passes somewhere. SELFTESTS: the
    output of runs that select no case, for the markers a failing case would skip."""
    rows, retry = [], []
    for row in parse_rows(raw):
        if row[2] != backend or row[5] != 'test' or not int(row[6]):
            continue
        if row[1] != commit[:9]:
            raise ValueError(f'wrong commit {row[1]} in a {backend} row')
        if row[3] not in ops:
            raise ValueError(f'row of unselected op {row[3]}')
        rows.append(row)
        if not (int(row[7]) and not row[8]):
            retry.append((scope, row[3], row[4]))
    text = raw + '\n' + selftests
    fails = [l.strip() for l in text.splitlines() if re.match(r'\s*FAIL ', l)]
    if fails:
        raise ValueError(f'{len(fails)} FAIL lines, first: {fails[0]}')
    for op in ops:
        if op in MARKER_ONLY:
            if not re.search(re.escape(MARKER_ONLY[op]) + r' cases=[1-9]', text):
                raise ValueError(f'{op}: PASS marker missing: {MARKER_ONLY[op]}')
        elif not any(r[3] == op and int(r[7]) and not r[8] for r in rows):
            raise ValueError(f'no passing {op} row')
    if len(retry) > MAX_EXCUSED:
        raise ValueError(f'{len(retry)} failing rows (limit {MAX_EXCUSED})')
    return len(rows), retry


def case_regex(params):
    return '^' + re.sub(r'([\\^$.|?*+()\[\]{}])', r'\\\1', params) + '$'


def rerun_state(raw, op, backend, commit, params):
    """0: the case ran on this commit and backend and every row passed; 2: it ran and a row failed; 1: it did not run."""
    rows = [r for r in parse_rows(raw) if (r[3], r[2], r[4]) == (op, backend, params) and r[5] == 'test' and r[6] == '1']
    if not rows:
        return 1
    if any(r[1] != commit[:9] for r in rows):
        return 1
    return 0 if all(r[7] == '1' and not r[8] for r in rows) else 2


def row(commit, backend, op, params, passed=True, supported=True):
    err = '' if passed else 'test failed'
    return (f"INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, "
            f"error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES "
            f"('t', '{commit[:9]}', '{backend}', '{op}', '{params}', 'test', '{int(supported)}', '{int(passed)}', '{err}', '0', '0', '0', '0', '0', '', '');")


def fixture_checks():
    c, b = '0123456789abc', 'CUDA0'
    mm = lambda q, n, passed=True: row(c, b, 'MUL_MAT', f'type_a={q},type_b=f32,m=64,n={n},k=256,bs=[1,1]', passed)
    good = [mm(q, n) for q in CHANGED_QUANTS for n in (1, 5, 6, 7, 8, 12)] + [mm('q4_0', n) for n in (1, 5, 12)] + [mm('f16', 3)]
    good += [row(c, b, 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=4,n_used=2,b=0,m=64,n=12,k=256'),
             row(c, b, 'SSM_CONV_RS', 'x'), row(c, b, 'TOP_K', 'x'), row(c, b, 'FLASH_ATTN_EXT', 'x'),
             row(c, b, 'GATED_DELTA_NET', 'x'), row(c, b, 'CONCAT', 'x'),
             row(c, b, 'ADD_SOFTPLUS_MUL', 'x'), row(c, 'CUDA1', 'MUL_MAT', 'type_a=q4_K,type_b=f32,m=64,n=5,k=256', False)]
    markers = '\n'.join('  ' + m + ' cases=3' for m in MARKERS.values())
    raw = '\n'.join(good) + '\n' + markers + '\n'
    comps = {'base', 'p55', 'k4', 'k6a', 'k6b', 'k7', 'k8', 'k10'}
    n, retry = check(raw, b, c, comps)
    assert n == len(good) - 1 and retry == [], (n, retry)
    # a failing case is listed with the right scope, including the ERR-prefixed form the real binary prints
    err = '[MUL_MAT] ERR = 0.000719441 > 0.000500000 '
    for q, n_, scope in (('q4_K', 5, 'in'), ('q4_0', 5, 'in'), ('q4_0', 1, 'out'), ('f16', 3, 'out'), ('q4_0', 12, 'in')):
        text = raw + err + mm(q, n_, False) + '\n'
        _, retry = check(text, b, c, comps)
        assert retry == [(scope, 'MUL_MAT', f'type_a={q},type_b=f32,m=64,n={n_},k=256,bs=[1,1]')], (q, n_, retry)
    assert in_scope('MUL_MAT', 'type_a=q4_0,n=12', {'p55'}) is False and in_scope('MUL_MAT', 'type_a=q4_0,n=12', {'k8'}) is True
    # hard failures
    base = mm('q4_0', 1, False)
    for name, text, kw in (
        ('no rows', '', {}),
        ('wrong commit', raw.replace(c[:9], 'ffffffff0', 1), {}),
        ('FAIL line', raw + '  FAIL RMS_NORM_SCALE_PAIR byte identity: ncols=1\n', {}),
        ('missing marker', raw.replace(MARKERS['k7'], 'x'), {}),
        ('missing op', '\n'.join(l for l in good if 'SSM_CONV_RS' not in l) + '\n' + markers + '\n', {}),
        ('missing GDN op', '\n'.join(l for l in good if 'GATED_DELTA_NET' not in l) + '\n' + markers + '\n', {}),
        ('missing coverage', '\n'.join(l for l in good if 'type_a=iq4_xs' not in l or 'n=7' not in l) + '\n' + markers + '\n', {}),
        ('too many failures', raw + '\n'.join(mm('q4_0', 1, False).replace('m=64', f'm={64 + i * 8}') for i in range(MAX_EXCUSED + 1)) + '\n', {}),
    ):
        try:
            check(text, b, c, comps)
        except ValueError:
            continue
        raise AssertionError(f'negative control escaped: {name}')
    # unselected components impose nothing: no SSM_CONV_RS row needed without k4, no p55 coverage without p55
    check('\n'.join(l for l in good if 'SSM_CONV_RS' not in l and 'iq4_xs' not in l) + '\n' + markers.replace(MARKERS['k4'], 'x') + '\n', b, c, {'base', 'k7'})
    # rerun gate
    case = f'type_a=q4_0,type_b=f32,m=64,n=1,k=256,bs=[1,1]'
    ok, bad = mm('q4_0', 1), err + mm('q4_0', 1, False)
    assert rerun_state(ok, 'MUL_MAT', b, c, case) == 0 and rerun_state(bad, 'MUL_MAT', b, c, case) == 2
    assert rerun_state('', 'MUL_MAT', b, c, case) == 1 and rerun_state(ok.replace('n=1', 'n=2'), 'MUL_MAT', b, c, case) == 1
    assert rerun_state(ok, 'MUL_MAT', b, 'f' * 13, case) == 1 and rerun_state(ok, 'MUL_MAT', 'CUDA1', c, case) == 1
    assert case_regex('a=[1,1],b=x.y') == r'^a=\[1,1\],b=x\.y$'
    # op-filtered runs
    fa = [row(c, b, 'FLASH_ATTN_EXT', f'hsk=256,kv={k}') for k in (512, 1024)] + [row(c, b, 'TOP_K', 'x')]
    n, retry = ops_check('\n'.join(fa), b, c, 'in', ['FLASH_ATTN_EXT', 'TOP_K'])
    assert n == 3 and retry == []
    _, retry = ops_check('\n'.join(fa) + '\n' + row(c, b, 'TOP_K', 'y', False), b, c, 'out', ['FLASH_ATTN_EXT', 'TOP_K'])
    assert retry == [('out', 'TOP_K', 'y')], retry
    for name, text, ops in (('missing op', '\n'.join(fa), ['FLASH_ATTN_EXT', 'TOP_K', 'CONCAT']),
                            ('unselected op', '\n'.join(fa), ['FLASH_ATTN_EXT']),
                            ('wrong commit', '\n'.join(fa).replace(c[:9], 'ffffffff0', 1), ['FLASH_ATTN_EXT', 'TOP_K']),
                            ('FAIL line', '\n'.join(fa) + '\n  FAIL ADD_RMS_NORM byte identity\n', ['FLASH_ATTN_EXT', 'TOP_K']),
                            ('only failing rows', row(c, b, 'TOP_K', 'x', False), ['TOP_K'])):
        try:
            ops_check(text, b, c, 'in', ops)
        except ValueError:
            continue
        raise AssertionError(f'ops-check negative control escaped: {name}')
    # a marker-only op passes on its marker, from the run itself or from a no-case self-test run, and fails without it
    k7 = '  ' + MARKERS['k7'] + ' cases=360\n'
    ops = ['FLASH_ATTN_EXT', 'TOP_K', 'RMS_NORM_SCALE_PAIR']
    assert ops_check('\n'.join(fa) + '\n' + k7, b, c, 'out', ops) == (3, [])
    assert ops_check('\n'.join(fa), b, c, 'out', ops, k7) == (3, [])
    for name, text, selftests in (('marker missing', '\n'.join(fa), ''),
                                  ('zero cases', '\n'.join(fa), k7.replace('cases=360', 'cases=0')),
                                  ('FAIL marker', '\n'.join(fa), k7 + '  FAIL RMS_NORM_SCALE_PAIR matcher: layout=1 expected pair\n'),
                                  ('FAIL marker in the run', '\n'.join(fa) + '\n  FAIL RMS_NORM_SCALE_PAIR negative control\n', k7)):
        try:
            ops_check(text, b, c, 'out', ops, selftests)
        except ValueError:
            continue
        raise AssertionError(f'ops-check marker control escaped: {name}')
    print('full-suite checks and negative controls PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixtures']:
        fixture_checks()
    elif sys.argv[1:2] == ['--regex']:
        print(case_regex(sys.argv[2]))
    elif sys.argv[1:2] == ['--rerun']:
        sql, op, backend, commit, params = sys.argv[2:]
        sys.exit(rerun_state(Path(sql).read_text(), op, backend, commit, params))
    elif sys.argv[1:2] == ['check']:
        sql, backend, commit, comps = sys.argv[2:6]
        selftests = ''.join(Path(f).read_text() for f in sys.argv[6:])
        n, retry = check(Path(sql).read_text() + selftests, backend, commit, set(comps.split(',')))
        Path(sql + '.retry').write_text(''.join(f'{s}\t{o}\t{p}\n' for s, o, p in retry))
        print(f'full_rows_M={n}')
        print(f'failing_rows_to_rerun_M={len(retry)}')
    elif sys.argv[1:2] == ['ops-check'] and len(sys.argv) >= 7:
        sql, backend, commit, scope, ops = sys.argv[2:7]
        assert scope in ('in', 'out')
        selftests = ''.join(Path(f).read_text() for f in sys.argv[7:])
        n, retry = ops_check(Path(sql).read_text(), backend, commit, scope, ops.split(','), selftests)
        Path(sql + '.retry').write_text(''.join(f'{s}\t{o}\t{p}\n' for s, o, p in retry))
        print(f'op_rows_M={n}')
        print(f'failing_rows_to_rerun_M={len(retry)}')
    else:
        raise SystemExit(__doc__)
