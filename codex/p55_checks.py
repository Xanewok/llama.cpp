"""Reject empty, skipped, failed or incorrectly stamped CUDA correctness results."""
from pathlib import Path
import re
import sys

RECORDED = Path('/home/xanewok/repos/mmvq-w2r4/codex/runs/pertype2/canary-a1/00-gpu0-patch-aligned.sql')


CHANGED_QUANTS = ('q4_K', 'q5_K', 'q6_K', 'iq4_xs', 'q8_0')
INSERT = 'INSERT INTO test_backend_ops'
# test-backend-ops draws its inputs from std::random_device, so a near-threshold NMSE case outside the changed launch
# configs can fail once by chance. Contract: a failure there is excused only if the same case passes 3 fresh reruns (bw-stack9-p55.sh).
MAX_EXCUSED = 8


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


def in_scope(params):
    """Cases whose MMVQ launch config package 55 changes or may change: the changed quants at any n, any quant at n=5..8.
    Reads both the stock `type_a=..,n=..` form and the `mmvq-general:<quant>:n<N>:` named form of the recorded fixture."""
    quant = re.search(r'type_a=(\w+)|mmvq-general:(\w+):', params)
    n = re.search(r'\bn=(\d+)|:n(\d+):', params)
    if quant is None or n is None:
        return True
    return (quant.group(1) or quant.group(2)) in CHANGED_QUANTS or 5 <= int(n.group(1) or n.group(2)) <= 8


def check(raw, op, backend, commit):
    """Returns (supported row count, out-of-scope failing op_params that need the rerun gate)."""
    rows, excused = [], []
    for row in parse_rows(raw):
        if row[3] == op and row[2] == backend and row[5] == 'test' and int(row[6]):
            if row[1] != commit[:9]:
                raise ValueError('CUDA correctness or commit gate failed: wrong commit')
            if int(row[7]) and not row[8]:
                rows.append(row)
            elif in_scope(row[4]) or int(row[7]):
                raise ValueError(f'CUDA correctness gate failed: {row[4]}')
            else:
                rows.append(row)
                excused.append(row[4])
    if not rows:
        raise ValueError('no supported CUDA correctness rows')
    if len(excused) > MAX_EXCUSED:
        raise ValueError(f'{len(excused)} failing cases outside the changed configs')
    if op == 'MUL_MAT':
        for quant in CHANGED_QUANTS:
            for columns in (5, 6, 7, 8):
                if not any(re.search(r'\b' + quant + r'\b', r[4]) and re.search(rf'(?:\bn={columns}\b|:n{columns}:)', r[4]) for r in rows):
                    raise ValueError(f'missing quant/column coverage: {quant} n={columns}')
    return len(rows), excused


def case_regex(params):
    return '^' + re.sub(r'([\\^$.|?*+()\[\]{}])', r'\\\1', params) + '$'


def passed_once(raw, op, backend, commit, params):
    """True only when the rerun produced the case, on this commit and backend, with every row passing."""
    rows = [r for r in parse_rows(raw) if (r[3], r[2], r[4]) == (op, backend, params)]
    return bool(rows) and all(r[1] == commit[:9] and r[5] == 'test' and r[6] == '1' and r[7] == '1' and not r[8] for r in rows)


def fixture_checks():
    raw = RECORDED.read_text()
    assert check(raw, 'MUL_MAT', 'CUDA0', '26a2a2758')[0] > 0
    fail_row = next(l for l in raw.splitlines() if l.startswith(INSERT) and 'mmvq-general:iq4_xs:n5:' in l)
    err = '[MUL_MAT] ERR = 0.000719441 > 0.000500000 '
    failed = lambda l: err + l.replace("'test', '1', '1', ''", "'test', '1', '0', 'test failed'", 1)
    out_of_scope = lambda l, i=0: l.replace('iq4_xs', 'q5_1').replace(':n5:', ':n1:').replace(':m8:', f':m{8 + i}:')
    assert failed(fail_row) != err + fail_row, 'fixture row not mutated'
    assert in_scope(fail_row) and not in_scope(out_of_scope(fail_row))
    for mutant, op, backend, commit in (
        ('', 'MUL_MAT', 'CUDA0', '26a2a2758'),
        (raw, 'MUL_MAT_ID', 'CUDA0', '26a2a2758'),
        (raw, 'MUL_MAT', 'CUDA1', '26a2a2758'),
        (raw, 'MUL_MAT', 'CUDA0', '000000000'),
        (raw.replace("'test', '1', '1'", "'test', '1', '0'"), 'MUL_MAT', 'CUDA0', '26a2a2758'),
        ('\n'.join(l for l in raw.splitlines() if 'iq4_xs' not in l), 'MUL_MAT', 'CUDA0', '26a2a2758'),
        # failing changed-quant case in the ERR-prefixed form the real binary prints
        (raw.replace(fail_row, failed(fail_row)), 'MUL_MAT', 'CUDA0', '26a2a2758'),
        # more failing out-of-scope cases than the excuse budget
        (raw + '\n' + '\n'.join(failed(out_of_scope(fail_row, i)) for i in range(MAX_EXCUSED + 1)), 'MUL_MAT', 'CUDA0', '26a2a2758'),
    ):
        try:
            check(mutant, op, backend, commit)
        except ValueError:
            continue
        raise AssertionError('correctness negative control escaped')
    # an out-of-scope failure is excused (listed for rerun), an in-scope one never is
    oos = out_of_scope(fail_row)
    n_rows, excused = check(raw + '\n' + failed(oos), 'MUL_MAT', 'CUDA0', '26a2a2758')
    assert len(excused) == 1 and 'q5_1:n1:' in excused[0], excused
    # rerun gate: passes only on a clean row for the same case/commit/backend
    params = excused[0]
    ok_rerun = oos
    assert passed_once(ok_rerun, 'MUL_MAT', 'CUDA0', '26a2a2758', params)
    for bad in ('', failed(oos), ok_rerun.replace("'test', '1', '1'", "'test', '1', '0'"), ok_rerun.replace(':n1:', ':n2:')):
        assert not passed_once(bad, 'MUL_MAT', 'CUDA0', '26a2a2758', params), 'rerun negative control escaped'
    assert not passed_once(ok_rerun, 'MUL_MAT', 'CUDA0', '000000000', params)
    assert not passed_once(ok_rerun, 'MUL_MAT', 'CUDA1', '26a2a2758', params)
    assert case_regex('a=[1,1],b=x.y') == r'^a=\[1,1\],b=x\.y$'
    print('recorded CUDA correctness SQL and negative controls PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixtures']:
        fixture_checks()
    elif sys.argv[1:2] == ['--regex']:
        print(case_regex(sys.argv[2]))
    elif sys.argv[1:2] == ['--rerun-passed']:
        sql, op, backend, commit, params = sys.argv[2:]
        sys.exit(0 if passed_once(Path(sql).read_text(), op, backend, commit, params) else 1)
    else:
        sql, op, backend, commit = sys.argv[1:]
        n, excused = check(Path(sql).read_text(), op, backend, commit)
        Path(sql + '.retry').write_text(''.join(p + '\n' for p in excused))
        print(f'correctness_rows_M={n}')
        print(f'excused_rerun_cases_M={len(excused)}')
