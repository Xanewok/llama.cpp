"""Reject empty, skipped, failed or incorrectly stamped CUDA correctness results."""
from pathlib import Path
import re
import sys

RECORDED = Path('/home/xanewok/repos/mmvq-w2r4/codex/runs/pertype2/canary-a1/00-gpu0-patch-aligned.sql')


def check(raw, op, backend, commit):
    rows = []
    for line in raw.splitlines():
        if line.startswith('INSERT INTO test_backend_ops'):
            body = line.split(' VALUES (', 1)[1].rsplit(');', 1)[0]
            row = [v.replace("''", "'") for v in re.findall(r"'((?:[^']|'')*)'", body)]
            if len(row) != 16:
                raise ValueError('SQL field count changed')
            if row[3] == op and row[2] == backend and row[5] == 'test' and int(row[6]):
                if row[1] != commit[:9] or not int(row[7]) or row[8]:
                    raise ValueError('CUDA correctness or commit gate failed')
                rows.append(row)
    if not rows:
        raise ValueError('no supported CUDA correctness rows')
    if op == 'MUL_MAT':
        for quant in ('q4_K', 'q5_K', 'q6_K', 'iq4_xs', 'q8_0'):
            for columns in (5, 6, 7, 8):
                if not any(re.search(r'\b' + quant + r'\b', r[4]) and re.search(rf'(?:\bn={columns}\b|:n{columns}:)', r[4]) for r in rows):
                    raise ValueError(f'missing quant/column coverage: {quant} n={columns}')
    return len(rows)


def fixture_checks():
    raw = RECORDED.read_text()
    assert check(raw, 'MUL_MAT', 'CUDA0', '26a2a2758') > 0
    for mutant, op, backend, commit in (
        ('', 'MUL_MAT', 'CUDA0', '26a2a2758'),
        (raw, 'MUL_MAT_ID', 'CUDA0', '26a2a2758'),
        (raw, 'MUL_MAT', 'CUDA1', '26a2a2758'),
        (raw, 'MUL_MAT', 'CUDA0', '000000000'),
        (raw.replace("'test', '1', '1'", "'test', '1', '0'"), 'MUL_MAT', 'CUDA0', '26a2a2758'),
        ('\n'.join(l for l in raw.splitlines() if 'iq4_xs' not in l), 'MUL_MAT', 'CUDA0', '26a2a2758'),
    ):
        try:
            check(mutant, op, backend, commit)
        except ValueError:
            continue
        raise AssertionError('correctness negative control escaped')
    print('recorded CUDA correctness SQL and negative controls PASS')


if __name__ == '__main__':
    print(f'correctness_rows_M={check(Path(sys.argv[1]).read_text(), *sys.argv[2:])}')
