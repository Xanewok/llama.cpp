"""restrict_summary.py OUT: perf-<dev>-<arm>-<i>-<set>.out files of bw-restrict-probe.sh -> perf.tsv (stdout), verdict + served impact (stderr/served.tsv).

Arms: stock (GGML_CUDA_RESTRICT empty on Hopper+ with PDL, as upstream) and forced (-DGGML_CUDA_FORCE_RESTRICT, __restrict__ kept,
PDL on). Per (dev, op, params): stock median over >= 3 runs, forced median over >= 2 runs, A/A spread = max - min of the stock runs.
cost (brief, 2026-10-09): both forced repeats exceed stock median + A/A spread. Family verdicts (fixed before the run): "cost" when
>= 50 % of the family's rows are flagged and the median delta is >= MATERIAL percent, "none" when <= 15 % are flagged, else "mixed".
Null control: MUL_MAT rows that cc 12.0 routes to MMQ (q4_K n>5, q5_K n>6, q6_K n>7) run kernels #24030 did not touch; if more than
25 % of them are flagged the A/A noise is too high and the verdict is void (rc 5).
"""
import math
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r'^perf-(CUDA\d)-(stock|forced)-\d+-(\w+)\.out$')
MMQ_MAX = {'q4_K': 5, 'q5_K': 6, 'q6_K': 7}
MMVQ_MAX = 8
MATERIAL = 0.5  # percent of the stock median
MM = re.compile(r'type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),')
FA = re.compile(r'kv=(\d+),nb=(\d+),')


def rows(path):
    content = path.read_text()
    statements = re.findall(r'CREATE TABLE IF NOT EXISTS test_backend_ops \([\s\S]*?\);|INSERT INTO test_backend_ops [^\n]*;', content)
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.executescript('\n'.join(statements))
    out = [dict(r) for r in db.execute('SELECT * FROM test_backend_ops')]
    db.close()
    for r in out:
        assert r['supported'] and r['passed'], (path, r['op_params'])
        assert math.isfinite(r['time_us']) and r['time_us'] > 0, (path, r['op_params'])
    return out


def classify(op, params):
    """-> (family, short key, routed to MMQ)"""
    if op == 'MUL_MAT':
        t, m, n, k = MM.search(params).groups()
        mmq = t in MMQ_MAX and int(n) > MMQ_MAX[t]
        return ('mmq-null' if mmq else 'mmvq'), f'{t} m={m} k={k} n={n}', mmq
    if op == 'FLASH_ATTN_EXT':
        kv, nb = FA.search(params).groups()
        return 'fa', f'kv={kv} nb={nb}', False
    return op.lower(), params, False


def load(out):
    cells = defaultdict(lambda: defaultdict(list))
    for path in sorted(out.glob('perf-*.out')):
        m = NAME.match(path.name)
        if not m:
            continue
        dev, arm, _ = m.groups()
        per_run = defaultdict(list)
        for r in rows(path):
            per_run[(r['op_name'], r['op_params'])].append(r['time_us'])
        for key, v in per_run.items():
            cells[(dev, *key)][arm].append(statistics.median(v))
    assert cells, 'no perf rows'
    return cells


def main():
    out = Path(sys.argv[1])
    cells = load(out)
    table, fam = [], defaultdict(list)
    print('dev\tfamily\top\tshape\tstock_us\tforced_us\tdelta_us\tdelta_pct\taa_spread_us\tf1_minus_stock\tf2_minus_stock\tcost')
    for key in sorted(cells, key=lambda k: (k[0], k[1], classify(k[1], k[2])[1])):
        dev, op, params = key
        s, f = cells[key]['stock'], cells[key]['forced']
        assert len(s) >= 3 and len(f) >= 2, (key, len(s), len(f))
        family, short, _ = classify(op, params)
        sm, fm = statistics.median(s), statistics.median(f)
        spread = max(s) - min(s)
        d = [x - sm for x in f]
        cost = all(x > spread for x in d)
        row = dict(dev=dev, family=family, op=op, short=short, stock=sm, forced=fm, delta=fm - sm, pct=100 * (fm - sm) / sm,
                   spread=spread, d1=d[0], d2=d[1], cost=cost)
        table.append(row)
        fam[(dev, family)].append(row)
        print(f"{dev}\t{family}\t{op}\t{short}\t{sm:.3f}\t{fm:.3f}\t{fm - sm:+.3f}\t{row['pct']:+.2f}\t{spread:.3f}\t{d[0]:+.3f}\t{d[1]:+.3f}\t{int(cost)}")
    void = False
    groups = defaultdict(list)
    for (dev, family), rs in fam.items():
        groups[(dev, family)] += rs
        if family == 'mmvq':
            for r in rs:
                groups[(dev, 'mmvq/' + r['short'].split()[0])].append(r)
    for (dev, family), rs in sorted(groups.items()):
        n, flagged = len(rs), sum(r['cost'] for r in rs)
        med = statistics.median(r['pct'] for r in rs)
        if flagged >= 0.5 * n and med >= MATERIAL:
            verdict = 'cost'
        elif flagged >= 0.5 * n:
            verdict = f'flagged but median < {MATERIAL} %'
        else:
            verdict = 'none' if flagged <= 0.15 * n else 'mixed'
        if family == 'mmq-null' and flagged > 0.25 * n:
            void = True
            verdict = 'VOID (null control noisy)'
        top = max(rs, key=lambda r: r['pct'])
        print(f"verdict {dev} {family}: rows={n} flagged={flagged} median_delta_pct={med:+.2f} max_delta_pct={top['pct']:+.2f} "
              f"({top['short']}) -> {verdict}", file=sys.stderr)
    served(out, table)
    return 5 if void else 0


# Served calls per verify round and GPU (D from the model metadata in the report: 64 layers = 16 attention + 48 GDN, plus the MTP
# layer for 5 draft steps; type mix is G, so MUL_MAT is totalled per pure type).
MM_NP1 = {(8704, 5120, 6): 128, (5120, 8704, 6): 64, (6144, 5120, 6): 16, (3072, 5120, 6): 48, (5120, 5120, 6): 48,
          (5120, 3072, 6): 64, (8704, 5120, 1): 10, (5120, 8704, 1): 5, (6144, 5120, 1): 5, (5120, 3072, 1): 5}
# np2: 12-column target GEMMs run on MMQ (untouched); the GDN out projection reads its weights twice at n=6 (3D input, 10-06 note),
# draft steps run per slot at n=1.
MM_NP2 = {(5120, 3072, 6): 96, (8704, 5120, 1): 20, (5120, 8704, 1): 10, (6144, 5120, 1): 10, (5120, 3072, 1): 10}
OTHER_NP1 = {('fa', 'nb=6'): 16, ('fa', 'nb=1'): 5, ('ssm_conv', 'ne_a=[9,5120,1,1]'): 48, ('gated_delta_net', 'n_seq_tokens=6,n_seqs=1'): 48,
             ('get_rows', 'r=1'): 48, ('get_rows', 'm=6'): 1, ('set_rows', 'r=6,'): 32, ('set_rows', 'r=1,'): 10}
OTHER_NP2 = {('fa', 'nb=6'): 32, ('fa', 'nb=1'): 10, ('ssm_conv', 'ne_a=[9,5120,2,1]'): 48, ('gated_delta_net', 'n_seq_tokens=6,n_seqs=2'): 48,
             ('get_rows', 'r=2'): 48, ('get_rows', 'm=6'): 1, ('set_rows', 'r=12,'): 32, ('set_rows', 'r=1,'): 20}
HEAD_NP1 = 1


def one(rows, family, key, contains=False):
    r = [x for x in rows if x['family'] == family and (key in x['short'] if contains else x['short'] == key)]
    assert len(r) == 1, (family, key, len(r))
    return r[0]


def add(profile, component, label, calls, r, lines, totals):
    lines.append(f"{profile}\t{label}\t{calls}\t{r['delta']:+.3f}\t{calls * r['delta'] / 1000:+.4f}\t{int(r['cost'])}")
    totals[(profile, component)][0] += calls * r['delta'] / 1000
    totals[(profile, component)][1] += calls * r['delta'] / 1000 * r['cost']


def served(out, table):
    lines = ['profile\tcomponent\tcalls_per_round\tdelta_us_per_call\tms_per_round\tcost_flagged']
    totals = defaultdict(lambda: [0.0, 0.0])
    cuda0 = [r for r in table if r['dev'] == 'CUDA0']
    for profile, mm, other, head in (('np1_k5', MM_NP1, OTHER_NP1, HEAD_NP1), ('np2_k5', MM_NP2, OTHER_NP2, 0)):
        for ty in ('iq4_xs', 'q4_K', 'q5_K', 'q6_K', 'q8_0'):
            for (m, k, n), calls in mm.items():
                r = [x for x in cuda0 if x['op'] == 'MUL_MAT' and x['short'] == f'{ty} m={m} k={k} n={n}']
                assert len(r) == 1, (ty, m, k, n)
                add(profile, f'MUL_MAT {ty}', f'MUL_MAT {ty} {m}x{k} n={n}', calls, r[0], lines, totals)
        if head:
            add(profile, 'head', 'MUL_MAT q6_K head n=6', 1, one(cuda0, 'mmvq', 'q6_K m=124160 k=5120 n=6'), lines, totals)
        for (family, needle), calls in other.items():
            if family == 'fa':
                for kv in (8192, 32768, 90112):
                    add(profile, f'FA kv={kv}', f'FA kv={kv} {needle}', calls, one(cuda0, family, f'kv={kv} {needle}'), lines, totals)
                continue
            add(profile, family, f'{family} {needle}', calls, one(cuda0, family, needle, contains=True), lines, totals)
    (out / 'served.tsv').write_text('\n'.join(lines) + '\n')
    for (profile, comp), (all_d, flagged_d) in sorted(totals.items()):
        print(f'served {profile} {comp}: sum_all {all_d:+.4f} ms/round, cost-flagged rows only {flagged_d:+.4f} ms/round', file=sys.stderr)


sys.exit(main())
