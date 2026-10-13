"""Judge the FA q8_0 timing A/B of the stack11 build vs frozen/stack10 (fa-identity DEVICE perf, one GPU, arms alternated).

Contract: every run of both arms timed the same full case list; per case the median of each arm's runs (at least MIN_REPS) is
compared on the round, not the kernel: (stack11 - stack10) x FA_LAYERS <= ROUND_US, nb=1 included (the decode control: the
stack's decode kernel, not the verify one). One timed op is one full-attention layer's FA; a round runs FA_LAYERS of them, so the
bound keeps the per-round cost under the A/B neutral band (0.15 ms/round). The ratio is reported, not judged: master abee0c847's FA
rewrite costs np1 verify +3.5..+4.2 % at 8k (a3, M) = +2.5 us/op, ~0.04 ms/round (owner deferred the call, 10-13).
Noise is (max - min) / median of an arm's runs, reported per case; it does not void the verdict.
usage: fa_q8_perf.py judge OUT.tsv RUN...    RUN = .../id-DEV-perf-<new|s10>-r<rep>.txt; rc 0 pass, 1 over the band (the table
                                            is written either way), 5 too little or inconsistent data
       fa_q8_perf.py selftest
"""
import re
import statistics
import sys
import tempfile
from pathlib import Path

FA_LAYERS = 16  # Swift 1.5: 64 layers, full_attention_interval 4
ROUND_US = 100.0
MIN_REPS = 3
N_CASES = 20
ARMS = ('new', 's10')
RUN = re.compile(r'id-(\w+)-perf-(new|s10)-r(\d+)\.txt$')


class NoData(Exception):
    pass


def parse(paths):
    runs = {a: [] for a in ARMS}
    for p in paths:
        m = RUN.search(str(p))
        if not m:
            raise NoData(f'not a perf run: {p}')
        table = {}
        for line in Path(p).read_text().splitlines():
            f = line.split()
            if len(f) != 3 or f[0] != 'perf' or not re.fullmatch(r'[0-9.]+', f[2]):
                raise NoData(f'{p}: bad line: {line}')
            table[f[1]] = float(f[2])
        if len(table) != N_CASES:
            raise NoData(f'{p}: {len(table)} cases, expected {N_CASES}')
        runs[m.group(2)].append(table)
    for a in ARMS:
        if len(runs[a]) < MIN_REPS:
            raise NoData(f'arm {a}: {len(runs[a])} runs, need {MIN_REPS}')
    cases = set(runs['new'][0])
    if any(set(t) != cases for a in ARMS for t in runs[a]):
        raise NoData('runs disagree on the case list')
    return runs


def table(runs):
    rows = []
    for case in sorted(runs['new'][0], key=lambda c: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', c)]):
        v = {a: [t[case] for t in runs[a]] for a in ARMS}
        med = {a: statistics.median(v[a]) for a in ARMS}
        noise = {a: (max(v[a]) - min(v[a]))/med[a] for a in ARMS}
        ratio = med['new']/med['s10']
        kind = 'decode-control' if ',nb=1,' in case else 'verify'
        rows.append(dict(case=case, kind=kind, n_new=len(v['new']), n_s10=len(v['s10']), new_med_us=med['new'], s10_med_us=med['s10'],
                         ratio=ratio, noise_new=noise['new'], noise_s10=noise['s10'], verdict='PASS' if (med['new'] - med['s10'])*FA_LAYERS <= ROUND_US else 'OVER'))
    return rows


def write(rows, out):
    cols = list(rows[0])
    fmt = lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)
    Path(out).write_text('\t'.join(cols) + '\n' + ''.join('\t'.join(fmt(r[c]) for c in cols) + '\n' for r in rows))


def judge(out, paths):
    try:
        rows = table(parse(paths))
    except NoData as e:
        print(f'fa q8 perf: NO DATA: {e}')
        return 5
    write(rows, out)
    for r in rows:
        print(f'{r["verdict"]} {r["kind"]:14} {r["case"]} new {r["new_med_us"]:.2f} s10 {r["s10_med_us"]:.2f} us ratio {r["ratio"]:.4f} '
              f'noise new {r["noise_new"]:.3f} s10 {r["noise_s10"]:.3f}')
    over = [r for r in rows if r['verdict'] != 'PASS']
    print(f'fa q8 perf: {len(rows)} shapes, ratio max {max(r["ratio"] for r in rows):.4f}, bound {ROUND_US:.0f} us/round over {FA_LAYERS} layers, run-to-run noise max '
          f'{max(max(r["noise_new"], r["noise_s10"]) for r in rows):.3f}, over the band: {len(over)} -> {"FAIL" if over else "PASS"}')
    return 1 if over else 0


def selftest():
    cases = [f'hsk=256,nr=6x{s},kv={kv},nb={nb},type=q8_0' for s in (1, 2) for kv in (8192, 32768) for nb in (1, 5, 6, 7, 8)]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def runs(slow=None, reps=3, drop=None):
            paths = []
            for arm in ARMS:
                for rep in range(1, reps + 1):
                    p = d/f'id-CUDA0-perf-{arm}-r{rep}.txt'
                    lines = [f'perf {c} {100.0*(1 + 0.01*rep)*(slow if arm == "new" and c == cases[3] and slow else 1):.3f}'
                             for c in cases if c != drop]
                    p.write_text('\n'.join(lines) + '\n')
                    paths.append(p)
            return paths

        assert judge(d/'t.tsv', runs()) == 0
        assert len((d/'t.tsv').read_text().splitlines()) == N_CASES + 1
        assert judge(d/'t.tsv', runs(slow=1.06)) == 0  # 100 us/op: +6 us x 16 = 96 us/round
        assert judge(d/'t.tsv', runs(slow=1.07)) == 1 and 'OVER' in (d/'t.tsv').read_text()
        for p in d.glob('id-*'):
            p.unlink()
        assert judge(d/'t.tsv', runs(reps=2)) == 5
        assert judge(d/'t.tsv', runs(drop=cases[0])) == 5
        bad = runs()
        bad[0].write_text(re.sub(r' [0-9.]+\n', ' UNSUPPORTED\n', bad[0].read_text(), count=1))
        assert judge(d/'t.tsv', bad) == 5
    print('fa q8 perf selftest PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            selftest()
        print('fa q8 perf selftest PASS')
    elif sys.argv[1:2] == ['judge'] and len(sys.argv) > 3:
        sys.exit(judge(sys.argv[2], sys.argv[3:]))
    else:
        raise SystemExit(__doc__)
