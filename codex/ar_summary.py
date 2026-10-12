"""perf-*.out of the ar-readfirst window -> perf.tsv (per cell and batch: median of per-run medians, IQR, delta to base) plus
the gate. Adapted from wt/ar-duplex codex/ar_duplex_summary.py.

usage: ar_summary.py OUTDIR            print perf.tsv to stdout, exit 0
       ar_summary.py --gate PERF_TSV   exit 0 if the render gate passes, 1 if not, 6 if the table is incomplete
       ar_summary.py --selftest

Render gate (coordinator 10-12): arm rf vs base, both batches: median at least 10 % lower at rows6 and rows12, and no decode
cell (rows1/6/12/18) more than 2 % slower. Brief falsifier (reported, not gated): gain under 20 % at rows6 or rows12, or any
decode-cell regression. Mechanism lines: rf vs sf (same binary, only the store position differs) and rf vs one (one round).
"""
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r'^perf arm=(\S+) cell=(\S+) ne=(\d+) batch=(\d+) calls=(\d+) med_us=([\d.]+) p25_us=([\d.]+) p75_us=([\d.]+)$')
CELLS = ['rows1', 'rows6', 'rows12', 'rows18', 'rows64', 'mb0.5']
DECODE = ['rows1', 'rows6', 'rows12', 'rows18']
GAIN_CELLS = ['rows6', 'rows12']
ARMS = ['base', 'rf', 'sf', 'one', 'rv2']
MIN_RUNS = {'base': 3, 'rf': 3, 'sf': 2, 'one': 2, 'rv2': 2}
GATE_GAIN_PCT = -10.0
FALSIFIER_GAIN_PCT = -20.0
REGRESSION_PCT = 2.0


def table(out):
    runs = defaultdict(list)   # (arm, cell, batch) -> [(med, iqr, calls)]
    for f in sorted(out.glob('perf-*.out')):
        for line in f.read_text().splitlines():
            m = PAT.match(line)
            if m:
                arm, cell, _, batch, calls, med, p25, p75 = m.groups()
                runs[(arm, cell, int(batch))].append((float(med), float(p75) - float(p25), int(calls)))
    return runs


def render(runs):
    med = {k: st.median(r[0] for r in v) for k, v in runs.items()}
    arms = [a for a in ARMS if any(k[0] == a for k in runs)]
    lines = ['batch\tcell\tarm\truns\tcalls_per_run\tmed_us\tiqr_us\tdelta_us\tdelta_pct\tmin_run_med\tmax_run_med']
    for batch in (1, 8):
        for cell in CELLS:
            b = med.get(('base', cell, batch))
            for arm in arms:
                v = runs.get((arm, cell, batch))
                if not v:
                    continue
                m = med[(arm, cell, batch)]
                d = '' if b is None else f'{m - b:+.2f}'
                dp = '' if b is None else f'{100 * (m - b) / b:+.1f}'
                lines.append(f'{batch}\t{cell}\t{arm}\t{len(v)}\t{v[0][2]}\t{m:.2f}\t{st.median(r[1] for r in v):.2f}\t{d}\t{dp}\t'
                             f'{min(r[0] for r in v):.2f}\t{max(r[0] for r in v):.2f}')
    lines.append('')
    for other in ('sf', 'one'):
        for batch in (1, 8):
            parts = []
            for cell in ('rows6', 'rows12', 'rows18', 'rows64'):
                a, o = med.get(('rf', cell, batch)), med.get((other, cell, batch))
                if a and o:
                    parts.append(f'{cell}:{100 * (a - o) / o:+.1f}%')
            lines.append(f'mechanism\trf_vs_{other}\tbatch={batch}\t{",".join(parts) or "n/a"}')
    lines.append('')
    complete = all(len(runs.get((a, c, b), [])) >= MIN_RUNS[a] for a in ('base', 'rf') for c in DECODE for b in (1, 8))
    gate_fail, falsified = [], []
    for batch in (1, 8):
        for cell in DECODE:
            b, a = med.get(('base', cell, batch)), med.get(('rf', cell, batch))
            if not (b and a):
                continue
            pct = 100 * (a - b) / b
            if pct > REGRESSION_PCT:
                gate_fail.append(f'b{batch}:{cell}:{pct:+.1f}%>+{REGRESSION_PCT:g}%')
                falsified.append(f'b{batch}:{cell}:regression {pct:+.1f}%')
            if cell in GAIN_CELLS:
                if pct > GATE_GAIN_PCT:
                    gate_fail.append(f'b{batch}:{cell}:{pct:+.1f}%>{GATE_GAIN_PCT:g}%')
                if pct > FALSIFIER_GAIN_PCT:
                    falsified.append(f'b{batch}:{cell}:gain {pct:+.1f}% above {FALSIFIER_GAIN_PCT:g}%')
    verdict = 'INCOMPLETE' if not complete else ('PASS' if not gate_fail else 'FAIL')
    lines.append(f'gate\t{verdict}\t{";".join(gate_fail) or "none"}')
    lines.append(f'falsifier\t{"n/a" if not complete else ("FALSIFIED" if falsified else "not falsified")}\t{";".join(falsified) or "none"}')
    return '\n'.join(lines) + '\n'


def gate(tsv):
    line = next((l for l in Path(tsv).read_text().splitlines() if l.startswith('gate\t')), None)
    if line is None:
        return 6
    return {'PASS': 0, 'FAIL': 1}.get(line.split('\t')[1], 6)


def selftest():
    import tempfile
    def write(d, rf_scale, arms=ARMS, rf_rows1=1.0):
        for i, arm in enumerate(arms):
            for rep in range(MIN_RUNS[arm] + 1):
                with open(Path(d, f'perf-{i}-{rep}-{arm}.out'), 'w') as f:
                    for c, base_us in zip(CELLS, (10, 25, 42, 61, 200, 160)):
                        for b in (1, 8):
                            s = 1.0 if arm != 'rf' else (rf_rows1 if c == 'rows1' else rf_scale)
                            m = base_us * s + 0.01 * rep
                            f.write(f'perf arm={arm} cell={c} ne=5120 batch={b} calls=240 med_us={m:.2f} p25_us={m - .1:.2f} p75_us={m + .1:.2f}\n')
    rc = 0
    for name, scale, rows1, arms, want_gate, want_fals in (('pass', 0.75, 1.0, ARMS, 0, 'not falsified'),
                                                          ('weak', 0.85, 1.0, ARMS, 0, 'FALSIFIED'),
                                                          ('none', 0.95, 1.0, ARMS, 1, 'FALSIFIED'),
                                                          ('regress', 0.75, 1.05, ARMS, 1, 'FALSIFIED'),
                                                          ('missing', 0.75, 1.0, ['base', 'sf'], 6, 'n/a')):
        with tempfile.TemporaryDirectory() as d:
            write(d, scale, arms, rows1)
            text = render(table(Path(d)))
            Path(d, 'perf.tsv').write_text(text)
            g = gate(Path(d, 'perf.tsv'))
            f = next(l for l in text.splitlines() if l.startswith('falsifier\t')).split('\t')[1]
            ok = g == want_gate and f == want_fals
            print(f'selftest {name}: gate rc={g} falsifier={f} {"ok" if ok else "FAIL"}')
            rc |= not ok
    print('ar_summary: selftest', 'FAIL' if rc else 'PASS')
    return 1 if rc else 0


def main():
    if sys.argv[1] == '--selftest':
        return selftest()
    if sys.argv[1] == '--gate':
        return gate(sys.argv[2])
    sys.stdout.write(render(table(Path(sys.argv[1]))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
