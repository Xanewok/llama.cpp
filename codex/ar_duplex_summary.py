"""perf-*.out of the ar-duplex window -> TSV: per cell and batch, median of the per-run medians, the IQR, the delta to the base arm,
and the per-row slope. Exit 0 always; the gates are in bw-ar-duplex.sh. usage: ar_duplex_summary.py OUTDIR"""
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r'^perf arm=(\S+) cell=(\S+) ne=(\d+) batch=(\d+) calls=(\d+) med_us=([\d.]+) p25_us=([\d.]+) p75_us=([\d.]+)$')
ROWS = {'rows1': 1, 'rows6': 6, 'rows12': 12, 'rows18': 18, 'rows64': 64}


def main():
    out = Path(sys.argv[1])
    runs = defaultdict(list)   # (arm, cell, batch) -> [(med, iqr, calls)]
    for f in sorted(out.glob('perf-*.out')):
        for line in f.read_text().splitlines():
            m = PAT.match(line)
            if m:
                arm, cell, _, batch, calls, med, p25, p75 = m.groups()
                runs[(arm, cell, int(batch))].append((float(med), float(p75) - float(p25), int(calls)))
    arms = sorted({k[0] for k in runs}, key=lambda a: (a != 'base', a))
    cells = ['rows1', 'rows6', 'rows12', 'rows18', 'rows64', 'mb0.5']
    med = {k: st.median(r[0] for r in v) for k, v in runs.items()}
    print('batch\tcell\tarm\truns\tcalls_per_run\tmed_us\tiqr_us\tdelta_us\tdelta_pct\tmin_run_med\tmax_run_med')
    for batch in (1, 8):
        for cell in cells:
            b = med.get(('base', cell, batch))
            for arm in arms:
                v = runs.get((arm, cell, batch))
                if not v:
                    continue
                m = med[(arm, cell, batch)]
                d = '' if b is None else f'{m - b:+.2f}'
                dp = '' if b is None else f'{100 * (m - b) / b:+.1f}'
                print(f'{batch}\t{cell}\t{arm}\t{len(v)}\t{v[0][2]}\t{m:.2f}\t{st.median(r[1] for r in v):.2f}\t{d}\t{dp}\t{min(r[0] for r in v):.2f}\t{max(r[0] for r in v):.2f}')
    print()
    print('slope\tbatch\tarm\tus_per_row_6_to_12\tus_per_row_6_to_18\tfixed_us_at_1_row\tfalsifier_12rows_gt_2.6')
    for batch in (1, 8):
        for arm in arms:
            t = {ROWS[c]: med.get((arm, c, batch)) for c in ROWS}
            if None in (t[6], t[12], t[18], t[1]):
                continue
            s12 = (t[12] - t[6]) / 6
            s18 = (t[18] - t[6]) / 12
            print(f'slope\t{batch}\t{arm}\t{s12:.3f}\t{s18:.3f}\t{t[1]:.2f}\t{"n/a" if arm == "base" else ("FALSIFIED" if s12 > 2.6 else "not falsified")}')
    print()
    print('slower_than_base\tbatch\tarm\tcells where median > base median by > 2 pct (the default-on rule is "not slower at any size")')
    for batch in (1, 8):
        for arm in arms:
            if arm == 'base':
                continue
            slow = []
            for cell in cells:
                b, a = med.get(('base', cell, batch)), med.get((arm, cell, batch))
                if b and a and a > b * 1.02:
                    slow.append(f'{cell}:{100 * (a - b) / b:+.1f}%')
            print(f'slower_than_base\t{batch}\t{arm}\t{",".join(slow) or "none"}')


main()
