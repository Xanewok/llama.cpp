"""Summarize the window-size ABBA of test-backend-ops perf: usage fa_window_perf.py OUT_DIR DEV...

Reads OUT_DIR/perf-DEV-NN-wW.log (NN = run order, W = GGML_CUDA_FA_KV_WINDOW cells, 0 = one-pass conversion). Per case: the mean
us/run of every window size and its ratio to the one-pass runs; A/A spread = |first - last| one-pass run. Prints the table and writes
perf-summary.json. Exit 1 if a log lacks a case. A case whose one-pass runs differ by more than 3 % is flagged NOISE (its ratios are no evidence).
"""
import json
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r'^\s*FLASH_ATTN_EXT\((.*)\):\s+(\d+) runs - +([0-9.]+) us/run')
FILE = re.compile(r'perf-(CUDA\d)-(\d+)-w(\d+)\.log$')


def parse(path):
    rows = {}
    for line in path.read_text(errors='replace').splitlines():
        m = LINE.match(line.replace('\x1b[0m', ''))
        if m:
            rows[m.group(1)] = float(m.group(3))
    return rows


def short(name):
    f = dict(p.split('=', 1) for p in re.split(r',(?![^\[]*\])', name) if '=' in p)
    return f"hs{f['hsk']} nr23={f['nr23']} kv={f['kv']} nb={f['nb']} sinks={f['sinks']}"


def main():
    out, devs = Path(sys.argv[1]), sys.argv[2:]
    bad, summary = [], {}
    for dev in devs:
        runs = {}
        for p in sorted(out.glob(f'perf-{dev}-*.log')):
            m = FILE.search(p.name)
            runs.setdefault(int(m.group(3)), []).append((int(m.group(2)), parse(p)))
        base = runs.get(0, [])
        if len(base) < 2 or not base[0][1]:
            bad.append(f'{dev}: fewer than two one-pass runs or no cases parsed')
            continue
        cases = sorted(base[0][1])
        table = {}
        for case in cases:
            row = {}
            for w, items in runs.items():
                vals = [r[case] for _, r in items if case in r]
                if len(vals) != len(items):
                    bad.append(f'{dev} w={w}: case missing {short(case)}')
                    continue
                row[w] = statistics.mean(vals)
            first, last = base[0][1][case], base[-1][1][case]
            row['aa'] = abs(first - last)/min(first, last)
            if row['aa'] > 0.03:
                print(f'NOISE {dev}: one-pass runs differ by {100*row["aa"]:.1f} % at {short(case)}; its ratios are not evidence')
            table[case] = row
        summary[dev] = table
        windows = sorted(w for w in runs if w)
        print(f'{dev}: us/run mean over the ABBA runs; ratio = window / one-pass; A/A = spread of the first and last one-pass run')
        print('  ' + 'case'.ljust(50) + 'one-pass'.rjust(10) + ''.join(f'  w={w}'.rjust(16) for w in windows) + '   A/A')
        for case, row in table.items():
            print('  ' + short(case).ljust(50) + f'{row[0]:10.1f}' + ''.join(f'{row[w]:10.1f} {row[w]/row[0]:5.3f}' for w in windows if w in row) +
                  f'  {100*row["aa"]:4.1f}%')
    (out/'perf-summary.json').write_text(json.dumps({d: {short(k): {str(w): v for w, v in r.items()} for k, r in t.items()}
                                                     for d, t in summary.items()}, indent=1))
    for b in bad:
        print('PERF GATE VIOLATION:', b)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
