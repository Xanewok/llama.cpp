import pathlib
import re
import statistics

for mode in (0, 1):
    for arm in ('base', 'patch'):
        totals = []
        upload_totals = []
        for run in (1, 2, 3):
            fills, uploads = [], []
            for line in pathlib.Path(f'codex/{mode}-{arm}-{run}.log').read_text().splitlines():
                if match := re.search(r'input_fill: .*us=(\d+)', line):
                    fills.append(int(match[1]))
                    uploads.append(0)
                elif match := re.search(r'input_upload: .*us=(\d+)', line):
                    uploads[-1] += int(match[1])
            if len(fills) != 20:
                raise SystemExit(f'Expected 20 decodes, got {len(fills)}')
            # Exclude the initial verify and catch-up builds; keep identical input shapes in each arm.
            rows = [a + b for a, b in zip(fills[2:], uploads[2:])]
            totals.append(statistics.mean(rows))
            upload_totals.append(statistics.mean(uploads[2:]))
        print(f'M {"CPU" if mode == 0 else "Meta(CPU,CPU)"} {arm}: '
              f'fill+upload {statistics.median(totals):.3f} us/decode; '
              f'upload {statistics.median(upload_totals):.3f} us/decode; '
              f'run means {totals}; 18 decodes/run, 3 runs')
