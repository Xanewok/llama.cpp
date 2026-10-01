#!/usr/bin/env python3
import array
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
runs = [root / name for name in ('0-up', '1-patch', '2-patch', '3-up')]
ok = True
for run in runs:
    if (run / 'status').read_text().strip() != '0':
        print(f'M: {run.name} failed; read {run / "probe.log"}')
        ok = False
names = [set(p.name for p in (run / 'logits').glob('*.f32')) for run in runs]
if not names[0] or any(n != names[0] for n in names):
    raise SystemExit('Incomplete or unequal logit artifacts')
for name in sorted(names[0]):
    values = []
    hashes = []
    for run in runs:
        data = (run / 'logits' / name).read_bytes()
        hashes.append(hashlib.sha256(data).hexdigest())
        a = array.array('f')
        a.frombytes(data)
        values.append(a)
    if len({len(v) for v in values}) != 1:
        raise SystemExit(f'Unequal row sizes: {name}')
    if len(set(hashes)) != 1:
        error = max(abs(a-b) for a, b in zip(values[0], values[1]))
        print(f'M: DIFFER {name}: max_abs(up,patch)={error:.9g}; hashes={hashes}')
        ok = False
print(f'M: {len(names[0])} rows per boot; bit-identical across up/patch/repeated boots: {ok}')
raise SystemExit(0 if ok else 1)
