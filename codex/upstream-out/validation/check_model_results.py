import json
import sys
from pathlib import Path

mode, directory = sys.argv[1:]
assert mode == 'chain'
p = Path(directory)
for np in (1, 2):
    off = json.loads((p / f'np{np}-0.json').read_text())
    on = json.loads((p / f'np{np}-1.json').read_text())
    assert off and off == on, ('tokens or counters', np)
    a = [line.split() for line in (p / f'np{np}-0.dump').read_text().splitlines()]
    b = [line.split() for line in (p / f'np{np}-1.dump').read_text().splitlines()]
    assert a and len(a) == len(b), ('draft rows', np)
    assert all(x[:13] == y[:13] for x, y in zip(a, b)), ('draft IDs', np)
    dq = max(abs(float(x) - float(y)) for row_a, row_b in zip(a, b) for x, y in zip(row_a[13:], row_b[13:]))
    assert dq <= 2e-7, ('draft q', np, dq)
    if np == 2:
        assert (p / 'np2-0.dump').read_bytes() == (p / 'np2-1.dump').read_bytes()
        assert 'MTP chain: k=' not in (p / 'np2-1.log').read_text()
        shapes = [[line for line in (p / f'np2-{arm}.log').read_text().splitlines() if line.startswith('PACK_SHAPE ')] for arm in (0, 1)]
        assert shapes[0] and shapes[0] == shapes[1], 'np2 decode shapes differ'
        assert any('type=1' in line and 'seqs=1' in line for line in shapes[0]), 'idle-peer draft path absent'
        assert any('type=1' in line and 'seqs=2' in line for line in shapes[0]), 'two-slot draft path absent'
    else:
        assert 'MTP chain: k=3' in (p / 'np1-1.log').read_text()
    print(f'M: np{np} identity PASS, draft rows={len(a)}, max_abs_dq={dq}')
