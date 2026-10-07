#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1])
    for base in (root / 'tool/full').glob('*/*.bin'):
        other = root / 'tool/rerun' / base.relative_to(root / 'tool/full')
        if hashlib.sha256(base.read_bytes()).digest() != hashlib.sha256(other.read_bytes()).digest():
            raise ValueError('draft rerun is not bit-identical: ' + str(base))
    arms = json.loads((root / 'controls/summary.json').read_text())['arms']
    if arms['rerun']['paired_delta_G'] != 0 or arms['rerun']['stream_se_D'] != 0:
        raise ValueError('identical arm has nonzero delta')
    base = arms['full']['tokens_per_round_G']
    result = {'deterministic_M': True, 'identical_delta_D': 0.0, 'base_tokens_per_round_G': base}
    for name in ('misaligned', 'permuted'):
        estimate = arms[name]['tokens_per_round_G']
        result[name + '_tokens_per_round_G'] = estimate
        result[name + '_acceptance_loss_fraction_D'] = 1 - (estimate - 1) / (base - 1)
        if result[name + '_acceptance_loss_fraction_D'] < 0.5:
            raise ValueError(name + ' failed the 50% acceptance-loss control')
    (root / 'controls/negative-controls.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
