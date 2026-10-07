#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from spec_tf import array, draft_probs, load_stream, paired


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--full', type=Path, required=True)
    p.add_argument('--sliced', type=Path, required=True)
    p.add_argument('--vocab', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=1e-3)
    args = p.parse_args()
    h = json.loads((args.target / 'header.json').read_text())
    b = json.loads((args.full / 'header.json').read_text())
    subset = json.loads(args.vocab.read_text())
    maximum = 0.0
    deltas = []
    count = 0
    for i, stream in enumerate(h['streams']):
        s, k = stream['n_continuation'] - 1, b['k']
        n = stream['n_tokens']
        valid = np.arange(stream['n_prompt'], n - 1)[:, None] + np.arange(k) < n - 1
        ids = array(args.full / str(i), 'q_ids', '<i4', [s, k, 128])[valid]
        logits = array(args.full / str(i), 'q_logits', '<f4', [s, k, 128])[valid]
        offline_ids, q = draft_probs(ids, logits, subset, b['temperature'], suppress=b['suppress_ids'])
        real_ids = array(args.sliced / str(i), 'q_sample_ids', '<i4', [s, k, 10])[valid]
        real_q = array(args.sliced / str(i), 'q_probs', '<f4', [s, k, 10])[valid]
        aligned = ((offline_ids[:, :, None] == real_ids[:, None, :]) * real_q[:, None, :]).sum(axis=-1)
        # Missing real mass must fail even if the omitted offline probabilities are small.
        missing = ((real_ids[:, :, None] != offline_ids[:, None, :]).all(axis=-1) * real_q).sum(axis=-1)
        maximum = max(maximum, float(np.abs(q - aligned).max()), float(missing.max()))
        count += len(q)
        offline = load_stream(args.target, args.full, i, subset)[-1]
        real = load_stream(args.target, args.sliced, i)[-1]
        deltas.append(paired(real, offline))
    result = {'rows_M': count, 'max_abs_error_M': maximum, 'tolerance': args.tolerance,
              'paired_delta_offline_minus_real_D': float(np.mean(deltas))}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return int(maximum > args.tolerance)


if __name__ == '__main__':
    raise SystemExit(main())
