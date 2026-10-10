#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from spec_tf import array, draft_candidates, draft_probs, load_stream, paired, validate_rows


def compare_rows(ids, logits, real_ids, real_q, subset, temperature, suppress, n_vocab):
    validate_rows(real_ids, real_q, n_vocab)
    _, _, reasons = draft_candidates(ids, logits, subset, suppress=suppress)
    covered = ~np.logical_or.reduce(list(reasons.values()))
    if not covered.any():
        return covered, reasons, 0.0
    offline_ids, q = draft_probs(ids[covered], logits[covered], subset, temperature, suppress=suppress)
    real_ids, real_q = real_ids[covered], real_q[covered]
    aligned = ((offline_ids[:, :, None] == real_ids[:, None, :]) * real_q[:, None, :]).sum(axis=-1)
    # Missing real mass must fail even if the omitted offline probabilities are small.
    missing = ((real_ids[:, :, None] != offline_ids[:, None, :]).all(axis=-1) * real_q).sum(axis=-1)
    maximum = max(float(np.abs(q - aligned).max()), float(missing.max()))
    return covered, reasons, maximum


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
    c = json.loads((args.sliced / 'header.json').read_text())
    if b['target_header'] != h or c['target_header'] != h or b['k'] != c['k']:
        raise ValueError('control arms must reference the same target and horizon')
    subset = json.loads(args.vocab.read_text())
    maximum = 0.0
    deltas = []
    count = 0
    total = 0
    starts_total = 0
    starts_covered = 0
    streams = []
    for i, stream in enumerate(h['streams']):
        s, k = stream['n_continuation'] - 1, b['k']
        n = stream['n_tokens']
        starts = array(args.full / str(i), 'starts', '<i4', [s, 1])[:, 0]
        real_starts = array(args.sliced / str(i), 'starts', '<i4', [s, 1])[:, 0]
        if not np.array_equal(starts, real_starts):
            raise ValueError('control arms have different starts')
        valid = starts[:, None] + np.arange(k) < n - 1
        ids = array(args.full / str(i), 'q_ids', '<i4', [s, k, b['q_width']])[valid]
        logits = array(args.full / str(i), 'q_logits', '<f4', [s, k, b['q_width']])[valid]
        real_ids = array(args.sliced / str(i), 'q_sample_ids', '<i4', [s, k, 10])[valid]
        real_q = array(args.sliced / str(i), 'q_probs', '<f4', [s, k, 10])[valid]
        covered, reasons, error = compare_rows(ids, logits, real_ids, real_q, subset, b['temperature'], b['suppress_ids'], h['n_vocab'])
        maximum = max(maximum, error)
        count += int(covered.sum())
        total += len(covered)
        chain_covered = np.zeros(valid.shape, dtype=bool)
        chain_covered[valid] = covered
        complete = starts + k < n
        chain_covered = chain_covered.all(axis=1) & complete
        starts_total += int(complete.sum())
        starts_covered += int(chain_covered.sum())
        if chain_covered.any():
            offline = load_stream(args.target, args.full, i, subset, chain_covered)[-1]
            real = load_stream(args.target, args.sliced, i, start_mask=chain_covered)[-1]
            deltas.append(paired(real, offline))
        excluded = np.argwhere(valid)[~covered]
        streams.append({'stream': i, 'id': stream['id'], 'rows_total_M': len(covered),
                        'rows_M': int(covered.sum()), 'max_abs_error_M': error,
                        'starts_total_M': int(complete.sum()), 'starts_covered_M': int(chain_covered.sum()),
                        'excluded_by_reason_M': {name: int(mask.sum()) for name, mask in reasons.items()},
                        'excluded_positions_M': [{'start': int(starts[row]), 'step': int(step + 1),
                                                  'position': int(starts[row] + step)} for row, step in excluded]})
    result = {'rows_M': count, 'max_abs_error_M': maximum, 'tolerance': args.tolerance,
              'rows_total_M': total, 'rows_excluded_M': total - count,
              'excluded_fraction_D': (total - count) / total if total else None,
              'starts_total_M': starts_total, 'starts_covered_M': starts_covered,
              'delta_streams_M': len(deltas),
              'paired_delta_offline_minus_real_D': float(np.mean(deltas)) if deltas else None,
              'delta_scope': 'covered complete chains only; coverage-conditioned, not a full-arm estimate',
              'streams': streams, 'passed': count > 0 and maximum <= args.tolerance}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
