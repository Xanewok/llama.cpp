#!/usr/bin/env python3
"""Compare full rows and laws produced by the frozen C++ sampler."""
import argparse
import json
import functools
import math
import pathlib
import re
import sys

import numpy as np


def load(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    row = np.fromfile(path, dtype='<f4').astype(np.float64)
    if len(row) != meta['n_vocab'] or not np.all(np.isfinite(row) | np.isneginf(row)):
        raise ValueError('invalid row: ' + str(path))
    law = {int(i): float(p) for i, p in meta['law'] if p > 0}
    if abs(sum(law.values()) - 1) > 1e-5:
        raise ValueError('invalid law: ' + str(path))
    return row, meta, law


def raw_kl(a, b):
    la = a - np.max(a)
    lb = b - np.max(b)
    la -= np.log(np.sum(np.exp(la)))
    lb -= np.log(np.sum(np.exp(lb)))
    mask = np.isfinite(la)
    if np.any(~np.isfinite(lb[mask])):
        return 'infinity'
    return max(0.0, float(np.sum(np.exp(la[mask]) * (la[mask] - lb[mask]))))


def difference(a, b):
    ar, am, ap = a
    br, bm, bp = b
    if ar.shape != br.shape or am['next_token'] != bm['next_token']:
        raise ValueError('incompatible rows or teacher histories')
    finite = np.isfinite(ar) & np.isfinite(br)
    union = ap.keys() | bp.keys()
    removed = sorted(ap.keys() - bp.keys())
    kl = 'infinity' if removed else max(0.0, sum(p * math.log(p / bp[i]) for i, p in ap.items()))
    return {
        'max_abs_M': float(np.max(np.abs(ar[finite] - br[finite]))) if finite.any() else 0.0,
        'finite_support_changes_M': int(np.count_nonzero(np.isfinite(ar) != np.isfinite(br))),
        'raw_KL_D': raw_kl(ar, br), 'filtered_KL_D': kl,
        'top20_changes_M': sorted(set(am['top20']) ^ set(bm['top20'])),
        'filtered_support_changes_M': sorted(ap.keys() ^ bp.keys()),
        'max_probability_diff_D': max((abs(ap.get(i, 0) - bp.get(i, 0)) for i in union), default=0),
        'filtered_TV_D': sum(abs(ap.get(i, 0) - bp.get(i, 0)) for i in union) / 2,
        'next_probability_delta_D': bm['next_probability_D'] - am['next_probability_D'],
        'loop_set_probability_delta_D': bm['loop_set_probability_D'] - am['loop_set_probability_D'],
        'same_seed_sample_changes_M': sum(x != y for x, y in zip(am['samples_M'], bm['samples_M'])),
    }


def beyond(test, noise):
    if noise is None:
        return None
    numerical = ('max_abs_M', 'max_probability_diff_D', 'filtered_TV_D', 'finite_support_changes_M')
    return any(test[k] > noise[k] + 1e-12 for k in numerical) or any(
        set(test[k]) - set(noise[k]) for k in ('top20_changes_M', 'filtered_support_changes_M'))


def merge_noise(items):
    if not items:
        return None
    result = dict(items[0])
    for item in items[1:]:
        for key in ('max_abs_M', 'max_probability_diff_D', 'filtered_TV_D', 'finite_support_changes_M'):
            result[key] = max(result[key], item[key])
        for key in ('top20_changes_M', 'filtered_support_changes_M'):
            result[key] = sorted(set(result[key]) | set(item[key]))
    return result


def analyze(root, strict=False):
    boots = []
    for path in sorted(root.glob('boots/*/boot.json')):
        boots.append((path.parent, json.loads(path.read_text())))
    if not boots:
        boots = [(root, {'config': 'cpu', 'boot': 0})]
    @functools.lru_cache(maxsize=64)
    def get(path):
        return load(path)

    records = []
    missing = []
    configs = {}
    for boot, meta in boots:
        configs.setdefault(meta['config'], []).append(boot)
    for boot, meta in boots:
        files = sorted(boot.glob('**/verify-*.f32'))
        for file in files:
            row = int(re.search(r'-r(\d+)', file.stem)[1])
            single = file.parent / f'single-r{row}.f32'
            repeat = file.parent / f'repeat-r{row}.f32'
            if not single.exists() or not repeat.exists():
                missing.append(str(file))
                continue
            try:
                get(file)
                get(single)
                get(repeat)
            except (OSError, ValueError) as error:
                missing.append(str(file) + ': ' + str(error))
                continue
            noises = [difference(get(single), get(repeat))]
            boot_pairs = []
            for other in configs[meta['config']]:
                other_file = other / single.relative_to(boot)
                other_batch = other / file.relative_to(boot)
                if other != boot and other_file.exists() and other_batch.exists():
                    boot_pairs += [difference(get(single), get(other_file)), difference(get(file), get(other_batch))]
            noise = merge_noise(noises + boot_pairs)
            diff = difference(get(single), get(file))
            records.append({'kind': 'verify_vs_single', 'row': str(file.relative_to(root)),
                            'metrics': diff, 'noise': noise, 'repeated_boot': bool(boot_pairs),
                            'beyond_noise': beyond(diff, noise) if boot_pairs or strict else None})
        # Record paths only after all conditional rows exist. This is not free-running generation.
        for point in sorted(boot.glob('**/point.json')):
            parent = point.parent
            for kind, count in [('single', 6), ('verify-k3', 4), ('verify-k4', 5), ('verify-k5', 6), ('draft', 5)]:
                rows = [parent / f'{kind}-r{i}.f32' for i in range(count)]
                if all(p.exists() for p in rows):
                    probability = math.prod(get(p)[1]['next_probability_D'] for p in rows)
                    records.append({'kind': 'teacher_path_probability', 'point': str(parent.relative_to(root)),
                                    'path': kind, 'positions_D': count, 'probability_D': probability})

    # Cross-configuration effects use each configuration's n=1 and include its independent boot noise.
    reference = next((c for c in configs if c == 'tensor-q8_0-1-subset'), next(iter(configs)))
    base = configs[reference][0]
    for config, group in configs.items():
        if config == reference:
            continue
        boot = group[0]
        for file in sorted(boot.glob('**/*.f32')):
            if not (file.stem.startswith('single-') or file.stem.startswith('draft-')):
                continue
            rel = file.relative_to(boot)
            ref = base / rel
            if not ref.exists():
                missing.append(str(ref))
                continue
            noises = []
            for name, first, item in [(reference, base, ref), (config, boot, file)]:
                for other in configs[name][1:]:
                    other_file = other / rel
                    if other_file.exists():
                        noises.append(difference(get(item), get(other_file)))
            diff = difference(get(ref), get(file))
            has_both = any((p / rel).exists() for p in configs[reference][1:]) and any(
                (p / rel).exists() for p in group[1:])
            noise = merge_noise(noises) if has_both else None
            records.append({'kind': 'config_effect', 'reference': reference, 'config': config, 'row': str(rel),
                            'metrics': diff, 'noise': noise, 'beyond_noise': beyond(diff, noise),
                            'expected_subset_mask_effect': file.stem.startswith('draft-') and 'full' in config})
    expected = root / 'expected.json'
    expected_boots = json.loads(expected.read_text()) if expected.exists() else []
    actual = {p.name for p, _ in boots}
    absent = [name for name in expected_boots if name not in actual or not (root / 'boots' / name / 'COMPLETE').exists()]
    summary = {'records_M': len(records), 'boots_M': len(boots), 'missing_boots': absent, 'missing_rows': missing,
               'flags_M': sum(r.get('beyond_noise') is True for r in records),
               'unresolved_noise_M': sum(r.get('beyond_noise', False) is None for r in records),
               'complete': not missing and not absent and bool(records)}
    (root / 'comparison.json').write_text(json.dumps({'summary': summary, 'records': records}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if strict and (not summary['complete'] or summary['flags_M']):
        raise ValueError('strict equivalence oracle failed')
    return summary


def validate_fixture(root):
    tested = 0
    for file in sorted(root.glob('d*/*.f32')):
        if not (file.stem.startswith('single-') or file.stem.startswith('verify-')):
            continue
        fresh = file.with_name('independent-' + file.name)
        a, b = load(file), load(fresh)
        if not np.array_equal(a[0], b[0]) or a[2] != b[2] or a[1]['samples_M'] != b[1]['samples_M']:
            raise ValueError('independent context oracle failed: ' + str(file))
        tested += 1
    if tested != 42:
        raise ValueError('incomplete CPU fixture: ' + str(tested))
    print(f'M: independent contexts reproduce {tested} rows, filtered laws and seeded samples')


def self_test():
    a = (np.array([0., -1., -2.]), {'next_token': 0, 'top20': [0, 1], 'next_probability_D': .7,
          'loop_set_probability_D': 1., 'samples_M': [0, 1]}, {0: .7, 1: .3})
    zero = difference(a, a)
    assert not beyond(zero, zero)
    b = (np.array([0., 2., -2.]), {**a[1], 'top20': [1, 2], 'next_probability_D': .2,
         'samples_M': [1, 1]}, {0: .2, 1: .8})
    assert beyond(difference(a, b), zero)
    c = (b[0], b[1], {1: 1.})
    assert difference(a, c)['filtered_KL_D'] == 'infinity'
    assert beyond(difference(a, c), zero)
    assert beyond(zero, None) is None
    print('M: comparison controls PASS; changed logits, law and support are detected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', type=pathlib.Path)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--validate-fixture', action='store_true')
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.validate_fixture:
            validate_fixture(args.root)
        else:
            analyze(args.root, args.strict)
    except (ValueError, AssertionError) as error:
        print('FAIL:', error, file=sys.stderr)
        sys.exit(1)
