#!/usr/bin/env python3
"""Reanalyse recorded teacher-forced dumps without loading a model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import resource

import numpy as np

from gpu_payload import COV99, VOCAB
from spec_tf import array, validate_rows


ARMS = ('base', 'q3_k', 'iq3_s', 'q8_0', 'cov99', 'full')


def read(path):
    return json.loads(path.read_text())


def se(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


def inventory(root):
    target = read(root / 'target/header.json')
    recorded = read(root / 'streams.json')
    if len(target['streams']) != len(recorded):
        raise ValueError('target and recorded stream counts differ')
    result = {}
    for name in ('target', *ARMS):
        verdict = read(root / f'{name}.verdict.json')
        if verdict != {'returncode': 0, 'fault_rc': 0}:
            raise ValueError(f'{name}: extraction verdict failed')
        header = read(root / name / 'header.json')
        if name != 'target' and header['target_header'] != target:
            raise ValueError(f'{name}: target header differs')
        log = (root / f'{name}.log').read_text()
        files, size = 0, 0
        support = []
        for i, stream in enumerate(target['streams']):
            n, s = stream['n_tokens'], stream['n_continuation'] - 1
            if name == 'target':
                shapes = {'tokens': [n], 'hidden': [n, target['n_embd']],
                          'p_ids': [n, target['p_width']], 'p_probs': [n, target['p_width']],
                          'p_raw_ids': [n, 24], 'p_raw_logits': [n, 24], 'p_logsumexp': [n]}
                tokens = array(root / name / str(i), 'tokens', '<i4', [n])
                if stream['id'] != recorded[i]['id'] or not np.array_equal(tokens, recorded[i]['prompt'] + recorded[i]['continuation']):
                    raise ValueError(f'{name}/{i}: recorded tokens differ')
                positions = np.arange(stream['n_prompt'], n - 1)
                p_ids = array(root / name / str(i), 'p_ids', '<i4', [n, target['p_width']])[positions]
                p = array(root / name / str(i), 'p_probs', '<f4', [n, target['p_width']])[positions]
                validate_rows(p_ids, p, target['n_vocab'])
                outside = ~((p_ids == tokens[positions + 1, None]) & (p > 0)).any(axis=1)
                support.append({'id': stream['id'], 'rows_M': len(positions),
                                'true_tokens_outside_p_M': int(outside.sum()),
                                'outside_positions_M': positions[outside].tolist()})
            else:
                k = header['k']
                shapes = {'starts': [s, 1], 'q_ids': [s, k, header['q_width']],
                          'q_logits': [s, k, header['q_width']], 'q_sample_ids': [s, k, 10], 'q_probs': [s, k, 10]}
                starts = array(root / name / str(i), 'starts', '<i4', [s, 1])[:, 0]
                if not np.array_equal(starts, np.arange(stream['n_prompt'], n - 1)):
                    raise ValueError(f'{name}/{i}: starts differ')
            expected = {root / name / str(i) / f'{key}.bin' for key in shapes}
            if set((root / name / str(i)).glob('*.bin')) != expected:
                raise ValueError(f'{name}/{i}: binary inventory differs')
            for key, shape in shapes.items():
                path = root / name / str(i) / f'{key}.bin'
                expected_size = int(np.prod(shape)) * 4
                if path.stat().st_size != expected_size:
                    raise ValueError(f'{path}: binary size differs')
                files += 1
                size += expected_size
            stage = 'A' if name == 'target' else 'B'
            if f'stage {stage} stream {i} complete' not in log:
                raise ValueError(f'{name}/{i}: completion log missing')
        result[name] = {'verdict_M': verdict, 'streams_M': len(target['streams']), 'binary_files_M': files, 'bytes_M': size}
        if name == 'target':
            result[name]['support'] = support
            result[name]['true_tokens_outside_p_M'] = sum(row['true_tokens_outside_p_M'] for row in support)
    return result


def run_logged(cmd, path):
    with path.open('x') as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.input.resolve()):
        raise ValueError('output must be outside the input dumps')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    snapshot = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in args.input.rglob('*') if p.is_file()}
    report = {'inventory': inventory(args.input), 'sample_unit': 'stream (one per prompt depth)',
              'absolute_label': 'G', 'delta_label': 'G', 'se_label': 'D', 'controls': {}, 'arms': {}}
    here = Path(__file__).resolve().parent
    for name, vocab in (('base', VOCAB), ('cov99', COV99)):
        output = args.output / f'control-{name}.json'
        rc = run_logged([sys.executable, str(here / 'check_vocab.py'), '--target', str(args.input / 'target'),
                         '--full', str(args.input / 'full'), '--sliced', str(args.input / name),
                         '--vocab', str(vocab), '--output', str(output)], args.output / f'control-{name}.log')
        control = read(output)
        report['controls'][name] = {key: value for key, value in control.items() if key != 'streams'}
        report['controls'][name]['returncode_M'] = rc
    cmd = [sys.executable, str(here / 'spec_tf.py'), '--target', str(args.input / 'target'),
           '--output', str(args.output / 'estimate')]
    for name in ARMS:
        cmd += ['--arm', str(args.input / name)]
    if run_logged(cmd, args.output / 'estimate.log'):
        raise ValueError('real-arm estimate failed')
    summary = read(args.output / 'estimate/summary.json')
    target = read(args.input / 'target/header.json')
    for name in ARMS:
        means, inverse_means, beta_means, reaches, depths = [], [], [], [], []
        for i, stream in enumerate(target['streams']):
            with np.load(args.output / f'estimate/{name}-{i}.npz') as data:
                t, beta, r = data['T'], data['beta'], data['r']
                weights = np.concatenate((np.ones_like(r[:, :1]), np.cumprod(r[:, :-1], axis=1)), axis=1)
                reach = np.mean(weights * beta, axis=0)
                means.append(float(t.mean()))
                inverse_means.append(float(np.mean(1 / t)))
                beta_means.append(beta.mean(axis=0))
                reaches.append(reach)
                depths.append({'id': stream['id'], 'prompt_tokens_M': stream['n_prompt'],
                               **summary['arms'][name]['streams'][i], 'mean_T_G': means[-1],
                               'beta_mean_D': beta_means[-1].tolist(), 'reach_probability_G': reach.tolist(),
                               'stream_se_D': None})
        arm = summary['arms'][name]
        report['arms'][name] = {key: value for key, value in arm.items() if key != 'streams'}
        report['arms'][name].update({'starts_M': sum(row['starts'] for row in depths),
                                    'mean_T_G': float(np.mean(means)), 'mean_T_se_D': se(means),
                                    'harmonic_se_D': se(inverse_means) / np.mean(inverse_means) ** 2,
                                    'beta_mean_D': np.mean(beta_means, axis=0).tolist(),
                                    'beta_stream_se_D': [se(column) for column in np.array(beta_means).T],
                                    'reach_probability_G': np.mean(reaches, axis=0).tolist(),
                                    'reach_stream_se_D': [se(column) for column in np.array(reaches).T],
                                    'depths': depths})
    # The design registers a direction check, not a significance or speed threshold.
    report['verdicts'] = {'q3_k_direction': 'PASS' if report['arms']['q3_k']['paired_delta_G'] < 0 else 'FAIL',
                          'control_base': 'PASS' if report['controls']['base']['passed'] else 'FAIL',
                          'control_cov99': 'PASS' if report['controls']['cov99']['passed'] else 'FAIL',
                          'offline_full_population': 'UNIDENTIFIED where top-128 coverage fails; real slices used',
                          'gpu_row_equivalence': 'NOT MEASURED against a served GPU trace',
                          'gpu_selection_bias': 'NOT MEASURED; absolute estimates and deltas remain G',
                          'other_arm_efficacy': 'DESCRIPTIVE; design registers no decision threshold'}
    if snapshot != {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in args.input.rglob('*') if p.is_file()}:
        raise ValueError('input dump inventory changed during reanalysis')
    report['input_sizes_and_mtimes_unchanged_M'] = True
    report['elapsed_s_M'] = time.monotonic() - started
    report['self_peak_rss_KiB_M'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['child_peak_rss_KiB_M'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report['concurrent_rss_bound_KiB_D'] = report['self_peak_rss_KiB_M'] + report['child_peak_rss_KiB_M']
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'arms'}, indent=2))
    return int(any(not control['passed'] for control in report['controls'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
