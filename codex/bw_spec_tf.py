#!/usr/bin/env python3
"""Build-window evidence checks. This file never starts a model or submits an item."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tomllib

from gpu_payload import BASE, COV99, DATA, FROZEN, MODELS, ROOT, VOCAB


CPU_MODEL = Path('/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf')
WORKTREES = Path('/home/xanewok/repos/qwen38-eval-worktrees')
RESULTS = Path('/home/xanewok/repos/qwen38-eval/results')
FIXTURES = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')
SCRIPTS = (
    'bw-spec-tf.sh', 'bw_spec_tf.py', 'gate.sh', 'cpu_pipeline.sh', 'cpu_controls.sh',
    'cpu_validate.py', 'check_rows.py', 'check_controls.py', 'check_vocab.py', 'spec_tf.py',
    'run_tests.sh', 'test_spec_tf.py', 'test_estimator_io.py', 'test_import_streams.py',
    'test_gpu_payload.py', 'gen_gpu_item.py', 'gpu_payload.py', 'gpu_ab.sh', 'import_streams.py',
)


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def preflight():
    paths = [CPU_MODEL, BASE, VOCAB, COV99, ROOT / 'codex/bw-spec-tf.toml',
             ROOT / 'codex/cpu-exact/streams.json', ROOT / 'codex/cpu-exact/trace.jsonl',
             FIXTURES / 'xid-lines.txt', FIXTURES / 'llama-sse/long-tool-s1-1.json',
             Path('/usr/local/cuda-13.2/bin/nvcc'),
             Path('/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3/lib/cmake/cccl/cccl-config.cmake')]
    paths += [WORKTREES / 'bin' / name for name in ('buildgate', 'cpugate', 'wt')]
    paths += [ROOT / 'codex' / name for name in SCRIPTS]
    paths += [DATA / f'depth-{depth}.json' for depth in range(1500, 19001, 500)]
    for model in MODELS[1:]:
        paths += [model, model.with_name(model.name.replace('00001-of-00002', '00002-of-00002'))]
    for run in ('2026-10-06-nh6-nextn-lowbit-j5q/a2', '2026-10-06-nh6-nextn-q8-j5q/a1'):
        paths.append(RESULTS / run / 'pins.json')
    manifest = tomllib.loads((ROOT / 'codex/bw-spec-tf.toml').read_text())
    paths += [Path(p) for p in manifest['pins']['files']]
    missing = sorted({str(p) for p in paths if not p.is_file() or not p.stat().st_size})
    require(not missing, 'missing/empty input files: ' + ', '.join(missing))
    require(not FROZEN.exists() and not FROZEN.is_symlink(), str(FROZEN) + ' already exists')
    require(all(str(ROOT / 'codex' / name) in manifest['pins']['files'] for name in SCRIPTS),
            'a payload dependency is not pinned')
    import numpy
    print('preflight inputs exist; final freeze is absent; numpy=' + numpy.__version__)


def rows(root, expected):
    streams = read(root / 'streams.json')
    result = read(root / 'rows.json')
    target = read(root / 'tool/target/header.json')
    draft = read(root / 'tool/full/header.json')
    require(len(streams) == len(result['streams']) == len(target['streams']) == expected, 'stream matrix is incomplete')
    require(not result['errors'] and result['tolerance'] == 1e-3, 'row tolerance check failed')
    require(target['backend_sampling'] and target['continuation_chunk'] == 6, 'target conditioning flags differ')
    require(draft['backend_sampling'] and draft['carry_streams'] and draft['k'] == 5, 'draft conditioning flags differ')
    require(draft['target_header'] == target, 'draft references a different target')
    for i, (stream, row) in enumerate(zip(streams, result['streams'])):
        require(stream['id'] == f'p{i // 2:02d}-s{(1729, 2718)[i % 2]}', 'prompt/seed matrix differs')
        require(row['stream'] == i and row['rounds_M'] > 0 and row['p_rows_M'] > 0 and row['q_rows_M'] > 0,
                'row check has no observations')
        require(len(stream['continuation']) == 256, 'stream did not reach the token limit')
        require(row['support_changes_M'] == 0 and row['true_tokens_outside_tool_p_M'] == 0, 'probability support changed')
        for name in ('p_max_abs_error_M', 'q_max_abs_error_M'):
            require(math.isfinite(row[name]) and 0 <= row[name] <= 1e-3, 'nonfinite or excessive row error')
    print(f'row equivalence PASS; streams_M={expected}; p/q tolerance=1e-3; support changes_M=0')


def slice_vocab(root):
    import numpy as np
    from spec_tf import array

    target = read(root / 'tool/target/header.json')
    draft = read(root / 'tool/full/header.json')
    candidates, frequent = set(), Counter()
    for i, stream in enumerate(target['streams']):
        n, k = stream['n_continuation'] - 1, draft['k']
        valid = np.arange(n)[:, None] + np.arange(k) < n
        ids = array(root / f'tool/full/{i}', 'q_ids', '<i4', [n, k, 128])[valid]
        sampled = array(root / f'tool/full/{i}', 'q_sample_ids', '<i4', [n, k, 10])[valid]
        probs = array(root / f'tool/full/{i}', 'q_probs', '<f4', [n, k, 10])[valid]
        candidates.update(int(v) for v in np.unique(ids) if v >= 0)
        frequent.update(int(v) for v in sampled[np.arange(len(sampled)), probs.argmax(axis=1)])
    require(frequent and len(candidates) > 128, 'slice control has insufficient rows or candidates')
    removed = frequent.most_common(1)[0][0]
    subset = sorted(candidates - {removed})
    require(10 <= len(subset) < target['n_vocab'], 'slice control is not a proper subset')
    (root / 'slice-vocab.json').write_text(json.dumps(subset) + '\n')
    (root / 'slice-wrong-vocab.json').write_text(json.dumps(sorted(candidates)) + '\n')
    (root / 'slice-design.json').write_text(json.dumps({
        'subset_size_D': len(subset), 'removed_token': removed,
        'removed_top1_rows_M': frequent[removed],
        'coverage': 'union of every saved top-128 row, excluding one frequent top-1 token',
    }, indent=2) + '\n')
    print('slice vocab written; coverage and ties must pass check_vocab.py')


def slice_result(root, negative=False):
    path = root / ('slice-negative.json' if negative else 'slice-control.json')
    result = read(path)
    require(result['rows_M'] > 0 and result['tolerance'] == 1e-3, 'slice check has no observations')
    error = result['max_abs_error_M']
    require(math.isfinite(error) and (error > 1e-3 if negative else 0 <= error <= 1e-3), 'slice verdict is invalid')
    require(math.isfinite(result['paired_delta_offline_minus_real_D']), 'slice delta is nonfinite')
    if not negative:
        header = read(root / 'tool/sliced/header.json')
        require(header['draft_vocab'] == read(root / 'slice-vocab.json') and header['carry_streams'] and header['backend_sampling'],
                'real slice did not load the intended vocab or sampler')
    print(('wrong-vocab negative control' if negative else 'real sliced-head control') + ' PASS')


def deterministic(root):
    target = read(root / 'tool/target/header.json')
    full, rerun = root / 'tool/full', root / 'tool/rerun'
    require(read(full / 'header.json') == read(rerun / 'header.json'), 'draft rerun header changed')
    expected = {f'{i}/{name}.bin' for i in range(len(target['streams']))
                for name in ('starts', 'q_ids', 'q_logits', 'q_sample_ids', 'q_probs')}
    for arm in (full, rerun):
        require({str(p.relative_to(arm)) for p in arm.glob('*/*.bin')} == expected, 'draft file inventory differs')
    for stream in target['streams']:
        require(stream['n_continuation'] == 256, 'draft rerun has a short stream')
    for name in sorted(expected):
        require((full / name).stat().st_size > 0, 'empty draft output')
        require(hashlib.sha256((full / name).read_bytes()).digest() == hashlib.sha256((rerun / name).read_bytes()).digest(),
                'draft rerun differs: ' + name)
    controls = read(root / 'controls/summary.json')['arms']
    require(controls['rerun']['paired_delta_G'] == controls['rerun']['stream_se_D'] == 0, 'identical-arm control differs')
    repeated = read(root / 'deterministic-server/streams.json')
    require(len(repeated) == 2 and repeated == read(root / 'streams.json')[:2], 'seeded server tokens differ')
    original = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
    repeat_trace = [json.loads(line) for line in (root / 'deterministic-server/trace.jsonl').read_text().splitlines()]
    require(repeat_trace and repeat_trace == [row for row in original if row['r'] < 2], 'seeded server probability trace differs')
    print(f'deterministic PASS; identical draft files_M={len(expected)}; seeded server streams_M=2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=['preflight', 'rows', 'slice-vocab', 'slice-result', 'slice-negative', 'deterministic'])
    parser.add_argument('root', nargs='?', type=Path)
    parser.add_argument('--streams', type=int, default=32)
    args = parser.parse_args()
    if args.check == 'preflight':
        preflight()
    else:
        require(args.root is not None, 'output directory is required')
        if args.check == 'rows':
            rows(args.root, args.streams)
        elif args.check == 'slice-vocab':
            slice_vocab(args.root)
        elif args.check in ('slice-result', 'slice-negative'):
            slice_result(args.root, args.check == 'slice-negative')
        else:
            deterministic(args.root)


if __name__ == '__main__':
    main()
