#!/usr/bin/env python3
"""Cold Swift MUL_MAT sweep. --prepare reads GGUF headers only; --run executes queued GPU work."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = [Path('/home/xanewok/models') / f'Swift-1.5-Qwen3.8-27B-{s}.gguf'
          for s in ('IQ4_XS', 'J5q-bf16')]
TYPES = {14: 'q6_K', 13: 'q5_K', 23: 'iq4_xs'}
BLOCK_BYTES = {14: 210, 13: 176, 23: 136}
WIDTHS = (1, 5, 6, 7, 8, 12)
HEADER_LIMIT = 200 * 1024**2


class Header:
    def __init__(self, stream):
        self.stream = stream

    def read(self, n):
        if n < 0 or self.stream.tell() + n > HEADER_LIMIT:
            raise ValueError('GGUF metadata exceeds the 200 MiB read limit')
        data = self.stream.read(n)
        if len(data) != n:
            raise ValueError('truncated GGUF metadata')
        return data

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))

    def string(self):
        n, = self.unpack('Q')
        return self.read(n).decode('utf-8')

    def value(self, kind, keep=False):
        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                   6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind in formats:
            return self.unpack(formats[kind])[0]
        if kind == 8:
            return self.string() if keep else self.read_string_skip()
        if kind == 9:
            subtype, count = self.unpack('IQ')
            if count > 2_000_000 or subtype == 9:
                raise ValueError('unsupported GGUF metadata array')
            if subtype in formats:
                self.read(count * struct.calcsize('<' + formats[subtype]))
            else:
                for _ in range(count):
                    self.value(subtype)
            return None
        raise ValueError(f'unknown GGUF metadata type {kind}')

    def read_string_skip(self):
        n, = self.unpack('Q')
        self.read(n)
        return None


def tensor_table(path):
    with path.open('rb') as stream:
        h = Header(stream)
        magic, version, count, kvs = h.unpack('4sIQQ')
        if magic != b'GGUF' or version not in (2, 3) or max(count, kvs) > 1_000_000:
            raise ValueError(f'invalid GGUF header: {path}')
        metadata = {}
        for _ in range(kvs):
            key = h.string()
            kind, = h.unpack('I')
            keep = key in ('general.architecture', 'general.name') or key.endswith(('.block_count', '.nextn_predict_layers'))
            value = h.value(kind, keep)
            if keep:
                metadata[key] = value
        tensors = []
        for _ in range(count):
            name = h.string()
            rank, = h.unpack('I')
            if not 1 <= rank <= 4:
                raise ValueError(f'invalid rank: {name}')
            shape = list(h.unpack('Q' * rank))
            kind, offset = h.unpack('IQ')
            tensors.append(dict(name=name, shape=shape, type=kind, offset=offset))
        end = stream.tell()
        stream.seek(0)
        digest = hashlib.sha256(stream.read(end)).hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, metadata_bytes=end,
                header_sha256=digest, metadata=metadata, tensors=tensors)


def fixtures(sources):
    rows = {}
    exclusions = []
    for source in sources:
        arch = source['metadata']['general.architecture']
        layers = source['metadata'][arch + '.block_count'] - source['metadata'].get(arch + '.nextn_predict_layers', 0)
        for tensor in source['tensors']:
            name, dims, kind = tensor['name'], tensor['shape'], tensor['type']
            if kind not in TYPES or len(dims) != 2 or not name.endswith('.weight'):
                continue
            if name == 'token_embd.weight' or (name.startswith('blk.') and int(name.split('.')[1]) >= layers):
                exclusions.append(dict(model=source['path'], **tensor, reason='embedding lookup or MTP'))
                continue
            role = name.split('.')[-2]
            k, m = dims
            if role in ('ffn_down', 'attn_output', 'ssm_out'):
                k //= 2
            elif role in ('ffn_gate', 'ffn_up', 'attn_q', 'attn_k', 'attn_v', 'attn_qkv', 'attn_gate', 'output'):
                m //= 2
            else:
                raise ValueError(f'add an explicit TP2 split policy for {name}')
            if k % 256 or m < 1024:
                raise ValueError(f'unexpected TP2 shape: {name} {m} {k}')
            key = (source['path'], kind, m, k, role == 'ssm_out')
            row = rows.setdefault(key, dict(model=source['path'], type=TYPES[kind], type_id=kind,
                                             m=m, k=k, gdn=role == 'ssm_out', tensors=[]))
            row['tensors'].append(name)
    result = list(rows.values())
    for row in result:
        row['count'] = len(row['tensors'])
        row['weight_bytes'] = row['m'] * row['k'] // 256 * BLOCK_BYTES[row['type_id']]
    return result, exclusions


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(out):
    sources = [tensor_table(path) for path in MODELS]
    rows, exclusions = fixtures(sources)
    manifest = dict(sources=[{k: v for k, v in source.items() if k != 'tensors'} for source in sources],
                    shapes=rows, exclusions=exclusions,
                    split_source='src/llama-model.cpp:393-921; equal TP2, separate FFN gate/up, actual output width',
                    widths=list(WIDTHS), cold_bytes=64 * 1024**2)
    write_json(out / 'shapes.json', manifest)
    lines = ['| Model | Type | m | k | GDN | Count |', '|---|---|---:|---:|---|---:|']
    for row in rows:
        lines.append(f"| {Path(row['model']).stem} | {row['type']} | {row['m']} | {row['k']} | {row['gdn']} | {row['count']} |")
    (out / 'shapes.md').write_text('\n'.join(lines) + '\n')
    return manifest


def ci(values):
    mean = statistics.mean(values)
    rng = random.Random(1729)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return dict(mean=mean, ci95=[means[49], means[1949]], samples=values)


def measure(binary, row, n, batches, arm, log):
    env = dict(os.environ, GGML_TEST_FLATCOST_CASE=f"{row['type_id']},{row['m']},{n},{row['k']},{batches}",
               GGML_CUDA_DISABLE_GRAPHS='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    env.pop('GGML_CUDA_MMVQ_MAX_COLS', None)
    if arm != 'default':
        env['GGML_CUDA_MMVQ_MAX_COLS'] = '8' if arm == 'mmvq' else '0'
    command = [str(binary), 'perf', '-o', 'MUL_MAT', '-b', 'CUDA0']
    with log.open('w') as stream:
        process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    if process.returncode:
        raise RuntimeError(f'bench rc={process.returncode}; inspect {log}')
    output = log.read_text()
    if 'Device 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0,' not in output:
        raise ValueError(f'expected RTX 5060 Ti / CC 12.0; inspect {log} before changing the GPU selection')
    records = [json.loads(line[len('FLATCOST '):]) for line in output.splitlines()
               if line.startswith('FLATCOST ')]
    if len(records) != 1 or not math.isfinite(records[0]['op_us']) or records[0]['op_us'] <= 0:
        raise ValueError(f'expected one positive FLATCOST record; inspect {log}')
    record = records[0]
    copies = math.ceil(64 * 1024**2 / row['weight_bytes']) + 1
    if record['copies'] != copies or record['weight_bytes'] != row['weight_bytes']:
        raise ValueError(f'weight rotation mismatch; inspect {log}')
    return record['op_us']


def run(args, manifest):
    if os.environ.get('GPUQ_ACTIVE') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('--run requires GPUQ_ACTIVE=1 and an explicit CUDA_VISIBLE_DEVICES')
    if args.binary is None or not args.binary.is_file():
        raise ValueError('--run requires --binary pointing to the in-item test-backend-ops')
    result = dict(metric='complete operation us including quantization, MMQ fixup and amortized host graph launch; no kernel-only timing',
                  repeats=args.repeats, ci='paired bootstrap percentile 95%, 2000 resamples',
                  cold='distinct initialized A/B per ring; ceil(64 MiB/weight_bytes)+1; full warm ring; CUDA graphs disabled',
                  rows=[], weighted=[])
    physical_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    all_rows = ([next(row for row in manifest['shapes'] if row['type_id'] == kind and row['m'] < 10000)
                 for kind in TYPES] if args.canary else manifest['shapes'])
    result['shapes'] = all_rows
    measured = {}
    for gpu in physical_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        for shape_id, row in enumerate(all_rows):
            cases = [(n, 1) for n in ((6,) if args.canary else WIDTHS)]
            if row['gdn'] and not args.canary:
                cases.append((6, 2))
            for n, batches in cases:
                key = (gpu, row['type_id'], row['m'], row['k'], n, batches)
                if key in measured:
                    previous = measured[key]
                    result['rows'].append(dict(previous, shape_id=shape_id, reused_from_shape_id=previous['shape_id']))
                    continue
                arms = ['default', 'mmq'] + (['mmvq'] if n <= 8 else [])
                samples = {arm: [] for arm in arms}
                aa = []
                for repeat in range(args.repeats):
                    order = arms if repeat % 2 == 0 else list(reversed(arms))
                    for seq, arm in enumerate(['default'] + order + ['default']):
                        log = args.out / f'gpu{gpu}-s{shape_id}-n{n}-b{batches}-r{repeat}-{seq}-{arm}.log'
                        value = measure(args.binary, row, n, batches, arm, log)
                        if seq == 0:
                            before = value
                        elif seq == len(order) + 1:
                            aa.append(value / before - 1)
                        else:
                            samples[arm].append(value)
                drift = ci(aa)
                valid = max(abs(value) for value in aa) <= 0.005
                entry = dict(gpu=gpu, shape_id=shape_id, n=n, batches=batches, valid=valid,
                             aa_drift=drift, arms={arm: ci(values) for arm, values in samples.items()})
                for arm, values in samples.items():
                    entry['arms'][arm]['effective_weight_gb_s'] = row['weight_bytes'] / entry['arms'][arm]['mean'] / 1000
                    entry['arms'][arm]['speedup_vs_default'] = ci([a / b for a, b in zip(samples['default'], values)])
                measured[key] = entry
                result['rows'].append(entry)
                write_json(args.out / 'results.json', result)
    indexed = {(e['gpu'], e['shape_id'], e['n'], e['batches']): e for e in result['rows']}
    for gpu in physical_gpus:
        for model in MODELS:
            group = [(i, row) for i, row in enumerate(all_rows) if row['model'] == str(model)]
            if not group:
                continue
            for n in ((6,) if args.canary else WIDTHS):
                entries = {entry['shape_id']: entry for entry in result['rows']
                           if entry['gpu'] == gpu and entry['n'] == n and entry['batches'] == 1}
                curves = {arm: [sum(row['count'] * entries[i]['arms'][arm]['samples'][r] for i, row in group)
                                for r in range(args.repeats)] for arm in entries[group[0][0]]['arms']}
                for arm, values in curves.items():
                    result['weighted'].append(dict(gpu=gpu, model=str(model), n=n, arm=arm,
                                                    op_us=ci(values), valid=all(entries[i]['valid'] for i, _ in group)))
            if not args.canary:
                # A type/shape threshold must be fixed across n, rather than choosing each width independently.
                controls = {}
                for cutoff in range(4, 9):
                    controls[cutoff] = {}
                    for n in WIDTHS:
                        controls[cutoff][n] = [sum(row['count'] * indexed[gpu, i, n, 1]['arms']['mmvq' if n <= cutoff else 'mmq']['samples'][r] for i, row in group)
                            for r in range(args.repeats)]
                summaries = {}
                defaults = {n: [sum(row['count'] * indexed[gpu, i, n, 1]['arms']['default']['samples'][r] for i, row in group) for r in range(args.repeats)]
                    for n in WIDTHS}
                def positive_increment(cutoff):
                    arm7 = 'default' if cutoff is None else ('mmvq' if 7 <= cutoff else 'mmq')
                    arm8 = 'default' if cutoff is None else ('mmvq' if 8 <= cutoff else 'mmq')
                    return sum(row['count'] * max(0,
                        indexed[gpu, i, 8, 1]['arms'][arm8]['mean'] -
                        indexed[gpu, i, 7, 1]['arms'][arm7]['mean']) for i, row in group)

                increment = positive_increment(None)
                for cutoff, curve in controls.items():
                    summaries[cutoff] = dict(flatness_t8_t1=ci([b / a for a, b in zip(curve[1], curve[8])]),
                        n6_ratio=ci([b / a for a, b in zip(defaults[6], curve[6])]),
                        positive_increment_removed=(1 - positive_increment(cutoff) / increment) if increment > 0 else None)
                write_json(args.out / f'controls-gpu{gpu}-{model.stem}.json',
                           dict(cutoffs=controls, default=defaults, summaries=summaries,
                                note='Exploratory global cutoffs; choose per-type/shape from raw cells and confirm independently.'))
    write_json(args.out / 'results.json', result)
    lines = ['| GPU | Shape | n | Batches | Arm | us (95% CI) | Weight GB/s | Valid A/A |',
             '|---|---:|---:|---:|---|---|---:|---|']
    for row in result['rows']:
        for arm, value in row['arms'].items():
            lo, hi = value['ci95']
            lines.append(f"| {row['gpu']} | {row['shape_id']} | {row['n']} | {row['batches']} | {arm} | {value['mean']:.3f} [{lo:.3f}, {hi:.3f}] | {value['effective_weight_gb_s']:.1f} | {row['valid']} |")
    lines += ['', '| GPU | Model | n | Arm | Count-weighted us (95% CI) | Valid A/A |',
              '|---|---|---:|---|---|---|']
    for row in result['weighted']:
        value = row['op_us']
        lo, hi = value['ci95']
        lines.append(f"| {row['gpu']} | {Path(row['model']).stem} | {row['n']} | {row['arm']} | {value['mean']:.3f} [{lo:.3f}, {hi:.3f}] | {row['valid']} |")
    (args.out / 'results.md').write_text('\n'.join(lines) + '\n')
    if any(not row['valid'] for row in result['rows']):
        raise RuntimeError('A/A drift exceeds 0.5%; comparison INVALID; inspect results.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true', help='read at most 200 MiB of each GGUF header; no GPU execution')
    mode.add_argument('--run', action='store_true', help='execute the GPU sweep inside gpuq')
    parser.add_argument('--out', type=Path, required=True, help='artifact directory')
    parser.add_argument('--binary', type=Path, help='in-item test-backend-ops; required with --run')
    parser.add_argument('--repeats', type=int, default=6, help='paired blocks per cell, default 6, minimum 4')
    parser.add_argument('--canary', action='store_true', help='run all parsers and A/A gates on one shape per type at n=6 on both GPUs')
    args = parser.parse_args()
    if args.repeats < 4 or (args.prepare and (args.binary or args.canary)):
        parser.error('use repeats >=4; --binary and --canary require --run')
    os.nice(19)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args.out)
    if args.run:
        frozen = json.loads((ROOT / 'evidence/shapes.json').read_text())
        if manifest != frozen:
            raise ValueError('GGUF headers changed; regenerate and review evidence/shapes.json before queueing')
        run(args, manifest)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f'flatcost: {error}', file=sys.stderr)
        sys.exit(5)
