#!/usr/bin/env python3
"""Queue-only extraction, not a GPU speed benchmark. Preflight never runs binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from import_streams import recorded_stream


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/home/xanewok/repos/qwen38-eval-worktrees/frozen/spec-tf-v2')
DATA = Path('/home/xanewok/repos/qwen38-eval/results/2026-10-06-nh6-nextn-lowbit-j5q/a2/0-base')
BASE = Path('/home/xanewok/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf')
MODELS = [BASE, Path('/mnt/scratch/work/models/nh6/c/J5q-nh6-c-00001-of-00002.gguf'),
          Path('/mnt/scratch/work/models/nh6/d/J5q-nh6-d-00001-of-00002.gguf'),
          Path('/mnt/scratch/work/models/nh6/a/J5q-nh6-a-00001-of-00002.gguf')]
VOCAB = Path('/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json')
COV99 = Path('/home/xanewok/repos/qwen38-eval-worktrees/work/draft-vocab-cov/swift_draft_vocab_ids.cov99.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def journal_fault(lines, pid):
    faults = [line for line in lines.splitlines() if re.search(r'\bXid\b|fallen off the bus', line)]
    if not faults:
        return 0
    # An application fault in this payload is deterministic, never retry-as-transient.
    own = all(re.search(r'\bXid\s*\([^)]*\):\s*(13|31|43|45),', line) and
              re.search(rf'\bpid={pid}\b', line) for line in faults)
    return 8 if own else 3


def check_frozen():
    source = (FROZEN / 'SOURCE').read_text().splitlines()[0]
    sums = FROZEN / 'build/bin/SHA256SUMS'
    for line in sums.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(sums.parent / name) != expected:
            raise ValueError('frozen bytes changed')
    if not (sums.parent / 'round-trace').exists():
        raise ValueError('missing frozen teacher-forced executable')
    symbols = subprocess.run(['strings', str(sums.parent / 'libggml-base.so')],
                             check=True, capture_output=True, text=True)
    if source[:9] not in symbols.stdout.splitlines():
        raise ValueError('ggml source stamp mismatch')
    return source


def preflight():
    check_frozen()
    pins = json.loads((ROOT / 'codex/gpu_inputs.json').read_text())
    cache = {}
    for name, expected in pins.items():
        path = Path(name)
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size)
        if key not in cache:
            cache[key] = sha(path)
        if cache[key] != expected:
            raise ValueError('inputs changed: ' + name)
    stream = recorded_stream(DATA / 'depth-1500.json')
    if len(stream['continuation']) != 1536:
        raise ValueError('recorded artifact has insufficient data')
    symbols = subprocess.run(['strings', str(FROZEN / 'build/bin/libggml-cuda.so')], check=True, capture_output=True, text=True)
    if sum('DeviceTopK' in s for s in symbols.stdout.splitlines()) != 25:
        raise ValueError('DeviceTopK stamp mismatch')
    print('preflight passed; no GPU binary executed', flush=True)


def run(args):
    if os.environ.get('GPUQ_ITEM') != 'spec-tf-j5q':
        raise ValueError('GPU payload requires gpuq; do not execute manually')
    check_frozen()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    since = time.time()
    deadline = started + args.budget
    exe = FROZEN / 'build/bin/round-trace'
    env = dict(os.environ, GGML_SCHED_INPUT_STAGING='1', LLAMA_ARG_LOG_VERBOSITY='4',
               LD_LIBRARY_PATH=str(exe.parent), OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    streams = [recorded_stream(DATA / f'depth-{depth}.json') for depth in range(1500, 19001, 500)]
    if args.mode == 'canary':
        streams = streams[:1]
        streams[0]['continuation'] = streams[0]['continuation'][:16]
    stream_file = args.out / 'streams.json'
    stream_file.write_text(json.dumps(streams) + '\n')
    flags = ['-c', '102400', '-np', '1', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
             '-sm', 'tensor', '--tensor-split', '50,50', '-ngl', '99', '-b', '2048', '-ub', '512', '--kv-pad', '4096',
             '--spec-type', 'draft-mtp', '--spec-draft-n-max', '5', '--spec-draft-n-min', '0', '--spec-draft-p-min', '0',
             '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling', '--backend-sampling',
             '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16', '-t', '8', '-tb', '8',
             '--temp', '1', '--top-k', '20', '--top-p', '0.95', '--min-p', '0.05', '--ignore-eos',
             '--tf-chunk', '6', '--tf-carry-streams']
    events = os.environ.get('GPUQ_EVENTS')
    if events:
        with open(events, 'a') as f:
            f.write('measure_start\n')

    def stage(name, cmd):
        remain = deadline - time.monotonic()
        if remain <= 30:
            raise TimeoutError('budget exhausted before ' + name)
        (args.out / (name + '.command.json')).write_text(json.dumps(cmd))
        with (args.out / (name + '.log')).open('w') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
            (args.out / (name + '.pid')).write_text(str(proc.pid))
            try:
                rc = proc.wait(timeout=remain - 15)
            except subprocess.TimeoutExpired:
                raise TimeoutError('budget exhausted in ' + name)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            journal = subprocess.run(['journalctl', '-k', '-b', '--since', f'@{since}', '--no-pager'], capture_output=True, text=True, check=True)
            (args.out / (name + '.journal.txt')).write_text(journal.stdout)
            fault = journal_fault(journal.stdout, proc.pid)
            (args.out / (name + '.verdict.json')).write_text(json.dumps({'returncode': rc, 'fault_rc': fault}))
            if fault:
                raise PayloadExit(fault, 'kernel fault during ' + name)
            if rc:
                raise PayloadExit(8, 'payload failed: ' + name)

    target = args.out / 'target'
    names = ['base', 'q3_k', 'iq3_s', 'q8_0']
    if args.mode == 'canary':
        stage('canary', [str(exe), *flags, '-m', str(BASE), '--tf-stage', 'AB', '--tf-streams', str(stream_file),
                         '--tf-out', str(args.out), '--tf-control-vocab', str(VOCAB), '--tf-control-vocab', str(COV99)])
        (args.out / 'control-0').rename(args.out / 'base')
        (args.out / 'control-1').rename(args.out / 'cov99')
    else:
        stage('target', [str(exe), *flags, '-m', str(BASE), '--tf-stage', 'A', '--tf-streams', str(stream_file), '--tf-out', str(target)])
        for name, model in zip(names, MODELS):
            stage(name, [str(exe), *flags, '-m', str(model), '--spec-draft-vocab', str(VOCAB),
                         '--tf-stage', 'B', '--tf-target', str(target), '--tf-out', str(args.out / name)])
        stage('full', [str(exe), *flags, '-m', str(BASE), '--tf-stage', 'B', '--tf-target', str(target), '--tf-out', str(args.out / 'full')])
        stage('cov99', [str(exe), *flags, '-m', str(BASE), '--spec-draft-vocab', str(COV99),
                        '--tf-stage', 'B', '--tf-target', str(target), '--tf-out', str(args.out / 'cov99')])
    stage('control', ['python3', str(ROOT / 'codex/check_vocab.py'), '--target', str(target), '--full', str(args.out / 'full'),
                      '--sliced', str(args.out / 'base'), '--vocab', str(VOCAB), '--output', str(args.out / 'control-base.json')])
    stage('control-cov99', ['python3', str(ROOT / 'codex/check_vocab.py'), '--target', str(target), '--full', str(args.out / 'full'),
                            '--sliced', str(args.out / 'cov99'), '--vocab', str(COV99), '--output', str(args.out / 'control-cov99.json')])
    arms = ['base'] if args.mode == 'canary' else names
    cmd = ['python3', str(ROOT / 'codex/spec_tf.py'), '--target', str(target), '--output', str(args.out / 'estimate')]
    for name in arms + ['cov99', 'full']:
        cmd += ['--arm', str(args.out / name)]
    cmd += ['--spec-draft-vocab', str(VOCAB), '--spec-draft-vocab', str(COV99)]
    stage('estimate', cmd)
    (args.out / 'complete.json').write_text(json.dumps({'elapsed_s_M': time.monotonic() - started, 'absolute_label': 'G'}))


class PayloadExit(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['preflight', 'canary', 'run'], required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--budget', type=int, default=14400)
    args = p.parse_args()
    def stopped(signum, frame):
        raise TimeoutError('queue stop or budget termination')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        if args.mode == 'preflight':
            preflight()
        else:
            run(args)
        return 0
    except PayloadExit as e:
        print(str(e), flush=True)
        return e.code
    except TimeoutError as e:
        print(str(e), flush=True)
        return 7
    except ValueError as e:
        print(str(e), flush=True)
        return 6
    except Exception as e:
        print(f'payload error: {e}', flush=True)
        return 8


if __name__ == '__main__':
    raise SystemExit(main())
