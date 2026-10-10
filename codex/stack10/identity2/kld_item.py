"""Fixed-prefix, one-row target attention probe; execution is gpuq-only."""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[2]
WT = TREE.parents[1]
EVAL = Path('/home/xanewok/repos/qwen38-eval')
SNAPSHOT = HERE.parent / 'identity/snapshots/harness-385c353bf0ab87df00dc6b9773c80ae4827f0dec25cf821cb88c76e57d3393a9'
sys.path.insert(0, str(SNAPSHOT))
from harness import depth, lib

CORPUS = EVAL / 'recipes/_common/kld/tokens64k.bin'
MODEL = Path('/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf')
FIXTURES = EVAL / 'orchestration/gpuq/fixtures'
REAL_KLF = Path('/mnt/scratch/work/kld-s9-gdn4/out/kld-s9-gdn4/a1/canary/canary.klf')
COMMITS = {'stack9-p55': '45d90d3e4ec1b713bef3804bac5bfc93cbff94c1', 'stack10': 'a588f2a44382a361df3e2a456be0c80ca10dd1d6'}
DEVICES = ['GPU-ee9a3956-297c-604b-b4c1-f07930d27e46', 'GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f']
ARMS = {'s9p': ('stack9-p55', 512, {}), 'stock': ('stack10', 512, {}),
        'off': ('stack10', 512, {'GGML_CUDA_Q8_DECODE': '0'}), 'noise': ('stack9-p55', 256, {})}
PLAN = ['s9p', 'stock', 'off', 'noise', 'noise', 'off', 'stock', 's9p']


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def frozen(name):
    root = WT / 'frozen' / name
    bindir = root / 'build/bin'
    return dict(name=name, root=root, bindir=bindir, exe=bindir / 'round-trace', bin='build/bin',
                commit=COMMITS[name], runroot=None, pinned={}, repo=None)


def prepare(path, depths, rows):
    lib.check_input({'path': CORPUS, 'sha': '753920ad0a0dbfc135aece41284c74e96d70d856df0351100a5305511aabf412'}, 'corpus')
    seqs = depth.kldt_sequences(CORPUS)
    body = [t for seq in seqs for t in seq]
    with Path(path).open('wb') as f:
        f.write(b'KLDW' + struct.pack('<i', len(depths)))
        for d in depths:
            continuation = body[d:d + rows]
            if len(continuation) != rows:
                raise lib.Fail('corpus is too short', 'integrity')
            tokens = depth.corpus_prompt(seqs, d) + continuation
            f.write(struct.pack('<2i', d, rows) + struct.pack(f'<{len(tokens)}i', *tokens))


def read_logits(path):
    path = Path(path)
    if not 16 <= path.stat().st_size <= 200_000_000:
        raise ValueError('invalid logit file size')
    with path.open('rb') as f:
        header = f.read(16)
    if header[:4] != b'KLDL':
        raise ValueError('invalid logit magic')
    d, n, v = struct.unpack('<3i', header[4:])
    if not (0 < d <= 102400 and 0 < n <= 64 and 0 < v <= 300000) or path.stat().st_size != 16 + n*v*4:
        raise ValueError('invalid logit dimensions')
    return (d, n, v), np.memmap(path, dtype='<f4', offset=16, mode='r', shape=(n, v))


def compare_rows(p, q):
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('non-finite or mismatched logits')
    klds = []
    for a, b in zip(p, q):
        a, b = a.astype(np.float64), b.astype(np.float64)
        lp = a - a.max()
        lq = b - b.max()
        lp -= np.log(np.exp(lp).sum())
        lq -= np.log(np.exp(lq).sum())
        kl = float(np.sum(np.exp(lp)*(lp - lq)))
        if kl < -1e-12:
            raise ValueError('negative KLD beyond double rounding')
        klds.append(max(0., kl))
    return dict(mean_kld_D=float(np.mean(klds)), p99_kld_D=float(np.quantile(klds, .99)), max_kld_D=max(klds),
                top1_agreement_D=float(np.mean(p.argmax(axis=1) == q.argmax(axis=1))),
                first_position_max_abs_dlogit_D=float(np.max(np.abs(p[0].astype(np.float64) - q[0]))),
                rows_bitwise_equal_D=int(np.all(p.view('<u4') == q.view('<u4'), axis=1).sum()), rows_D=len(p))


def verdict(pairs):
    noise = pairs['s9p_to_noise']
    keys = ('mean_kld_D', 'p99_kld_D', 'max_kld_D')
    # A prefill control measures a numerical scale, not an upper bound on valid reassociation.
    limits = {k: max(noise[k], 1e-10) for k in keys}
    old_ok = all(pairs['s9p_to_off'][k] <= limits[k] for k in keys)
    stock_ok = all(pairs['s9p_to_stock'][k] <= limits[k] for k in keys)
    isolated_ok = all(pairs['off_to_stock'][k] <= limits[k] for k in keys)
    return dict(noise_limits_D=limits, old_path_within_noise=old_ok,
                verdict='confounded' if not old_ok else 'consistent_with_reassociation' if stock_ok and isolated_ok else 'suspect_bug',
                caution='Within this noise scale supports reassociation; it does not prove correctness. Above it is a follow-up trigger, not proof of a bug.')


def analyze(out, depths, plan):
    result = {}
    for d in depths:
        arrays = {}
        headers = []
        for i, name in enumerate(plan):
            h, values = read_logits(out / f'{i}-{name}/depth-{d}.logits')
            if h[0] != d:
                raise lib.Fail('wrong depth in dump', 'integrity')
            headers.append(h)
            if name in arrays and not np.array_equal(arrays[name].view('<u4'), values.view('<u4')):
                raise lib.Fail('same-arm logits do not repeat', 'integrity')
            arrays[name] = values
        if len(set(headers)) != 1:
            raise lib.Fail('dump shapes differ', 'integrity')
        pairs = {f'{a}_to_{b}': compare_rows(arrays[a], arrays[b]) for a, b in itertools.permutations(arrays, 2)}
        if len(arrays) == 1:
            values = next(iter(arrays.values()))
            zero = compare_rows(values, values)
            changed = values.copy()
            changed[0, int(values[0].argmax())] += 1
            mutation = compare_rows(values, changed)
            control = verdict({'s9p_to_noise': zero, 's9p_to_off': zero, 's9p_to_stock': mutation, 'off_to_stock': mutation})
            if control['verdict'] != 'suspect_bug':
                raise lib.Fail('canary numerical negative control missed', 'integrity')
            pairs['self'] = zero
            pairs['synthetic_negative_control'] = mutation
        result[str(d)] = dict(shape_D=headers[0], pairs=pairs, **(verdict(pairs) if len(arrays) == 4 else {}))
    dump(out / 'SUMMARY.json', result)
    return result


def parser_tests():
    # This is an actual GPU dump, below the per-file read limit. Its log probabilities are also valid logits.
    if REAL_KLF.stat().st_size > 200_000_000:
        raise ValueError('real fixture exceeds read limit')
    with REAL_KLF.open('rb') as f:
        if f.read(4) != b'KLF1':
            raise ValueError('bad real fixture')
        v, count = struct.unpack('<2i', f.read(8))
        start, n = struct.unpack('<2i', f.read(8))
        if not (0 < v <= 300000 and count == 1 and n >= 2):
            raise ValueError('bad real fixture dimensions')
        rows = np.frombuffer(f.read(2*v*4), '<f4').reshape(2, v).copy()
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        p = Path(tmp) / 'real.logits'
        p.write_bytes(b'KLDL' + struct.pack('<3i', start, 2, v) + rows.tobytes())
        _, loaded = read_logits(p)
        assert compare_rows(loaded, loaded)['max_kld_D'] == 0
        changed = rows.copy()
        changed[0, int(rows[0].argmax())] += 1
        metrics = compare_rows(rows, changed)
        assert metrics['max_kld_D'] > 1e-10 and metrics['first_position_max_abs_dlogit_D'] > .99
        zero = compare_rows(rows, rows)
        assert verdict({'s9p_to_noise': zero, 's9p_to_off': zero, 's9p_to_stock': metrics, 'off_to_stock': metrics})['verdict'] == 'suspect_bug'
        bad = rows.copy()
        bad[0, 0] = np.nan
        try:
            compare_rows(rows, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('non-finite negative control missed')
        p.write_bytes(p.read_bytes()[:-4])
        try:
            read_logits(p)
        except ValueError:
            pass
        else:
            raise AssertionError('truncation negative control missed')
        out = Path(tmp) / 'analysis'
        out.mkdir()
        for i, name in enumerate(PLAN):
            dest = out / f'{i}-{name}'
            dest.mkdir()
            (dest / f'depth-{start}.logits').write_bytes(b'KLDL' + struct.pack('<3i', start, 2, v) + rows.tobytes())
        assert analyze(out, [start], PLAN)[str(start)]['verdict'] == 'consistent_with_reassociation'
        (out / f'7-s9p/depth-{start}.logits').write_bytes(b'KLDL' + struct.pack('<3i', start, 2, v) + changed.tobytes())
        try:
            analyze(out, [start], PLAN)
        except lib.Fail as error:
            assert error.rc == 6
        else:
            raise AssertionError('repeat identity negative control missed')
        p = Path(tmp) / 'journal.txt'
        assert lib.record_journal(0, p, FIXTURES / 'xid-lines.txt') > 0
        assert lib.gpu_memory((FIXTURES / 'nvidia-smi-query.csv').read_text(), DEVICES) >= 0
    real_log = (EVAL / 'results/2026-10-09-s10-depth-iq4/canary-a3/0-s10/server.log').read_text()
    lib.verify_log_patterns(real_log, [r'fattn_q8_decode:.*dk=256.*gqa=6.*seqs=1', r'mtp_fold_decode: engaged n_seq=1', r'persistent rotations backend=',
                                      r'one pinned copy per input, uploads slowest host link first'], 'recorded engagement')
    return dict(real_fixture=str(REAL_KLF), real_fixture_sha=lib.file_sha(REAL_KLF), rows_D=2, vocab_D=v,
                mutation_max_kld_D=metrics['max_kld_D'])


def environment(bindir, inp, out, extra, canary):
    return dict(HOME=os.environ['HOME'], PATH='/usr/local/bin:/usr/bin:/bin', CUDA_HOME='/usr/local/cuda-13.2',
                CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(DEVICES), GIT_NO_LAZY_FETCH='1',
                OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LD_LIBRARY_PATH=str(bindir),
                GGML_SCHED_INPUT_STAGING='1', LLAMA_ARG_KV_PAD='4096', LLAMA_MTP_CHAIN='1', LLAMA_MTP_FOLD_C='1',
                LLAMA_ARG_LOG_VERBOSITY='5' if canary else '1', S10_KLD_INPUT=str(inp), S10_KLD_OUT=str(out), **extra)


def build(out, name, remaining):
    target = out / ('build-' + name)
    target.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TMPDIR='/mnt/scratch/tmp-build', GGML_CCACHE='ON',
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    gate = str(WT / 'bin/buildgate')
    commands = [[gate, 'cmake', '-S', str(HERE), '-B', str(target), '-DCMAKE_BUILD_TYPE=Release',
                 '-DENGINE=' + str(TREE), '-DFROZEN=' + str(frozen(name)['bindir']),
                 '-DGGML_CCACHE=ON', '-DGGML_CUDA=OFF', '-DCUDAToolkit_ROOT=/usr/local/cuda-13.2', '-DGGML_CUDA_NCCL=OFF'],
                [gate, 'cmake', '--build', str(target), '-j4']]
    with (target / 'build.log').open('w') as log:
        for cmd in commands:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=min(120, remaining()), check=True)
    return target / 's10-kld'


def execute(args):
    if os.environ.get('GPUQ_ACTIVE') != '1':
        raise lib.Fail('execution requires gpuq', 'deterministic')
    out = args.out
    if out.exists():
        raise lib.Fail('output dir is not fresh', 'deterministic')
    out.mkdir(parents=True)
    deadline = time.monotonic() + args.budget_s
    def interrupted(signum, frame):
        raise lib.Fail('budget or queue interruption', 'timeout')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise lib.Fail('budget expired', 'timeout')
        return value
    canary = args.mode == 'canary'
    depths = [8192] if canary else [8192, 32768, 90112]
    plan = ['stock'] if canary else PLAN
    rows = 4 if canary else 64
    inp = out / 'input.bin'
    prepare(inp, depths, rows)
    binaries = {}
    pins = {name: lib.check_frozen(frozen(name)) for name in dict.fromkeys(ARMS[a][0] for a in plan)}
    if MODEL.stat().st_size != 15475951296:
        raise lib.Fail('model size changed', 'integrity')
    dump(out / 'plan.json', dict(plan=plan, depths=depths, rows=rows, corpus_sha=lib.file_sha(CORPUS), frozen=pins))
    lib.event('measure_start')
    for i, arm in enumerate(plan):
        name, ub, extra = ARMS[arm]
        if name not in binaries:
            binaries[name] = build(out, name, remaining)
        if i:
            lib.wait_gpu_idle(DEVICES, wait_s=min(90, remaining()))
        lib.check_frozen(frozen(name))
        dest = out / f'{i}-{arm}'
        dest.mkdir()
        env = environment(frozen(name)['bindir'], inp, dest, extra, canary)
        cmd = [str(binaries[name]), '-m', str(MODEL), '-c', '102400', '-np', '1', '-ctk', 'q8_0', '-ctv', 'q8_0',
               '-ngl', '99', '-fa', 'on', '-sm', 'tensor', '--tensor-split', '50,50', '-b', '2048', '-ub', str(ub), '-t', '8', '-tb', '8']
        dump(dest / 'command.json', cmd)
        dump(dest / 'env.json', env)
        started = time.time()
        proc = None
        timed_out = False
        try:
            with (dest / 'tool.log').open('w') as log:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        preexec_fn=lambda: lib._own_group(os.getppid()))
                (dest / 'pid').write_text(str(proc.pid))
                arm_deadline = time.monotonic() + min(600, remaining())
                checked_maps = False
                while proc.poll() is None:
                    if time.monotonic() > arm_deadline:
                        timed_out = True
                        break
                    if not checked_maps:
                        maps = Path(f'/proc/{proc.pid}/maps')
                        if maps.exists() and 'libllama' in maps.read_text():
                            outside = lib.libs_outside(proc.pid, frozen(name)['root'])
                            if outside:
                                raise lib.Fail('libraries outside frozen root: ' + ', '.join(outside), 'integrity')
                            checked_maps = True
                    time.sleep(.2)
                rc = proc.poll()
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            faults = lib.record_journal(int(started), dest / 'journal.txt')
        if faults is None:
            raise lib.Fail('kernel journal unreadable', 'integrity')
        if faults:
            raise lib.Fail('GPU fault since arm start', 'hazard')
        if timed_out:
            raise lib.Fail('budget or arm timeout', 'timeout')
        if rc:
            raise lib.Fail(f'logit tool exited {rc}', 'deterministic')
        text = (dest / 'tool.log').read_text()
        for d in depths:
            h, values = read_logits(dest / f'depth-{d}.logits')
            if h != (d, rows, 248320) or not np.isfinite(values).all():
                raise lib.Fail('invalid logit dump', 'integrity')
        if not checked_maps:
            raise lib.Fail('library provenance was not observed', 'integrity')
        if canary:
            lib.verify_log_patterns(text, [r'fattn_q8_decode:.*dk=256.*gqa=6.*seqs=1', r'mtp_fold_decode: engaged n_seq=1', r'persistent rotations backend=',
                                          r'one pinned copy per input, uploads slowest host link first'], 'one-row q8 and served inputs')
        lib.check_frozen(frozen(name))
    analyze(out, depths, plan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['preflight', 'canary', 'run'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--budget-s', type=int, default=5800)
    args = p.parse_args()
    try:
        if args.mode == 'preflight':
            result = parser_tests()
            args.out.mkdir(parents=True, exist_ok=True)
            prepare(args.out / 'input.bin', [8192, 32768, 90112], 64)
            dump(args.out / 'PREFLIGHT.json', result)
            print(json.dumps(result))
        else:
            execute(args)
        return 0
    except lib.Fail as error:
        print(str(error), file=sys.stderr)
        return error.rc
    except subprocess.TimeoutExpired:
        print('budget expired during build', file=sys.stderr)
        return 7
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 8


if __name__ == '__main__':
    sys.exit(main())
