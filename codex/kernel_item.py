"""GPU queue only: wide MMVQ memcheck and interleaved frozen-library perf probes."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/xanewok/repos/qwen38-eval')
from harness import lib
UP = Path('/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack6')
PATCH = HERE / 'frozen' / 'mmvq-np2'
GPU = ['GPU-ee9a3956-297c-604b-b4c1-f07930d27e46', 'GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f']


def sql_rows(text):
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    try:
        db.executescript(text)
        return [dict(r) for r in db.execute('SELECT * FROM test_backend_ops WHERE op_name = "MUL_MAT"')]
    finally:
        db.close()


def case_name(row):
    params = row['op_params']
    name = re.search(r'name=([^,]+)', params)
    if name:
        return name.group(1)
    shape = re.search(r'type_a=([^,]+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),', params)
    if not shape:
        raise lib.Fail('probe: integrity: unknown MUL_MAT params', 'integrity')
    quant, m, n, k = shape.groups()
    return f'mmvq-wide:{quant}:{m}:{k}:{n}'


def check_rows(rows, expected, perf=False, stamp=None):
    names = [case_name(r) for r in rows]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise lib.Fail('probe: integrity: missing, duplicate or unexpected cases', 'integrity')
    for row in rows:
        if not row['supported'] or not row['passed']:
            raise lib.Fail('probe: correctness failure: ' + row['op_params'], 'deterministic')
        if stamp and row['build_commit'] != stamp:
            raise lib.Fail('probe: integrity: ggml stamp mismatch', 'integrity')
        if perf and (not math.isfinite(row['time_us']) or row['time_us'] <= 0 or row['n_runs'] < 1):
            raise lib.Fail('probe: integrity: invalid perf timing', 'integrity')


def check_sanitizer(text):
    errors = re.findall(r'ERROR SUMMARY: (\d+) errors?', text)
    if 'COMPUTE-SANITIZER' not in text or not errors:
        raise lib.Fail('probe: integrity: missing sanitizer evidence', 'integrity')
    if any(int(n) for n in errors):
        raise lib.Fail('probe: sanitizer finding', 'deterministic')


def journal(since, out):
    result = subprocess.run(['journalctl', '-k', '--no-pager', '--since', f'@{int(since)}'],
                            capture_output=True, text=True, timeout=10)
    out.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise lib.Fail('probe: integrity: kernel journal unavailable', 'integrity')
    if re.search(lib.XID_RE, result.stdout):
        raise lib.Fail('probe: GPU hazard', 'hazard')


def selftest():
    rows = sql_rows((HERE / 'fixtures/perf-real.sql').read_text())
    names = [case_name(r) for r in rows]
    check_rows(rows, names, perf=True)
    for broken in (rows[:-1], rows + rows[:1], [dict(r, time_us=-1) for r in rows], [dict(r, passed=0) for r in rows]):
        try:
            check_rows(broken, names, perf=True)
        except lib.Fail:
            pass
        else:
            raise AssertionError('perf negative control escaped')
    san = (HERE / 'fixtures/memcheck-real.txt').read_text()
    check_sanitizer(san)
    for broken in (san.replace('ERROR SUMMARY: 0 errors', 'ERROR SUMMARY: 1 errors'), ''):
        try:
            check_sanitizer(broken)
        except lib.Fail:
            pass
        else:
            raise AssertionError('sanitizer negative control escaped')
    assert re.search(lib.XID_RE, (HERE / 'fixtures/journal-real.txt').read_text())
    assert not re.search(lib.XID_RE, 'stop_xid_hash_s_G')
    native = sql_rows((HERE / 'fixtures/cpu-registered-real.sql').read_text())
    check_rows(native, [f'mmvq-wide:{q}:512:5120:{n}' for q in ('iq4_xs', 'q5_K', 'q6_K', 'q8_0', 'q4_K') for n in range(9, 17)])
    print('probe: real-artifact parser controls PASS', flush=True)


def preflight():
    selftest()
    for root in (UP, PATCH):
        subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=root / 'build/bin', check=True,
                       stdout=subprocess.DEVNULL)
    if not (PATCH / 'build/bin/test-backend-ops').is_file():
        raise lib.Fail('probe: integrity: missing test executable', 'integrity')


def run(kind, out, canary, budget):
    if os.environ.get('GPUQ_ACTIVE') != '1':
        raise lib.Fail('probe: integrity: queue environment missing', 'integrity')
    out.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'console.log' for p in out.iterdir()):
        raise lib.Fail('probe: integrity: output is not fresh', 'integrity')
    preflight()
    if kind == 'perf':
        stamp = (PATCH / 'SOURCE').read_text().splitlines()[0]
        evidence = []
        for path in (HERE / 'gpu-results/memcheck').glob('full-a*/PASS.json'):
            record = json.loads(path.read_text())
            if record.get('kind') == 'memcheck' and record.get('canary') is False and record.get('frozen_commit') == stamp:
                evidence.append(path)
        if not evidence:
            raise lib.Fail('probe: integrity: clean full memcheck evidence missing', 'integrity')
    began = time.time()
    deadline = time.monotonic() + budget
    widths = [9, 12, 16] if canary else list(range(9, 17)) if kind == 'memcheck' else [6, 12]
    shapes = [(512, 5120)] if canary else [(8704, 5120), (5120, 8704), (6144, 5120), (512, 5120), (5120, 3072), (5120, 5120), (3072, 5120)]
    expected = [f'mmvq-wide:{t}:{m}:{k}:{n}' for t in ('iq4_xs', 'q5_K', 'q6_K', 'q8_0', 'q4_K') for m, k in shapes for n in widths]
    width_regex = '(' + '|'.join(map(str, widths)) + ')'
    shape_regex = '(' + '|'.join(f'm={m},n={width_regex},k={k},' for m, k in shapes) + ')'
    params = r'type_a=(iq4_xs|q5_K|q6_K|q8_0|q4_K),type_b=f32,' + shape_regex + r'bs=\[1,1\],nr=\[1,1\]'
    modes = ['wide', 'split'] if kind == 'memcheck' else ['up', 'wide', 'split', 'split', 'wide', 'up']
    if canary:
        modes = ['wide'] if kind == 'memcheck' else ['up', 'wide', 'split']
    devices = GPU[:1] if canary or kind == 'memcheck' else GPU
    records = []
    for dev in devices:
        for i, mode in enumerate(modes):
            root = UP if mode == 'up' else PATCH
            stamp = (root / 'SOURCE').read_text().splitlines()[0][:9]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, LD_LIBRARY_PATH=str(root / 'build/bin'),
                       OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8',
                       LD_PRELOAD=str(PATCH / 'build/bin/libcpu-limit.so'))
            env.pop('GGML_CUDA_MMVQ_WIDE', None)
            env.pop('GGML_CUDA_MMVQ_WIDE_SPLIT', None)
            env.pop('GGML_CUDA_DEBUG_MMVQ_WIDE', None)
            if mode != 'up':
                env.update(GGML_CUDA_MMVQ_WIDE='all', GGML_CUDA_DEBUG_MMVQ_WIDE='1',
                           GGML_CUDA_MMVQ_WIDE_SPLIT='1' if mode == 'split' else '0')
            command = ['nice', '-n', '19', str(PATCH / 'build/bin/test-backend-ops'), 'test' if kind == 'memcheck' else 'perf',
                       '-b', 'CUDA0', '-o', 'MUL_MAT', '-p', params, '--output', 'sql']
            if kind == 'memcheck':
                command = ['nice', '-n', '19', '/usr/local/cuda-13.2/bin/compute-sanitizer', '--tool', 'memcheck',
                           '--error-exitcode', '99', '--print-limit', '50',
                           '--log-file', str(out / (f'{dev[-4:]}-{i}-{mode}' + '.sanitizer.txt'))] + command[3:]
            stem = f'{dev[-4:]}-{i}-{mode}'
            journal(began, out / 'journal.txt')
            remaining = deadline - time.monotonic() - 15
            if remaining <= 0:
                raise lib.Fail('probe: budget expired', 'timeout')
            (out / (stem + '.command.json')).write_text(json.dumps(command))
            with (out / (stem + '.sql')).open('w') as stdout, (out / (stem + '.log')).open('w') as stderr:
                process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                try:
                    rc = process.wait(timeout=min(600 if kind == 'memcheck' else 250, remaining))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise lib.Fail('probe: budget stage cap', 'timeout')
                finally:
                    lib.stop_child(process, 10)
            journal(began, out / 'journal.txt')
            log = (out / (stem + '.log')).read_text()
            if rc:
                raise lib.Fail(f'probe: process rc {rc}', 'deterministic')
            rows = sql_rows((out / (stem + '.sql')).read_text())
            check_rows(rows, expected, perf=kind == 'perf', stamp=stamp)
            if kind == 'memcheck':
                # The sanitizer report goes to --log-file: its default stdout is the SQL stream.
                check_sanitizer((out / (stem + '.sanitizer.txt')).read_text())
            if mode != 'up':
                for quant in ('iq4_xs', 'q5_K', 'q6_K', 'q8_0', 'q4_K'):
                    for width in widths:
                        if width <= 8:
                            continue
                        signature = f'mmvq_wide: type={quant} columns={width} mode={mode}'
                        if signature not in log:
                            raise lib.Fail('probe: integrity: wide path did not engage: ' + signature, 'integrity')
            records.append(dict(device=dev, mode=mode, rows=rows))
            (out / 'results.json').write_text(json.dumps(records, indent=2))
    (out / 'PASS.json').write_text(json.dumps(dict(kind=kind, canary=canary, frozen_commit=(PATCH / 'SOURCE').read_text().splitlines()[0], wall_s_M=time.time()-began)))
    print('probe: PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['memcheck', 'perf'])
    parser.add_argument('--stage', choices=['selftest', 'preflight', 'canary', 'run'], required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'kernel-results')
    parser.add_argument('--budget', type=int, default=1200)
    args = parser.parse_args()
    try:
        if args.stage == 'selftest':
            selftest()
        elif args.stage == 'preflight':
            preflight()
        else:
            run(args.kind, args.out, args.stage == 'canary', args.budget)
        return 0
    except lib.Fail as error:
        print(error, file=sys.stderr)
        return error.rc

if __name__ == '__main__':
    sys.exit(main())
