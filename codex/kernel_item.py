"""GPU queue only: small-J MMQ reference, safe negative control, and hot/cold probes."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
sys.path.insert(0, str(EVAL))
from harness import lib

ROOTS = {name: HERE / 'frozen' / ('mmq-smallj' + suffix)
         for name, suffix in [('patch', ''), ('negative', '-negative')]}
ROOTS['base'] = Path('/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack6')
GPU = 'GPU-ee9a3956-297c-604b-b4c1-f07930d27e46'
TYPES = ('iq4_xs', 'q6_K', 'q5_K', 'q8_0', 'q4_K')
SHAPES = ((8704, 5120), (5120, 8704), (5120, 6144), (3072, 5120), (6144, 5120), (5120, 3072), (5120, 5120))
WIDTHS = (6, 8, 9, 12, 16, 17, 24)


def sql_rows(content):
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    try:
        # Reference failures print diagnostics before INSERT even with --output sql.
        statements = re.findall(r'CREATE TABLE IF NOT EXISTS test_backend_ops \([\s\S]*?\);|INSERT INTO test_backend_ops [^\n]*;', content)
        db.executescript('\n'.join(statements))
        return [dict(row) for row in db.execute('SELECT * FROM test_backend_ops WHERE op_name IN ("MUL_MAT", "MUL_MAT_ID")')]
    except sqlite3.Error as error:
        raise lib.Fail('probe: integrity: unreadable SQL: ' + str(error), 'integrity') from error
    finally:
        db.close()


def case_name(row):
    params = row['op_params']
    match = re.search(r'type_a=([^,]+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),bs=\[1,1\],nr=\[1,1\]', params)
    if match:
        quant, m, n, k = match.groups()
        return f'{quant}:{m}:{k}:{n}'
    match = re.search(r'type_a=([^,]+),type_b=f32,n_mats=4,n_used=2,b=(0|1),m=(\d+),n=(\d+),k=(\d+),', params)
    if match:
        quant, broadcast, m, n, k = match.groups()
        return f'id:{quant}:{m}:{k}:{n}:{broadcast}'
    match = re.search(r'name=([^,]+)', params)
    return match.group(1) if match else None


def check_rows(rows, expected, perf=False, stamp=None, negative=False):
    known = [row for row in rows if case_name(row) is not None]
    names = [case_name(row) for row in known]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise lib.Fail('probe: integrity: missing, duplicate or unexpected cases', 'integrity')
    for row in known:
        if not row['supported'] or bool(row['passed']) == negative:
            raise lib.Fail('probe: reference gate failed: ' + row['op_params'], 'deterministic')
        if stamp and row['build_commit'] != stamp:
            raise lib.Fail('probe: integrity: ggml stamp mismatch', 'integrity')
        if negative and row['error_message'] != 'test failed':
            raise lib.Fail('probe: integrity: negative control failed outside the reference comparison', 'integrity')
        if perf and (not math.isfinite(row['time_us']) or row['time_us'] <= 0 or row['n_runs'] < 1):
            raise lib.Fail('probe: integrity: invalid perf timing', 'integrity')


def check_sanitizer(content):
    summaries = re.findall(r'ERROR SUMMARY: (\d+) errors?', content)
    if 'COMPUTE-SANITIZER' not in content or not summaries:
        raise lib.Fail('probe: integrity: missing sanitizer evidence', 'integrity')
    if any(int(value) for value in summaries):
        raise lib.Fail('probe: sanitizer finding', 'deterministic')


def perf_summary(records):
    samples = {}
    for record in records:
        for row in record['rows']:
            key = ('cold:' if record['cold'] else 'hot:') + case_name(row)
            samples.setdefault(key, {}).setdefault(record['mode'], []).append(row['time_us'])
    result = {}
    for key, modes in samples.items():
        result[key] = {mode + '_us_M': statistics.median(values) for mode, values in modes.items()}
        if 'base' in modes and 'patch' in modes:
            result[key]['delta_us_D'] = result[key]['patch_us_M'] - result[key]['base_us_M']
    return result


def journal(since, out):
    result = subprocess.run(['journalctl', '-k', '--no-pager', '--since', f'@{int(since)}'],
                            capture_output=True, text=True, timeout=10)
    out.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise lib.Fail('probe: integrity: kernel journal unavailable', 'integrity')
    if re.search(lib.XID_RE, result.stdout):
        raise lib.Fail('probe: GPU hazard', 'hazard')


def library_gate(process, root, out):
    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        maps = Path(f'/proc/{process.pid}/maps').read_text()
        if '/libggml-cuda.so' in maps:
            outside = lib.libs_outside(process.pid, root)
            if outside:
                raise lib.Fail('probe: integrity: libraries outside frozen root: ' + ', '.join(outside), 'integrity')
            out.write_text('\n'.join(line for line in maps.splitlines() if '/libggml' in line))
            return
        time.sleep(0.02)
    raise lib.Fail('probe: integrity: loaded CUDA library proof missing', 'integrity')


def selftest():
    for filename in ('perf-real.sql', 'stock-real.sql'):
        rows = sql_rows((HERE / 'fixtures' / filename).read_text())
        names = [case_name(row) for row in rows if case_name(row)]
        assert names
        check_rows(rows, names, perf=True)
        summary = perf_summary([dict(mode=mode, cold=False, rows=rows) for mode in ('base', 'patch', 'patch', 'base')])
        assert all(item['delta_us_D'] == 0 for item in summary.values())
        assert sql_rows((HERE / 'fixtures' / filename).read_text().replace('INSERT INTO', '[MUL_MAT] ERR = 4.0 > 0.0005 INSERT INTO')) == rows
        for broken in (rows[:-1], rows + rows[:1], [dict(row, time_us=-1) for row in rows],
                       [dict(row, passed=0) for row in rows]):
            try:
                check_rows(broken, names, perf=True)
            except lib.Fail:
                pass
            else:
                raise AssertionError('SQL negative control escaped')
    san = (HERE / 'fixtures/memcheck-real.txt').read_text()
    check_sanitizer(san)
    for broken in ('', san.replace('ERROR SUMMARY: 0 errors', 'ERROR SUMMARY: 1 errors')):
        try:
            check_sanitizer(broken)
        except lib.Fail:
            pass
        else:
            raise AssertionError('sanitizer negative control escaped')
    assert re.search(lib.XID_RE, (HERE / 'fixtures/journal-real.txt').read_text())
    assert not re.search(lib.XID_RE, 'stop_xid_hash_s_G')
    from harness import depth
    depth.fixture_test(EVAL / 'orchestration/gpuq/fixtures')
    record = json.loads((HERE / 'fixtures/duo-depth-real.json').read_text())
    assert len(record['prefill']) == len(record['streams']) == 2
    for prefill in record['prefill']:
        depth.first_prefill(8192)(depth.final_timings(prefill))
    window = depth.overlap(record['streams'], 0.8)
    for stream in record['streams']:
        lib.steady_intervals(stream, 1, window=tuple(window['window_ns']))
        lib.token_identity(stream)
    print('probe: recorded-artifact parser controls PASS', flush=True)


def preflight(kind):
    selftest()
    for name in ('patch', 'base', 'negative') if kind == 'memcheck' else ('patch', 'base'):
        root = ROOTS[name]
        subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=root / 'build/bin', check=True,
                       stdout=subprocess.DEVNULL)
        if name != 'base' and not (root / 'build/bin/test-backend-ops').is_file():
            raise lib.Fail('probe: integrity: missing test executable', 'integrity')
        if not (root / 'RECIPE.txt').is_file():
            raise lib.Fail('probe: integrity: missing compile-switch provenance', 'integrity')


def require_memcheck():
    stamp = (ROOTS['patch'] / 'SOURCE').read_text().splitlines()[0]
    digest = lib.file_sha(ROOTS['patch'] / 'build/bin/SHA256SUMS')
    for path in (HERE / 'gpu-results/memcheck').glob('full-a*/PASS.json'):
        record = json.loads(path.read_text())
        if (record.get('kind') == 'memcheck' and record.get('canary') is False and
                record.get('frozen_commit') == stamp and record.get('binary_digest') == digest):
            return
    raise lib.Fail('probe: integrity: clean full memcheck and negative-control evidence missing', 'integrity')


def expected_cases(kind, canary, negative=False):
    shapes = ((65, 5120),) if negative else ((3072, 5120),) if canary else SHAPES
    if kind == 'memcheck' and not negative:
        shapes += ((65, 5120), (128, 5120))
    widths = (12,) if negative else (9, 12, 17) if canary else WIDTHS
    names = [f'{q}:{m}:{k}:{n}' for q in TYPES for m, k in shapes for n in widths
             if kind != 'memcheck' or m > 128 or n > 8]
    if not canary and not negative:
        names += [f'q6_K:124160:5120:{n}' for n in (6, 12)]
        names += [f'q8_0:512:5120:{n}' for n in WIDTHS]
    if kind == 'memcheck' and not negative:
        names += [f'id:{q}:{m}:5120:{n}:{b}' for q in TYPES for m in (65, 128)
                  for n in ((12,) if canary else (12, 24, 48)) for b in (0, 1)]
    return names


def run(kind, out, canary, budget):
    if os.environ.get('GPUQ_ACTIVE') != '1':
        raise lib.Fail('probe: integrity: queue environment missing', 'integrity')
    out.mkdir(parents=True, exist_ok=True)
    if any(path.name != 'console.log' for path in out.iterdir()):
        raise lib.Fail('probe: integrity: output is not fresh', 'integrity')
    preflight(kind)
    if kind == 'perf':
        require_memcheck()
    began = time.time()
    deadline = time.monotonic() + budget
    modes = [('patch', False), ('negative', False)] if kind == 'memcheck' else [
        (name, cold) for name in ('base', 'patch', 'patch', 'base') for cold in (False, True)]
    if kind == 'perf' and canary:
        modes = [('patch', False), ('patch', True)]
    records = []
    for i, (mode, cold) in enumerate(modes):
        negative = mode == 'negative'
        root = ROOTS[mode]
        stamp = (root / 'SOURCE').read_text().splitlines()[0][:9]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=str(root / 'build/bin'),
                   OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   GGML_TEST_MMQ_SMALLJ='1', GGML_CUDA_DEBUG_MMQ_SMALLJ='1' if canary or kind == 'memcheck' else '')
        for variable in ('GGML_TEST_MMQ_SMALLJ_COLD', 'GGML_TEST_MMQ_SMALLJ_CANARY'):
            env.pop(variable, None)
        if cold:
            env['GGML_TEST_MMQ_SMALLJ_COLD'] = '1'
        if canary:
            env['GGML_TEST_MMQ_SMALLJ_CANARY'] = '1'
        if kind == 'perf' and not canary:
            env.pop('GGML_CUDA_DEBUG_MMQ_SMALLJ', None)
        command = [str(ROOTS['patch'] / 'build/bin/test-backend-ops'),
                   'test' if kind == 'memcheck' else 'perf', '-b', 'CUDA0', '-o',
                   'MUL_MAT,MUL_MAT_ID' if kind == 'memcheck' and not negative else 'MUL_MAT', '-j', '1', '--output', 'sql']
        if negative:
            command += ['-p', r'm=65,n=12,k=5120,']
        elif canary:
            command += ['-p', r'n=(9|12|17),']
        stem = f'{i}-{mode}-' + ('cold' if cold else 'hot')
        if kind == 'memcheck' and not negative:
            command = ['/usr/local/cuda-13.2/bin/compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                       '--print-limit', '50', '--log-file', str(out / (stem + '.sanitizer.txt'))] + command
        command = ['nice', '-n', '19'] + command
        journal(began, out / 'journal.txt')
        remaining = deadline - time.monotonic() - 30
        if remaining <= 0:
            raise lib.Fail('probe: budget expired', 'timeout')
        (out / (stem + '.command.json')).write_text(json.dumps(command))
        with (out / (stem + '.sql')).open('w') as stdout, (out / (stem + '.log')).open('w') as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            try:
                if kind == 'perf':
                    library_gate(process, root, out / (stem + '.libraries.txt'))
                rc = process.wait(timeout=min(1400 if kind == 'memcheck' else 600, remaining))
            except subprocess.TimeoutExpired:
                raise lib.Fail('probe: budget stage cap', 'timeout')
            finally:
                lib.stop_child(process, 10)
        journal(began, out / 'journal.txt')
        if (rc == 0) == negative or rc not in (0, 1):
            raise lib.Fail(f'probe: reference process rc {rc}', 'deterministic')
        raw = (out / (stem + '.sql')).read_text()
        rows = sql_rows(raw)
        check_rows(rows, expected_cases(kind, canary, negative), perf=kind == 'perf', stamp=stamp, negative=negative)
        if negative:
            errors = re.findall(r'\[MUL_MAT\] ERR = ([0-9.]+) > ([0-9.]+)', raw)
            if len(errors) != len(TYPES) or any(float(error) <= float(limit) for error, limit in errors):
                raise lib.Fail('probe: integrity: missing numeric negative-control evidence', 'integrity')
        log = (out / (stem + '.log')).read_text()
        if kind == 'memcheck' and not negative:
            check_sanitizer((out / (stem + '.sanitizer.txt')).read_text())
        if mode != 'base' and (canary or kind == 'memcheck'):
            for quant in TYPES:
                if not re.search(r'mmq_smallj: device=0 type=' + quant + r' columns=12 J=16 I=64 threads=128 occupancy=2\b', log):
                    raise lib.Fail('probe: integrity: small-J path did not engage: ' + quant, 'integrity')
        if cold and len(re.findall(r'mmq_smallj_cold: .* copies=\d+ weight_bytes=\d+', log)) != len(rows):
            raise lib.Fail('probe: integrity: weight rotation evidence missing', 'integrity')
        records.append(dict(mode=mode, cold=cold, rows=rows))
        (out / 'results.json').write_text(json.dumps(records, indent=2))
        if kind == 'perf':
            (out / 'summary.json').write_text(json.dumps(perf_summary(records), indent=2))
    (out / 'PASS.json').write_text(json.dumps(dict(kind=kind, canary=canary,
        frozen_commit=(ROOTS['patch'] / 'SOURCE').read_text().splitlines()[0],
        binary_digest=lib.file_sha(ROOTS['patch'] / 'build/bin/SHA256SUMS'), wall_s_M=time.time()-began)))
    print('probe: PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['memcheck', 'perf'])
    parser.add_argument('--stage', choices=['selftest', 'preflight', 'canary', 'run'], required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'gpu-results')
    parser.add_argument('--budget', type=int, default=2400)
    args = parser.parse_args()
    try:
        if args.stage == 'selftest':
            selftest()
        elif args.stage == 'preflight':
            preflight(args.kind)
        else:
            run(args.kind, args.out, args.stage == 'canary', args.budget)
        return 0
    except lib.Fail as error:
        print(error, file=sys.stderr)
        return error.rc


if __name__ == '__main__':
    sys.exit(main())
