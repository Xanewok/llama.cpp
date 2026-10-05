"""Queue-only padded FA probe; selftest and preflight do not execute CUDA binaries."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
BASE = Path('/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack5')
PATCH = ROOT / 'codex/frozen/fa-kvmax-decode'
FIXTURES = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')
REAL_SQL = ROOT / 'codex/probe-fixtures/real-fa-perf.sql'
FIELDS = ('stamp', 'commit', 'backend', 'op', 'params', 'mode', 'supported', 'passed', 'error', 'us', 'flops', 'bw', 'mem', 'runs', 'device', 'reg')
HAZARD = re.compile(r'\bXid\b|fallen off the bus')


class Gate(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)


def sql_rows(raw):
    rows = []
    for line in raw.splitlines():
        if not line.startswith('INSERT INTO test_backend_ops'):
            continue
        body = line.split(' VALUES (', 1)[1].rsplit(');', 1)[0]
        values = re.findall(r"'((?:[^']|'')*)'", body)
        if len(values) != len(FIELDS):
            raise Gate(6, 'SQL field count changed')
        rows.append(dict(zip(FIELDS, (v.replace("''", "'") for v in values))))
    return rows


def metrics(raw, mode, commit):
    result = []
    for row in sql_rows(raw):
        if row['op'] != 'FLASH_ATTN_EXT':
            continue
        if row['backend'] != 'CUDA0' or row['mode'] != mode or row['commit'] != commit[:9]:
            raise Gate(6, 'backend, mode or library stamp differs')
        if row['supported'] != '1' or row['passed'] != '1' or row['error']:
            raise Gate(8, 'FA correctness or performance failed')
        if mode == 'perf' and (not math.isfinite(float(row['us'])) or float(row['us']) <= 0 or int(row['runs']) < 2):
            raise Gate(5, 'invalid performance sample')
        result.append(row)
    if not result:
        raise Gate(5, 'FA rows missing')
    return result


def key(row):
    p = row['params']
    required = ('hsk=256,hsv=256,nh=4,nr23=[6,1]', 'type_K=', 'type_V=', ',tail=')
    if not all(part in p for part in required):
        raise Gate(6, 'FA geometry changed')
    def field(name):
        return re.search(r'(?:^|,)' + name + r'=([^,]+)', p)[1]
    k, q, tail = int(field('kv')), int(field('nb')), int(field('tail'))
    typ = field('type_K')
    if typ != field('type_V') or typ not in ('f16', 'q8_0'):
        raise Gate(6, 'KV types changed')
    return k, q, typ, tail


def check_rows(rows, expected):
    keys = [key(row) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise Gate(5, 'missing or duplicate FA shapes')


def frozen(root):
    bindir = root / 'build/bin'
    for line in (bindir / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if hashlib.sha256((bindir / name.lstrip('*')).read_bytes()).hexdigest() != digest:
            raise Gate(6, 'frozen bytes changed')
    return (root / 'SOURCE').read_text().splitlines()[0]


def selftest():
    raw = REAL_SQL.read_text()
    rows = metrics(raw, 'perf', '590912a1d')
    assert len(rows) >= 9 and any(float(r['us']) > 0 for r in rows)
    for changed in (raw.replace("'CUDA0'", "'CPU'"), raw.replace("'perf', '1', '1'", "'perf', '1', '0'"), raw.replace("'590912a1d'", "'000000000'")):
        try:
            metrics(changed, 'perf', '590912a1d')
        except Gate:
            pass
        else:
            raise AssertionError('negative parser control escaped')
    assert HAZARD.search((FIXTURES / 'xid-lines.txt').read_text())
    assert not HAZARD.search('stop_xid_hash_s_G')
    from harness.depth import fixture_test
    fixture_test(FIXTURES)
    bad = re.sub(r"('', )'[0-9.]+', '[0-9.]+',", r"\1'nan', '0',", raw)
    try:
        metrics(bad, 'perf', '590912a1d')
    except Gate:
        pass
    else:
        raise AssertionError('non-finite time escaped')
    print('M: recorded GPU SQL, SSE, memory, journal parsers and negative controls pass')


def command(cmd, dest, deadline, env):
    left = deadline - time.monotonic()
    if left <= 0:
        raise Gate(7, 'budget expired')
    dest.with_suffix('.command.json').write_text(json.dumps(cmd) + '\n')
    with dest.with_suffix('.sql').open('w') as stdout, dest.with_suffix('.err').open('w') as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=left)
        except BaseException as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise Gate(7, 'budget expired') from exc
            raise
    if rc:
        raise Gate(8, f'FA process failed rc={rc}; raw artifacts saved')
    return dest.with_suffix('.sql').read_text()


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    selftest()
    commits = {'up': frozen(BASE), 'patch': frozen(PATCH)}
    exe = PATCH / 'build/bin/test-backend-ops'
    # Both arms use the same new test cases and their own unchanged backend libraries.
    for arm, root in (('up', BASE), ('patch', PATCH)):
        env = dict(os.environ, LD_LIBRARY_PATH=str(root / 'build/bin'))
        output = subprocess.run(['ldd', str(exe)], env=env, capture_output=True, text=True, check=True).stdout
        libs = re.findall(r'libggml\S+ => (\S+)', output)
        if not libs or any(Path(p).resolve().parent != (root / 'build/bin').resolve() for p in libs):
            raise Gate(6, 'backend libraries resolve outside the selected freeze')
    if args.mode == 'preflight':
        print(json.dumps({'commits': commits, 'test_binary': str(exe), 'plan': ['up', 'patch', 'patch', 'up']}))
        return
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('GPUQ_ACTIVE') != '1':
        raise Gate(6, 'GPU execution requires the parent queue')
    began = time.time()
    deadline = time.monotonic() + args.budget_s
    plan = ['patch'] if args.mode == 'canary' else ['up', 'patch', 'patch', 'up']
    expected = {(8448, 1, t, 255) for t in ('f16', 'q8_0')} if args.mode == 'canary' else {
        ((live + pad - 1)//pad*pad, q, t, (live + pad - 1)//pad*pad-live)
        for live in (8193, 32769, 90113) for pad in (256, 4096) for q in (1, 4, 6, 8) for t in ('f16', 'q8_0')}
    results = []
    for boot, arm in enumerate(plan):
        env = dict(os.environ, LD_LIBRARY_PATH=str((BASE if arm == 'up' else PATCH) / 'build/bin'), OMP_NUM_THREADS='8', OMP_THREAD_LIMIT='8')
        selector = r'kv=8448,nb=1,.*tail=255$' if args.mode == 'canary' else r'tail='
        for mode in ('test', 'perf'):
            chosen = r'kv=4096,nb=1,.*tail=4095$' if mode == 'test' and args.mode == 'canary' else (r'tail=' if mode == 'test' else selector)
            raw = command([str(exe), mode, '-b', 'CUDA0', '-o', 'FLASH_ATTN_EXT', '-p', chosen, '--output', 'sql'], args.out / f'{boot}-{arm}-{mode}', deadline, env)
            rows = metrics(raw, mode, commits[arm])
            if mode == 'perf' or args.mode == 'canary':
                check_rows(rows, {(4096, 1, t, 4095) for t in ('f16', 'q8_0')} if mode == 'test' else expected)
            if arm == 'patch' and 'decode KV bound:' not in (args.out / f'{boot}-{arm}-{mode}.err').read_text():
                raise Gate(6, 'decode bound did not engage')
            results.extend(dict(row, arm=arm, boot=boot) for row in rows)
        journal = subprocess.run(['journalctl', '-k', '-b', '--since', f'@{int(began)}', '--no-pager'], capture_output=True, text=True, check=True).stdout
        (args.out / f'{boot}-kernel-journal.log').write_text(journal)
        if HAZARD.search(journal):
            raise Gate(3, 'kernel hazard since item start')
    summary = {'rows_M': results, 'commits': commits}
    if args.mode == 'run':
        summary['perf_M'] = [dict(kv=k[0], q=k[1], type=k[2], tail=k[3], up_us=statistics.mean(float(r['us']) for r in results if r['arm']=='up' and r['mode']=='perf' and key(r)==k), patch_us=statistics.mean(float(r['us']) for r in results if r['arm']=='patch' and r['mode']=='perf' and key(r)==k)) for k in sorted(expected)]
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('selftest', 'preflight', 'canary', 'run'))
    p.add_argument('--out', type=Path, default=ROOT / 'codex/probe-selftest')
    p.add_argument('--budget-s', type=int, default=900)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise Gate(7, 'item interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        selftest() if args.mode == 'selftest' else run(args)
    except Gate as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'verdict.json').write_text(json.dumps({'class_code': exc.code, 'error': str(exc)}) + '\n')
        print(str(exc), flush=True)
        raise SystemExit(exc.code)
