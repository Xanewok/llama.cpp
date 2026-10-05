"""Queue-only memcheck of the fixed decode FA cases. No intentional GPU fault control."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import time

from fa_probe import Gate, PATCH, ROOT, check_journal, check_rows, command, frozen, metrics, run as probe_preflight, selftest as probe_selftest

SANITIZER = '/usr/local/cuda-13.2/bin/compute-sanitizer'
FIXTURES = ROOT / 'codex/sanitizer-fixtures'


def sanitizer_errors(raw):
    summaries = re.findall(r'^========= ERROR SUMMARY: (\d+) errors?', raw, re.M)
    if '========= COMPUTE-SANITIZER' not in raw or not summaries:
        raise Gate(6, 'missing sanitizer banner or summary')
    # A diagnostic can exist even if the summary format omits that class.
    diagnostics = [line for line in raw.splitlines() if re.match(r'^========= \S', line) and not line.startswith(('========= COMPUTE-SANITIZER', '========= ERROR SUMMARY'))]
    if any(int(n) for n in summaries) or diagnostics:
        raise Gate(8, 'memcheck reports a memory access error; raw report saved')


def selftest():
    probe_selftest()
    clean = (FIXTURES / 'memcheck-clean.log').read_text()
    sanitizer_errors(clean)
    fault = (FIXTURES / 'memcheck-fault-stack-on.log').read_text()
    for raw, code in [(fault, 8), ('plain log', 6), (clean.replace('ERROR SUMMARY: 0 errors', 'ERROR SUMMARY: 1 errors'), 8), (re.sub(r'^========= ERROR SUMMARY:.*$', '', clean, flags=re.M), 6)]:
        try:
            sanitizer_errors(raw)
        except Gate as exc:
            assert exc.code == code
        else:
            raise AssertionError('sanitizer negative control escaped')
    print('M: recorded clean/fault sanitizer reports and missing-summary controls pass')


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    selftest()
    if args.mode == 'preflight':
        args.mode = 'preflight'
        probe_preflight(args)
        if not Path(SANITIZER).is_file():
            raise Gate(6, 'compute-sanitizer missing')
        return
    if os.environ.get('GPUQ_ACTIVE') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise Gate(6, 'memcheck requires the parent queue and one visible GPU')
    commit = frozen(PATCH)
    deadline = time.monotonic() + args.budget_s
    began = time.time()
    selector = r'kv=4096,nb=1,.*tail=4095$' if args.mode == 'canary' else r'tail='
    expected = {(4096, 1, typ, 4095) for typ in ('f16', 'q8_0')} if args.mode == 'canary' else {
        (4096 if tail == 4095 else 8192, q, typ, tail)
        for q in (1, 4, 6, 8) for typ in ('f16', 'q8_0') for tail in (0, 1, 255, 256, 257, 4095)}
    dest = args.out / '0-patch-memcheck'
    env = dict(os.environ, LD_LIBRARY_PATH=str(PATCH / 'build/bin'))
    cmd = [SANITIZER, '--tool', 'memcheck', '--error-exitcode', '8', '--destroy-on-device-error', 'context', '--force-blocking-launches', '--print-limit', '32',
           str(PATCH / 'build/bin/test-backend-ops'), 'test', '-b', 'CUDA0', '-o', 'FLASH_ATTN_EXT', '-p', selector, '--output', 'sql']
    try:
        raw = command(cmd, dest, deadline, env, check=False)
    finally:
        check_journal(args.out / 'kernel-journal.log', began)
    errors = dest.with_suffix('.err').read_text()
    sanitizer_errors(errors + '\n' + raw)
    rc = json.loads(dest.with_suffix('.returncode.json').read_text())['rc']
    if rc:
        raise Gate(8, f'memcheck target failed rc={rc}; raw artifacts saved')
    rows = metrics(raw, 'test', commit)
    check_rows(rows, expected)
    if 'decode KV bound:' not in errors:
        raise Gate(6, 'decode bound did not engage')
    (args.out / 'summary.json').write_text(json.dumps({'commit': commit, 'passed_M': len(rows), 'memcheck_errors_M': 0}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('selftest', 'preflight', 'canary', 'run'))
    parser.add_argument('--out', type=Path, default=ROOT / 'codex/memcheck-selftest')
    parser.add_argument('--budget-s', type=int, default=600)
    args = parser.parse_args()
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
