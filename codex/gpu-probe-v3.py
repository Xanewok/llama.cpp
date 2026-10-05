#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('memcheck', HERE / 'memcheck-v3/memcheck.py')
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


def judge(logs, canary):
    exact = logs['exact']
    count = 3 if canary else 63
    if f'PASS: exact self-check cases={count * 2}\n' not in exact:
        raise ValueError('exact self-check did not complete on both GPUs')
    for device in (0, 1):
        if f'PASS: device=CUDA{device} cases={count} self-check\n' not in exact:
            raise ValueError(f'exact CUDA{device} case count')
        lines = re.findall(rf'q8_reuse: device={device} graph=\d+ rows=(\d+) skipped=(\d+)', exact)
        if len(lines) != count * 4:
            raise ValueError(f'CUDA{device} direct/capture/replay evaluation count {len(lines)}')
        if not any(int(r) == 6 and int(s) > 0 for r, s in lines):
            raise ValueError(f'CUDA{device} q8 reuse did not engage')
        if f'add_rms_norm: device={device} rows=6 columns=2560' not in exact:
            raise ValueError(f'CUDA{device} residual fusion did not engage')
        for op, expected in ((mc.SHARED_OP, 1 if canary else 60), (mc.FUSED_OPS, 1 if canary else 19)):
            text = logs[f'{device}-{op}']
            cases = mc.parse_cases(text, op)
            if cases != dict(ok=expected, fail=0, ns=0) or 'RTX 5060 Ti' not in text:
                raise ValueError(f'CUDA{device} {op}: {cases}, expected {expected}')
            if op == mc.SHARED_OP and not re.search(rf'q8_reuse: device={device} graph=\d+ rows=6 skipped=[1-9]\d*\b', text):
                raise ValueError(f'CUDA{device} shared backend test did not engage reuse')
            if op == mc.FUSED_OPS:
                shapes = {(int(r), int(c)) for r, c in re.findall(rf'add_rms_norm: device={device} rows=(\d+) columns=(\d+)', text)}
                if not ({(1, 512)} if canary else mc.FUSED_SHAPES) <= shapes:
                    raise ValueError(f'CUDA{device} fused backend test shapes {shapes}')


def selftest():
    rc = mc.selftest()
    sys.path.insert(0, '/home/xanewok/repos/qwen38-eval')
    from harness.depth import fixture_test
    fixture_test('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')
    # Positive engagement recordings do not exist yet. Synthetic lines exercise their gates;
    # the recorded backend/sanitizer/fault parsers above establish the real formats.
    logs = {'exact': 'PASS: exact self-check cases=6\n'}
    for d in (0, 1):
        logs['exact'] += f'PASS: device=CUDA{d} cases=3 self-check\n' + ''.join(
            f'q8_reuse: device={d} graph={i} rows=6 skipped=1\n' for i in range(12)) + f'add_rms_norm: device={d} rows=6 columns=2560\n'
        logs[f'{d}-{mc.SHARED_OP}'] = f'RTX 5060 Ti\nMUL_MAT_SHARED(type=q4_K,rows=6,strided=0,overwrite=0): OK\nq8_reuse: device={d} graph=1 rows=6 skipped=1\n'
        logs[f'{d}-{mc.FUSED_OPS}'] = f'RTX 5060 Ti\nRMS_NORM_MUL_ADD(ne=[512,1,1,1]): OK\nadd_rms_norm: device={d} rows=1 columns=512\n'
    judge(logs, True)
    for key in logs:
        broken = dict(logs)
        broken[key] = ''
        try:
            judge(broken, True)
        except ValueError:
            continue
        raise ValueError(f'missing {key} accepted')
    recorded = (HERE / 'memcheck-v3/fixtures/memcheck-clean.log').read_text(errors='replace')
    try:
        judge({key: recorded for key in logs}, True)
    except ValueError:
        pass
    else:
        raise ValueError('recorded old kernel log accepted as exact q8 probe')
    print('PASS: v3 probe parser controls')
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['preflight', 'canary', 'execute'])
    ap.add_argument('--out', type=Path)
    ap.add_argument('--budget-s', type=int, default=600)
    a = ap.parse_args()
    if a.mode == 'preflight':
        bad = mc.check_frozen()
        if bad:
            print('integrity:', bad)
            return 6
        return selftest()
    if os.environ.get('GPUQ_ACTIVE') != '1' or a.out is None:
        return 6
    os.nice(19)
    canary = a.mode == 'canary'
    a.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = time.monotonic() + a.budget_s
    env = mc.stage_env()
    for knob in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT'):
        env.pop(knob, None)
    plan = [('exact', [mc.BIN + '/q8-exact', '--canary' if canary else '--self-check'], 60 if canary else 120)]
    for d in (0, 1):
        for op, params in ((mc.SHARED_OP, mc.SHARED_CANARY if canary else ''), (mc.FUSED_OPS, mc.CANARY_P if canary else mc.FUSED_P)):
            plan.append((f'{d}-{op}', [mc.BIN + '/test-backend-ops', 'test', '-b', f'CUDA{d}', '-o', op, '-p', params], 30 if canary else 60))
    logs = {}
    rc = 0
    try:
        bad = mc.check_frozen()
        if bad:
            raise ValueError(str(bad))
        for name, argv, cap in plan:
            remaining = min(cap, deadline - time.monotonic() - 20)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, a.budget_s)
            with (a.out / f'{name}.log').open('wb') as f:
                p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                try:
                    status = p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                    raise
            logs[name] = (a.out / f'{name}.log').read_text(errors='replace')
            if status:
                raise ValueError(f'{name} rc={status}')
            faults = mc.journal_xids(start)
            if faults is None:
                raise ValueError('kernel journal unavailable')
            if faults:
                rc = 3
                break
        if not rc:
            judge(logs, canary)
    except subprocess.TimeoutExpired:
        print('budget: v3 probe expired')
        rc = 7
    except ValueError as e:
        print('integrity:', e)
        rc = 8
    finally:
        faults = mc.journal_xids(start)
        (a.out / 'faults.json').write_text(json.dumps(faults))
        if faults:
            rc = 3
        elif faults is None and rc == 0:
            rc = 6
        (a.out / 'verdict.json').write_text(json.dumps(dict(rc=rc, completed=list(logs))))
    print(f'probe: result rc={rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
