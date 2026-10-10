"""Padding invariance verdict and cross-route errors; thresholds are not accuracy gates."""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
import re
import sqlite3
import struct

REAL_SQL = Path('/home/xanewok/repos/qwen38-eval-worktrees/wt/restrict-probe/codex/window/restrict-a2/perf-CUDA0-stock-1-fa.out')
FIXTURES = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')


def sql_rows(path):
    if path.stat().st_size > 200_000_000:
        raise ValueError('oversized SQL log')
    text = path.read_text()
    statements = re.findall(r'CREATE TABLE IF NOT EXISTS test_backend_ops \([\s\S]*?\);|INSERT INTO test_backend_ops [^\n]*;', text)
    if not statements or 'CREATE TABLE' not in statements[0]:
        raise ValueError('missing SQL header')
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.executescript('\n'.join(statements))
    rows = [dict(r) for r in db.execute('SELECT * FROM test_backend_ops')]
    db.close()
    if not rows:
        raise ValueError('empty SQL result')
    return rows


def faults(text):
    return re.findall(r'\bXid\s*\([^\n]*|\bGPU [^\n]*fallen off the bus[^\n]*', text)


def engagement(text):
    return len(re.findall(r'\bfattn_q8_decode: device=\d+ dk=256 gqa=6 seqs=1$', text, re.M))


def expected_cases(canary):
    cases = set()
    for nh in (2, 4):
        for head_major in (0, 1):
            for depth in (8192, 32768, 90112):
                for delta in (-1, 0, 1, 5, 64):
                    visible = depth + delta
                    cases.add((nh, visible, (visible + 4095)//4096*4096, 1, head_major))
            cases.update((nh, v, kv, scale, head_major) for v, kv, scale in
                         ((8252, 12288, 1), (8193, 12288, 20), (1, 94208, 1)))
    return {(2, 8193, 12288, 1, 0)} if canary else cases


def case_key(params):
    # The vars format is shared with the existing served-shape attention suite.
    fields = dict(re.findall(r'(\w+)=([^,\[]+)(?:,|$)', params))
    for token in ('hsk=256,hsv=256,', 'nr23=[6,1]', 'nb=1,mask=1,sinks=0',
                  'max_bias=0.000000,logit_softcap=0.000000', 'type_K=q8_0,type_V=q8_0',
                  'permute=[0,2,1,3]', 'capacity=102400'):
        if token not in params:
            raise ValueError('wrong attention shape: ' + params)
    return tuple(int(fields[k]) for k in ('nh', 'visible', 'kv', 'q_scale', 'head_major'))


def evidence(row):
    if row['op_name'] != 'FLASH_ATTN_EXT' or row['test_mode'] != 'test' or row['supported'] != 1:
        raise ValueError('missing attention support or wrong mode')
    data = json.loads(row['error_message'])
    if set(data) != {'finite', 'invariant', 'control', 'compute', 'output', 'reference'}:
        raise ValueError('wrong evidence fields')
    for flag in ('finite', 'invariant', 'control', 'compute'):
        if type(data[flag]) is not int or data[flag] not in (0, 1):
            raise ValueError('invalid evidence flag')
    n = 256*case_key(row['op_params'])[0]*6
    for key in ('output', 'reference'):
        if len(data[key]) != n or not all(type(x) in (int, float) and math.isfinite(x) for x in data[key]):
            raise ValueError('non-finite or wrong output length')
    if not data['finite'] or not data['compute']:
        raise ValueError('non-finite output or failed computation')
    if data['invariant'] and not data['control']:
        raise ValueError('unmasked garbage negative control is void')
    if bool(row['passed']) != bool(data['invariant']):
        raise ValueError('SQL pass flag disagrees with evidence')
    return data


def errors(reference, output):
    denominator = sum(x*x for x in reference)
    squared = sum((a-b)**2 for a, b in zip(reference, output))
    return {'nmse_D': squared/denominator if denominator else (0.0 if not squared else None),
            'max_abs_D': max(abs(a-b) for a, b in zip(reference, output))}


def bits(values):
    return struct.pack('<' + 'f'*len(values), *values)


def summarize(out, canary=False):
    expected = expected_cases(canary)
    devices = ('CUDA0',) if canary else ('CUDA0', 'CUDA1')
    arms = ('on', 'off') if canary else ('on', 'off', 'off', 'on')
    table, defects = [], []
    for device in devices:
        runs = []
        for index, arm in enumerate(arms):
            prefix = out / f'{device}-{index}-{arm}'
            rows = sql_rows(prefix.with_suffix('.sql'))
            rc = int(prefix.with_suffix('.rc').read_text())
            if rc not in (0, 1):
                raise ValueError(f'{prefix}: process failed with {rc}')
            stderr = prefix.with_suffix('.err').read_text()
            engaged = engagement(stderr)
            if (arm == 'on' and engaged < 1) or (arm == 'off' and engaged != 0):
                raise ValueError(f'{prefix}: wrong grouped q8 route, markers={engaged}')
            cells = {}
            for row in rows:
                if not row['op_params'].startswith('padding=1,'):
                    continue
                if row['backend_name'] != device:
                    raise ValueError('wrong backend in SQL')
                key = case_key(row['op_params'])
                if key in cells:
                    raise ValueError('duplicate case')
                cells[key] = evidence(row)
            if cells.keys() != expected:
                raise ValueError(f'{prefix}: case coverage differs: missing={expected-cells.keys()} extra={cells.keys()-expected}')
            broken = [key for key, data in cells.items() if not data['invariant']]
            if (rc == 0) != (not broken):
                raise ValueError('process return code disagrees with padding result or ancillary tests failed')
            defects.extend({'device': device, 'arm': arm, 'case_D': key} for key in broken)
            runs.append(cells)
        for key in sorted(expected):
            on, off = runs[0][key], runs[1][key]
            for run in runs:
                if bits(run[key]['reference']) != bits(on['reference']):
                    raise ValueError('CPU reference changed between arms')
            table.append({'device': device, 'case_D': key, 'on_to_cpu': errors(on['reference'], on['output']),
                          'off_to_cpu': errors(on['reference'], off['output']), 'on_to_off': errors(off['output'], on['output']),
                          'on_repeat_equal_D': None if canary else bits(on['output']) == bits(runs[3][key]['output']),
                          'off_repeat_equal_D': None if canary else bits(off['output']) == bits(runs[2][key]['output'])})
    result = {'verdict': 'defect' if defects else 'no padding defect', 'scope': 'canary' if canary else 'full',
              'defects': defects, 'cross_route': table}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (out / 'errors.tsv').open('w') as f:
        f.write('device\tnh_D\tvisible_D\tNpad_D\tq_scale_D\thead_major_D\tpair\tNMSE_D\tmax_abs_D\n')
        for row in table:
            for pair in ('on_to_cpu', 'off_to_cpu', 'on_to_off'):
                f.write('\t'.join(map(str, (row['device'], *row['case_D'], pair, *row[pair].values()))) + '\n')
    return result


def selftest(out, save=True):
    rows = sql_rows(REAL_SQL)
    assert len(rows) == 9 and all(r['supported'] and r['passed'] for r in rows)
    assert len(faults((FIXTURES / 'xid-lines.txt').read_text())) == 4
    real_log = Path('/mnt/scratch/work/s10-identity2/s10-kld-iq4-v2/a1/canary/0-stock/tool.log')
    marker_count = engagement(real_log.read_text())
    assert marker_count > 0
    assert engagement(real_log.read_text().replace('fattn_q8_decode:', 'disabled_q8_decode:')) == 0
    # New evidence fields do not exist in archived GPU logs. Validate them on new CPU results too.
    result = {'real_SQL': str(REAL_SQL), 'real_SQL_rows_M': len(rows), 'real_engagement': str(real_log),
              'real_engagement_markers_M': marker_count, 'fault_fixture': str(FIXTURES / 'xid-lines.txt')}
    if (out / 'cpu.sql').exists():
        cpu_rows = [r for r in sql_rows(out / 'cpu.sql') if r['op_params'].startswith('padding=1,')]
        cells = {case_key(r['op_params']): evidence(r) for r in cpu_rows}
        assert cells.keys() == expected_cases(False)
        assert all(c['invariant'] for c in cells.values())
        mutated = dict(cpu_rows[0])
        data = json.loads(mutated['error_message'])
        data['invariant'] = 0
        mutated['error_message'] = json.dumps(data)
        mutated['passed'] = 0
        assert evidence(mutated)['invariant'] == 0
        for flag in ('control', 'compute', 'finite'):
            bad = dict(data, invariant=1, **{flag: 0})
            mutated['error_message'] = json.dumps(bad)
            try:
                evidence(mutated)
            except ValueError:
                pass
            else:
                raise AssertionError('broken gate accepted: ' + flag)
        if save:
            fixture = out / 'summary-fixture'
            fixture.mkdir(exist_ok=True)
            chosen = next(r for r in cpu_rows if case_key(r['op_params']) == (2, 8193, 12288, 1, 0))
            raw = (out / 'cpu.sql').read_text()
            header = re.search(r'CREATE TABLE IF NOT EXISTS test_backend_ops \([\s\S]*?\);', raw)[0]
            statement = next(line for line in raw.splitlines() if line.startswith('INSERT INTO') and chosen['op_params'] in line)
            for index, arm in enumerate(('on', 'off')):
                prefix = fixture / f'CUDA0-{index}-{arm}'
                prefix.with_suffix('.sql').write_text(header + '\n' + statement.replace("'CPU'", "'CUDA0'") + '\n')
                prefix.with_suffix('.rc').write_text('0\n')
                prefix.with_suffix('.err').write_text('fattn_q8_decode: device=0 dk=256 gqa=6 seqs=1\n' if arm == 'on' else '')
            assert summarize(fixture, True)['verdict'] == 'no padding defect'
            mutant = dict(chosen)
            data = json.loads(mutant['error_message'])
            data['invariant'] = 0
            old_json = chosen['error_message']
            broken_statement = statement.replace(old_json, json.dumps(data)).replace("'test', '1', '1'", "'test', '1', '0'")
            (fixture / 'CUDA0-0-on.sql').write_text(header + '\n' + broken_statement.replace("'CPU'", "'CUDA0'") + '\n')
            (fixture / 'CUDA0-0-on.rc').write_text('1\n')
            assert summarize(fixture, True)['verdict'] == 'defect'
            # Coverage and engagement gates must reject an empty or disabled route.
            (fixture / 'CUDA0-0-on.sql').write_text(header + '\n' + statement.replace("'CPU'", "'CUDA0'") + '\n')
            (fixture / 'CUDA0-0-on.rc').write_text('0\n')
            (fixture / 'CUDA0-0-on.err').write_text('')
            try:
                summarize(fixture, True)
            except ValueError:
                pass
            else:
                raise AssertionError('missing grouped route accepted')
        result['cpu_cases_M'] = len(cells)
        result['cpu_max_nmse_M'] = max(errors(c['reference'], c['output'])['nmse_D'] for c in cells.values())
        result['cpu_max_abs_M'] = max(errors(c['reference'], c['output'])['max_abs_D'] for c in cells.values())
    if save:
        (out / 'parser-check.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def execute(out, binary, budget, canary):
    if os.environ.get('GPUQ_ITEM') != 'bw-pkg54-padinv' or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise ValueError('GPU execution requires the gpuq window')
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    deadline = time.monotonic() + budget
    child = None

    def interrupted(signum, frame):
        raise InterruptedError('window interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    pattern = 'padding=1,.*nh=2,.*visible=8193,.*q_scale=1,head_major=0$' if canary else '^padding=1,'
    devices = ('CUDA0',) if canary else ('CUDA0', 'CUDA1')
    arms = ('on', 'off') if canary else ('on', 'off', 'off', 'on')
    try:
        for device in devices:
            for index, arm in enumerate(arms):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired('budget', budget)
                prefix = out / f'{device}-{index}-{arm}'
                env = dict(os.environ, GGML_CUDA_Q8_DECODE='1' if arm == 'on' else '0', OMP_NUM_THREADS='1')
                command = [str(binary), 'test', '-b', device, '-o', 'FLASH_ATTN_EXT', '-p', pattern, '--output', 'sql', '-j', '1']
                prefix.with_suffix('.command.json').write_text(json.dumps({'argv': command, 'Q8_DECODE': env['GGML_CUDA_Q8_DECODE']}) + '\n')
                with prefix.with_suffix('.sql').open('w') as stdout, prefix.with_suffix('.err').open('w') as stderr:
                    child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                    prefix.with_suffix('.pid').write_text(str(child.pid) + '\n')
                    try:
                        rc = child.wait(timeout=min(900, max(1, deadline-time.monotonic())))
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait(timeout=10)
                prefix.with_suffix('.rc').write_text(str(rc) + '\n')
                child = None
                journal = subprocess.run(['journalctl', '-k', '-b', '--since', '@' + str(start), '--no-pager', '-o', 'short-unix'],
                                         capture_output=True, text=True, timeout=30)
                prefix.with_suffix('.journal').write_text(journal.stdout + journal.stderr)
                if journal.returncode:
                    raise ValueError('kernel journal is unavailable')
                if faults(journal.stdout):
                    (out / 'summary.json').write_text(json.dumps({'verdict': 'hazard'}) + '\n')
                    return 3
                if rc not in (0, 1):
                    raise ValueError(f'{prefix}: binary failed with {rc}')
        result = summarize(out, canary)
        return 8 if result['verdict'] == 'defect' else 0
    except subprocess.TimeoutExpired:
        (out / 'summary.json').write_text(json.dumps({'verdict': 'budget', 'budget_s_D': budget}) + '\n')
        return 7
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('summary', 'selftest', 'journal', 'execute'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--read-only', action='store_true')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--budget-s', type=int, default=3600)
    args = parser.parse_args()
    try:
        if args.mode == 'execute':
            return execute(args.path, args.binary, args.budget_s, args.canary)
        if args.mode == 'journal':
            return 3 if faults(args.path.read_text()) else 0
        result = selftest(args.path, save=not args.read_only) if args.mode == 'selftest' else summarize(args.path, args.canary)
        print(json.dumps(result if args.mode == 'selftest' else {'verdict': result['verdict'], 'cases_M': len(result['cross_route'])}))
        return 8 if result.get('verdict') == 'defect' else 0
    except (ValueError, KeyError, AssertionError, OSError, sqlite3.Error, InterruptedError) as exc:
        if args.mode in ('summary', 'execute') and args.path.is_dir():
            (args.path / 'summary.json').write_text(json.dumps({'verdict': 'invalid', 'reason': str(exc)}, indent=2) + '\n')
        print('padinv: FAIL ' + str(exc))
        return 8


if __name__ == '__main__':
    raise SystemExit(main())
