#!/usr/bin/env python3
"""Real-artifact parser tests and CPU rehearsals of both queue stages."""
import copy
import json
import os
import socket
import subprocess
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

import sweep

ROOT = sweep.ROOT
DATA = sweep.DATA
FIXTURES = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')
NATIVE = Path('/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack4-depth-j5q/full-a1/2-patch/depth-8192.json')
RECORDED = Path('/home/xanewok/repos/k-sweep/runs/k-sweep-j5q-4/full-a1')


def check_real():
    parsed = []
    for path in sorted((FIXTURES / 'llama-sse').glob('*.json')):
        data = json.loads(path.read_text())
        if 'response' not in data:
            continue
        rows = data['chunks'] + [data['response']]
        t = sweep.timing(rows)
        assert t == data['response']['timings'], path
        assert sweep.timing(rows * 2 + [None, {}, dict(timings=None), dict(timings={'draft_n': None})]) == t
        assert sweep.timing([dict(timings=dict(t, draft_n_accepted=t['draft_n'] + 1))]) is None
        parsed.append(str(path))
    native = json.loads(NATIVE.read_text())
    assert sweep.timing(native['chunks']) == native['chunks'][-1]['timings']
    assert native['chunks'][-1]['generation_settings']['top_k'] == 20
    assert native['chunks'][-1]['stop'] is True
    assert native['chunks'][-1]['tokens_predicted'] == 512
    assert not sweep.hazard('stop_xid_hash_s_G; NotXid; Xidentifier')
    assert sweep.hazard((FIXTURES / 'xid-lines.txt').read_text())
    assert sum(sweep.gpu_memory((FIXTURES / 'nvidia-smi-query.csv').read_text()).values()) == 123
    assert sweep.timing([None, {}, dict(timings=None)]) is None
    return parsed + [str(NATIVE)]


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_case(out, model, canary=False, env=None, expected=0):
    command = ['bash', str(DATA / 'gpu_ab.sh'), '--model', model, '--stub', '--out', str(out),
               '--port', str(free_port()), '--budget-s', '120']
    if canary:
        command.append('--canary')
    result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                            env=dict(os.environ, **(env or {})))
    assert result.returncode == expected, (command, result.returncode, result.stdout, result.stderr)
    rows = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == (8 if canary else 48 if expected == 0 else 4)
    for path in out.glob('boot-*.pid.json'):
        pid = json.loads(path.read_text())['pid']
        assert not Path(f'/proc/{pid}').exists(), f'orphaned owned process {pid}'
    return rows


def check_prefix_gate(home):
    runner = sweep.Runner(SimpleNamespace(model='j5q', out=home / 'prefix-gates', budget_s=120))
    for i, prefix in enumerate(runner.corpus['prefixes']):
        baseline = runner.bounds['prefixes'][prefix['id']]
        lo, hi = baseline['bounds_D']
        n = baseline['rendered_prompt_tokens_M']
        replies = {'/apply-template': dict(prompt='fixture'), '/tokenize': dict(tokens=[i + 1] * n)}

        def http(path, body):
            if path == '/apply-template':
                assert body['tools'] == runner.corpus['tools']
                assert body['chat_template_kwargs'] == dict(enable_thinking=True, reasoning_effort='xhigh', preserve_thinking=True)
            return replies[path]

        runner.http = http
        assert len(runner.prepare(prefix)) == n
        for length in (lo, hi):
            replies['/tokenize']['tokens'] = [i + 1] * length
            assert len(runner.prepare(prefix)) == length
        for broken in ([i + 1] * (lo - 1), [i + 1] * (hi + 1),
                       [True] * n, [dict(id=i + 1)] * n, None, 'invalid'):
            replies['/tokenize']['tokens'] = broken
            try:
                runner.prepare(prefix)
            except sweep.Failure as error:
                assert error.code == 6
                for token in (f"prefix={prefix['id']}", 'length_M=', 'element_types=', 'recorded_M=', 'cpu_rendered_M='):
                    assert token in str(error), str(error)
            else:
                raise AssertionError(f"invalid prefix accepted: {prefix['id']}")
    prefix = runner.corpus['prefixes'][0]
    runner.bounds['prefixes'][prefix['id']]['recorded_prompt_tokens_M'] += 1
    replies['/tokenize']['tokens'] = [1] * runner.bounds['prefixes'][prefix['id']]['rendered_prompt_tokens_M']
    try:
        runner.prepare(prefix)
    except sweep.Failure as error:
        assert error.code == 6
    else:
        raise AssertionError('changed recorded length accepted')


def check_real_request(home):
    runner = sweep.Runner(SimpleNamespace(model='j5q', out=home / 'real-response', budget_s=120, stub=False))
    runner.arm = 'subset'
    prefix = next(p for p in runner.corpus['prefixes'] if p['id'] == 'p02')
    ids = json.loads((RECORDED / 'p02.tokens.json').read_text())['tokens']
    chunks = json.loads((RECORDED / 'b1-p02.response.json').read_text())['chunks']
    runner.http = lambda *args, **kwargs: chunks
    assert runner.one(prefix, ids, 5, 0, length=512)['valid']
    chunks = copy.deepcopy(chunks)
    chunks[-1]['generation_settings']['ignore_eos'] = True
    row = runner.one(prefix, ids, 5, 1, length=512)
    assert not row['valid'] and row['code'] == 6


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    artifacts = check_real()
    (DATA / 'runs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu-', dir=DATA / 'runs') as directory:
        home = Path(directory)
        check_prefix_gate(home)
        check_real_request(home)
        artifacts += [str(RECORDED / filename) for filename in ('p02.tokens.json', 'b1-p02.response.json')]
        for model in ('j5q', 'iq4'):
            rows = run_case(home / (model + '-full'), model)
            summary = sweep.summary(rows, sorted({r['prefix'] for r in rows}))
            assert summary['status'] == 'ok'
            assert abs(summary['ratios_D']['subset-k6/base-k5']['median_D'] - 1.21) < 1e-12
            assert summary['acceptance']['base-k5']['drafted_M'] == sum(r['drafted_M'] for r in rows if r['k'] == 5 and r['arm'] == 'base')
            corrupt = copy.deepcopy(rows)
            for row in corrupt:
                if row['k'] == 5 and row['arm'] == 'base' and row['boot'] == 11:
                    row['tok_s_M'] *= 2
            assert sweep.summary(corrupt, sorted({r['prefix'] for r in rows}))['status'] == 'INVALID'
            assert sweep.summary(rows[:-1], sorted({r['prefix'] for r in rows}))['status'] == 'INVALID'
            drift = copy.deepcopy(rows)
            drift[-1]['content_sha256'] = 'broken'
            recorded = sweep.summary(drift, sorted({r['prefix'] for r in rows}))
            assert recorded['status'] == 'ok' and len(recorded['seeded_mismatch_M']) == 1
            drift[-1]['prompt_sha256'] = 'broken'
            assert sweep.summary(drift, sorted({r['prefix'] for r in rows}))['status'] == 'INVALID'
            # Covering 1.0 is insufficient: a wide repeat interval must fail equivalence.
            wide = copy.deepcopy(rows)
            sessions = sorted({r['session'] for r in rows})
            for row in wide:
                if row['boot'] == 11:
                    row['tok_s_M'] *= 0.95 if row['session'] in sessions[:2] else 1.05
            assert sweep.summary(wide, sorted({r['prefix'] for r in rows}))['status'] == 'INVALID'
            run_case(home / (model + '-canary'), model, True)
        run_case(home / 'short-full', 'j5q', env={'STUB_SHORT': '1'}, expected=5)
        run_case(home / 'short-canary', 'iq4', True, {'STUB_SHORT': '1'}, expected=5)
        run_case(home / 'sampling-drift', 'j5q', env={'STUB_DRIFT': '1'}, expected=6)
        # A frozen-file failure must occur before a child is launched.
        target = home / 'frozen'
        target.write_text('original')
        config = json.loads((DATA / 'j5q.json').read_text())
        config['hashes'] = {str(target): sweep.sha(target)}
        sweep.verify(config)
        target.write_text('broken')
        try:
            sweep.verify(config)
        except sweep.Failure as e:
            assert e.code == 6
        else:
            raise AssertionError('changed freeze was accepted')
        expired = subprocess.run(['bash', str(DATA / 'gpu_ab.sh'), '--model', 'j5q', '--stub',
            '--out', str(home / 'budget'), '--budget-s', '1'], capture_output=True, text=True)
        assert expired.returncode == 7 and 'budget exhausted' in expired.stderr
        assert not list((home / 'budget').glob('boot-*.pid.json'))
    evidence = dict(status='ok', artifacts=artifacts, CPU_seconds_M=time.monotonic() - started,
                    full_rehearsals=2, canary_rehearsals=2,
                    negative_controls=['biased repeat', 'wide CI covering 1.0', 'missing prefix', 'seeded content mismatch',
                                       'short full and canary', 'sampling drift', 'recorded response EOS drift', 'frozen hash mutation',
                                       'impossible counters', 'null or repeated timing snapshots',
                                       'per-prefix bounds, bool/object/non-list tokens, recorded length drift'])
    sweep.write(DATA / 'cpu-validation.json', evidence)
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
