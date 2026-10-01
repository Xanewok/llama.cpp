#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import experiment


def main():
    zeros = '{"cmd":"' + '[0, 0, 0, 0],\\n' * 800 + '"}'
    planning = SimpleNamespace(quants=['IQ4_XS', 'J5q'], arms=list(experiment.ARMS)[:6],
        n=17, per_boot=8, up_dir=None)
    fixtures = [dict(id='letters'), dict(id='zeros')]
    plan = experiment.schedule(planning, fixtures)
    trials = [t for boot in plan for t in boot['trials']]
    assert len(trials) == 17 * 2 * 2 * 6
    assert len({t['id'] for t in trials}) == len(trials)
    assert len({t['seed'] for t in trials}) == len(trials)
    assert plan == experiment.schedule(planning, fixtures)
    assert [p['arm'] for p in plan[:6]] != [p['arm'] for p in plan[12:18]]
    terminal = dict(stop=True, content='', timings=dict(predicted_n=8192))
    sse = ['data: ' + json.dumps(dict(content=zeros, tokens=[1, 2], stop=False)),
           'data: ' + json.dumps(terminal)]
    result = experiment.collect(sse, dict(id='test'), 8192)
    assert result['repetition'] and result['limit_reached']
    try:
        experiment.collect(sse[:1], dict(id='test'), 8192)
    except RuntimeError:
        pass
    else:
        raise AssertionError('incomplete stream counted as a completed negative')
    assert experiment.request_body([1, 2], dict(seed=12), SimpleNamespace(length=64), {})['cache_prompt'] is False
    try:
        experiment.request_body([1], dict(seed=12), SimpleNamespace(length=64), dict(seed=99))
    except RuntimeError:
        pass
    else:
        raise AssertionError('seed override accepted')
    temp_root = experiment.WORKTREE / 'build-codex-test'
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as d:
        source = Path(d) / 'source'
        source.mkdir()
        (source / 'llama-server').write_text('fixture; never executable')
        (source / 'libfixture.so').write_text('library fixture')
        (source / 'SHA256SUMS').write_text(''.join(experiment.digest(source / n) + '  ' + n + '\n'
                                                  for n in ['llama-server', 'libfixture.so']))
        frozen = Path(d) / 'frozen'
        experiment.freeze(source, frozen)
        assert (frozen / 'ra-continuation-srv').read_text() == 'fixture; never executable'
        assert not (frozen / 'llama-server').exists()
        (source / 'libfixture.so').write_text('corrupted')
        try:
            experiment.freeze(source, frozen)
        except RuntimeError:
            pass
        else:
            raise AssertionError('corrupted library accepted')
    with tempfile.TemporaryDirectory(dir=temp_root) as d:
        out = Path(d)
        (out / 'trials').mkdir()
        statistical_plan = []
        for q in ['IQ4_XS', 'J5q']:
            for arm in ['no-spec', 'S5']:
                trials = [dict(id=q + arm + str(i), prefix='one') for i in range(8)]
                statistical_plan.append(dict(quant=q, arm=arm, build='stack2', trials=trials))
                for t in trials:
                    (out / 'trials' / (t['id'] + '.json')).write_text(json.dumps(dict(t,
                        repetition=(q == 'IQ4_XS' and arm == 'S5'))))
        experiment.summarize(out, statistical_plan)
        tests = json.loads((out / 'primary-tests.json').read_text())
        assert len(tests) == 3 and tests[0]['reject'] and all(t['family_complete'] for t in tests)
        (out / 'trials' / 'J5qS50.json').unlink()
        experiment.summarize(out, statistical_plan)
        assert all(t['p_two_sided_D'] is None for t in json.loads((out / 'primary-tests.json').read_text()))
    original_run, original_http, original_argv = experiment.subprocess.run, experiment.http, sys.argv
    def forbidden(*_, **__):
        raise AssertionError('plan mode accessed a process or HTTP endpoint')
    try:
        experiment.subprocess.run = forbidden
        experiment.http = forbidden
        sys.argv = ['experiment.py', '--out', str(temp_root / 'plan'), '--n', '2', '--length', '32']
        experiment.main()
    finally:
        experiment.subprocess.run, experiment.http, sys.argv = original_run, original_http, original_argv
    print('M: interleaving, independent seeds, SSE and Fisher-family censoring, controlled requests, renamed frozen binaries and hash failure control pass')


if __name__ == '__main__':
    main()
