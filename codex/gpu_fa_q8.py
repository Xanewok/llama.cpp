"""Stack10 uses the stack9 frozen-binary, PID, budget and journal guards."""
import copy
import json
import os
from pathlib import Path
import re
import sys
import statistics
import tempfile
import tomllib
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_stack9 as base


original_steady = base.lib.steady_intervals
original_finish = base.depth.finish


def steady_with_periods(record, minimum=40, k=5, window=None):
    result = original_steady(record, minimum, k, window)
    points = {}
    for c in record['chunks']:
        t = c.get('timings')
        if not isinstance(t, dict) or not all(isinstance(t.get(key), (int, float)) for key in
                ('predicted_ms', 'predicted_n', 'draft_n', 'draft_n_accepted')):
            continue
        n = t['draft_n']
        if n >= 0 and n % k == 0:
            points[int(n/k)] = (t['predicted_ms'], t['predicted_n'], c.get('t', -1))
    periods = []
    for a, b in zip(sorted(points), sorted(points)[1:]):
        before, after = points[a], points[b]
        reasons = []
        if a < 3:
            reasons.append('warmup')
        if b != a + 1:
            reasons.append('missing-round-boundary')
        if after[1] > result['timings']['predicted_n'] - (k + 1):
            reasons.append('tail')
        if window and not (window[0] <= before[2] <= after[2] <= window[1]):
            reasons.append('outside-overlap')
        if after[0] <= before[0]:
            reasons.append('nonpositive-period')
        periods.append(dict(first_round=a, last_round=b, ms_D=after[0]-before[0], excluded=reasons))
    result['untrimmed_periods'] = periods
    result['median_trimmed_rounds_D'] = 0
    result['excluded_periods_D'] = sum(bool(r['excluded']) for r in periods)
    return result


def counter_match(runs, arms):
    keys = ('predicted_n', 'draft_n', 'draft_n_accepted')
    by_arm = {a: [tuple(tuple(st['timings'][key] for key in keys) for st in run['streams'])
                  for run in runs if run['arm'] == a] for a in arms}
    return bool(by_arm[arms[0]]) and all(by_arm[a] == by_arm[arms[0]] for a in arms[1:]), by_arm


def finish_with_controls(x, results):
    if base.depth.streams(x) > 1 and x['id'] != 's10c-q8dkv-np2-real-j5q':
        for depth, result in results.items():
            matched, counts = counter_match(result['runs'], base.depth.arm_names(x))
            result['matched_counters'] = counts
            if not matched:
                raise base.lib.Fail(f'INVALID: depth {depth}: exact optimization arms have unmatched decode counters', 'insufficient_data')
    result = original_finish(x, results)
    if base.depth.streams(x) > 1:
        for row in result.values():
            for arm in base.depth.arm_names(x):
                timings = [st['timings'] for r in row['runs'] if r['arm'] == arm for st in r['streams']]
                row[f'{arm}_tokens_per_round_D'] = 1 + base.depth.arm_k(x, arm)*sum(t['draft_n_accepted'] for t in timings)/sum(t['draft_n'] for t in timings)
    return result


def install_controls():
    base.lib.steady_intervals = steady_with_periods
    base.depth.finish = finish_with_controls


def fixture_checks():
    base.fixture_checks()
    from harness import lib, mkitem
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        for source in (root/'items').glob('fa-q8-*.toml.in'):
            p = Path(tmp)/source.name.removesuffix('.in')
            p.write_text(source.read_text().replace('@STACK10_SHA@', '0'*40).replace('@NEW_SHA@', '1'*40))
            x = lib.read_experiment(p)
            assert type(x['priority']) is int and len(x['canary']['plan']) == 1
            assert x['budget']['canary_s'] <= 600
            assert len(x['plan'])*mkitem.boot_wall(x) + x['budget']['canary_s'] <= 0.7*mkitem.budget(x)['budget_s']
            assert all(x['plan'].count(a) >= 2 for a in x['arms'])
            assert '--cache-ram' not in x['server']['args']
            assert x['depth'].get('decode_start', 'joint') == 'joint'
            assert x['depth']['request']['ignore_eos'] is True
            for arm in x['arms'].values():
                for pattern in arm['canary_log']:
                    re.compile(pattern)
            record = (base.REAL/'server.log').read_text()
            # A recorded GPU boot without the new kernels must fail the candidate proof.
            for marker in (r'alpha_gate_fusion: device=0 width=[1-9]\d* rows=[1-9]\d*',
                           r'fattn_q8_mq: device=0 dk=256 ncols1=8 ncols2=8 rows=\d+ seqs=1\b', 'persistent rotations backend='):
                try:
                    lib.verify_log_patterns(record, [marker], 'absent stack10 proof')
                except lib.Fail as e:
                    assert e.cls == 'integrity'
                else:
                    raise AssertionError('missing stack10 engagement escaped')
    cache_log = (root/'stack9-evidence/recorded-getter.log').read_text()
    f16_cache = r'llama_kv_cache: size =[^\n]*,\s+1 layers,[^\n]*K \(f16\):[^\n]*V \(f16\):'
    base.lib.verify_log_patterns(cache_log, [f16_cache], 'recorded draft cache type')
    try:
        base.lib.verify_log_patterns(cache_log, [f16_cache.replace('f16','q8_0')], 'wrong recorded draft cache type')
    except base.lib.Fail:
        pass
    else:
        raise AssertionError('wrong draft KV type escaped')
    # Exercise acceptance and tokens/round using an actual recorded concurrent decode.
    r = json.loads((base.REAL_NP2/'depth-8192.json').read_text())
    for stream in r['streams']:
        tokens, counters = lib.token_identity(stream)
        t = counters
        assert tokens and 0 <= t['draft_n_accepted'] <= t['draft_n']
        assert 1 <= 1 + t['draft_n_accepted']/(t['draft_n']/5) <= 6
    stream = r['streams'][0]
    periods = steady_with_periods(stream, 40, 5)
    assert periods['untrimmed_periods'] and periods['excluded_periods_D'] > 0
    assert periods['steady_ms_per_round_D'] == original_steady(stream, 40, 5)['steady_ms_per_round_D']
    t = base.depth.final_timings(stream)
    runs = [dict(arm=a, streams=[dict(timings=copy.deepcopy(t))]) for a in ('up','patch')]
    assert counter_match(runs, ['up','patch'])[0]
    runs[1]['streams'][0]['timings']['draft_n_accepted'] += 1
    assert not counter_match(runs, ['up','patch'])[0], 'real-record counter mutation escaped'
    window = tomllib.loads((root/'bw-fa-q8-inkernel.toml').read_text())
    assert window['kind'] == 'filler' and window['priority'] == 18
    assert window['host_sensitive'] is False and window['preemptible'] is False
    assert window['run']['env']['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
    print('fa-q8 window: stack10 recorded parsers, engagement negative controls, item structure PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixtures']:
        sys.exit(fixture_checks())
    install_controls()
    base.install_runtime_guards(sys.argv[1:])
    base.campaign.harness_cmd = lambda *args: [sys.executable, '-P', str(Path(__file__).resolve()), *map(str, args)]
    sys.exit(base.harness_main())
