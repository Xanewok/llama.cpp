"""Adapt the depth harness for closest-supported configs and explicit history orders."""
import copy
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import tempfile

from harness import depth, lib
from harness.__main__ import main as harness_main

EVAL = Path('/home/xanewok/repos/qwen38-eval')
STACK = Path('/home/xanewok/repos/qwen38-eval-worktrees/wt/stack10b/codex')
sys.path.insert(0, str(STACK))
import gpu_stack10

ORDERS = {'short': [8192, 90112], 'long': [8192, 32768, 90112]}
original_boot, original_evaluate, original_finish = depth.boot, depth.evaluate_boot, depth.finish


def boot(x, arm, port, dest, canary, stub=False):
    if x.get('history'):
        x['depth']['depths'] = ORDERS[arm['name']]
    command, env = original_boot(x, arm, port, dest, canary, stub)
    env.pop('LLAMA_ROUND_TRACE', None)
    env['OPENBLAS_NUM_THREADS'] = '1'
    return command, env


def evaluate(x, dest, arm, results, canary=False):
    if depth.streams(x) > 1 or x['depth']['identity'] != 'off':
        failures = original_evaluate(x, dest, arm, results, canary)
        if depth.streams(x) > 1 and not failures:
            for n in depth.depths_of(x, canary):
                if n in results and results[n]['runs'] and results[n]['runs'][-1]['arm'] == arm:
                    record = json.loads((dest / f'depth-{n}.json').read_text())
                    results[n]['runs'][-1]['cold_prefill_ms_M'] = [depth.final_timings(p)['prompt_ms'] for p in record['prefill']]
        return failures
    if not x['depth'].get('identity_off_reason'):
        raise lib.Fail('identity off needs a reason', 'integrity')
    failures = []
    for n in depth.depths_of(x, canary):
        try:
            record = json.loads((dest / f'depth-{n}.json').read_text())
            result, ident = depth.gate(x['depth'], record, depth.first_prefill(n), k=depth.arm_k(x, arm))
            depth.settle(results, 'off', arm, n, ident)['runs'].append({'arm': arm, **result})
        except lib.Fail as error:
            failures.append(error)
        except (OSError, ValueError, KeyError) as error:
            failures.append(lib.Fail(f'unreadable record: {error}', 'deterministic'))
    return failures + (depth.proofs(x, dest, arm) if canary else [])


def stats(values):
    n = len(values)
    return {'mean': statistics.mean(values), 'se': statistics.stdev(values) / math.sqrt(n) if n > 1 else None,
            'n_boots': n, 'boots': values}


def finish(x, results):
    if x.get('history'):
        results = {90112: results[90112]}
    result = original_finish(x, results)
    for n, row in result.items():
        for arm in depth.arm_names(x):
            runs = [r for r in row['runs'] if r['arm'] == arm]
            ms = [statistics.mean(s['steady_ms_per_round_D'] for s in r['streams']) if 'streams' in r else r['steady_ms_per_round_D'] for r in runs]
            tok = [r['agg_tok_s'] if 'streams' in r else 1000*r['timings']['predicted_n']/r['timings']['predicted_ms'] for r in runs]
            row[f'{arm}_ms_per_round_D_stats'] = stats(ms)
            row[f'{arm}_tok_s_D_stats'] = stats(tok)
            cold = [statistics.mean(r['cold_prefill_ms_M']) if 'streams' in r else r['timings']['prompt_ms'] for r in runs]
            row[f'{arm}_cold_prefill_ms_M_stats'] = stats(cold)
        arms = depth.arm_names(x)
        for other in arms[1:]:
            for metric in ('ms_per_round_D', 'tok_s_D', 'cold_prefill_ms_M'):
                a = row[f'{arms[0]}_{metric}_stats']['boots']
                b = row[f'{other}_{metric}_stats']['boots']
                if len(a) != len(b): raise lib.Fail('unequal boot counts', 'integrity')
                row[f'{other}_minus_{arms[0]}_{metric}_stats'] = stats([y-z for z, y in zip(a, b)])
    return result


def history_result(x, out):
    records = {}
    for i, arm in enumerate(x['plan']):
        record = json.loads((Path(out) / f'{i}-{arm}/depth-90112.json').read_text())
        depth.gate(x['depth'], record, depth.first_prefill(90112), k=5)
        ident = lib.token_identity(record)
        if arm in records and records[arm] != ident:
            raise lib.Fail('same-order repeat identity failed', 'integrity')
        records[arm] = ident
    if set(records) != {'short', 'long'}:
        raise lib.Fail('history needs both orders', 'integrity')
    a, b = records['short'], records['long']
    first = next((i for i, pair in enumerate(zip(a[0], b[0])) if pair[0] != pair[1]), None)
    answer = {'order_dependent': a != b, 'tokens_equal': a[0] == b[0], 'counters_equal': a[1] == b[1],
              'first_different_token_D': first, 'short_counters': a[1], 'long_counters': b[1]}
    lib.write_json(Path(out) / 'history-diagnostic.json', answer)
    return answer


def must_fail(fn):
    try:
        fn()
    except (lib.Fail, ValueError):
        return
    raise AssertionError('negative control escaped')


def fixtures():
    depth.fixture_test(EVAL / 'orchestration/gpuq/fixtures')
    log = (EVAL / 'results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/server.log').read_text()
    patterns = [r'backend sampling .*SPLIT_MODE_TENSOR.*using CPU']
    lib.verify_log_patterns(log, patterns, 'recorded CPU fallback')
    must_fail(lambda: lib.verify_log_patterns(log.replace('using CPU', 'using GPU'), patterns, 'CPU fallback control'))
    np1 = EVAL / 'results/2026-10-09-s10-depth-iq4/full-a3/0-s9p/depth-90112.json'
    a = json.loads(np1.read_text())
    d = {'min_steady_intervals': 40, 'min_predicted': 256, 'k': 5}
    assert gpu_stack10.steady_with_periods(a, 40, 5)['untrimmed_periods']
    depth.gate(d, a, depth.first_prefill(90112), k=5)
    bad = copy.deepcopy(a)
    bad['chunks'][-1]['timings']['cache_n'] = 1
    must_fail(lambda: depth.gate(d, bad, depth.first_prefill(90112), k=5))
    bad = copy.deepcopy(a)
    next(c for c in bad['chunks'] if c.get('tokens'))['tokens'][0] += 1
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent / 'evidence') as tmp:
        x = {'plan': ['short', 'long', 'long', 'short'], 'depth': d}
        for i, arm in enumerate(x['plan']):
            dest = Path(tmp) / f'{i}-{arm}'; dest.mkdir()
            (dest / 'depth-90112.json').write_text(json.dumps(a if arm == 'short' else bad))
        assert history_result(x, tmp)['order_dependent']
        (Path(tmp) / '3-short/depth-90112.json').write_text(json.dumps(bad))
        must_fail(lambda: history_result(x, tmp))
        for i, arm in enumerate(x['plan']):
            (Path(tmp) / f'{i}-{arm}/depth-90112.json').write_text(json.dumps(a))
        assert not history_result(x, tmp)['order_dependent']
        off = {'plan': ['vanilla', 's9p', 's10'], 'arms': {v: {} for v in ('vanilla', 's9p', 's10')},
               'depth': d | {'depths': [90112], 'identity': 'off', 'identity_off_reason': 'different configs'}}
        dest = Path(tmp) / 'off'; dest.mkdir()
        (dest / 'depth-90112.json').write_text(json.dumps(a))
        results = {}
        for arm in off['plan']:
            assert evaluate(off, dest, arm, results) == []
        summary = finish(off | {'depth': off['depth'] | {'request': a['request']}}, results)
        assert summary[90112]['vanilla_ms_per_round_D_stats']['n_boots'] == 1
    np2 = EVAL / 'results/2026-10-07-gdn-getter-gate-np2-busy-j5q/canary-a1/0-up/depth-8192.json'
    r = json.loads(np2.read_text())
    w = depth.overlap(r['streams'], 0.8)
    for stream in r['streams']:
        depth.gate(d, stream, depth.replay_gate({}, 8192), tuple(w['window_ns']), k=5)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent / 'evidence') as tmp:
        x = {'plan': ['vanilla', 's9p', 's10', 's10', 's9p', 'vanilla'],
             'arms': {v: {} for v in ('vanilla', 's9p', 's10')},
             'depth': d | {'streams': 2, 'depths': [8192], 'identity': 'off', 'identity_off_reason': 'different configs', 'request': a['request']}}
        dest = Path(tmp)
        (dest / 'depth-8192.json').write_text(json.dumps(r))
        results = {}
        for arm in x['plan']:
            assert evaluate(x, dest, arm, results) == []
        row = finish(x, results)[8192]
        assert row['vanilla_ms_per_round_D_stats']['n_boots'] == 2
        assert row['s10_minus_vanilla_tok_s_D_stats']['se'] == 0
        assert row['vanilla_cold_prefill_ms_M_stats']['mean'] > 0
    shifted = copy.deepcopy(r['streams'])
    for c in shifted[1]['chunks']:
        if 't' in c: c['t'] += 10**15
    must_fail(lambda: depth.overlap(shifted, 0.8))
    actual = Path('/home/xanewok/repos/qwen38-eval-worktrees/wt/s10-identity/codex/stack10/identity/results-90k')
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent / 'evidence') as tmp:
        x = {'plan': ['short', 'long', 'long', 'short'], 'depth': d}
        for i, arm in enumerate(x['plan']):
            dest = Path(tmp) / f'{i}-{arm}'; dest.mkdir()
            source = actual / ('full-a2' if arm == 'short' else 'full-a3') / '0-stock/depth-90112.json'
            (dest / 'depth-90112.json').write_bytes(source.read_bytes())
        assert history_result(x, tmp)['order_dependent']
    print('real SSE, np1/np2 timing, token, cache, overlap, history and summary controls PASS')


def runtime_guards(x, canary):
    start, stop, journal = lib.Server.start, lib.Server.stop, lib.record_journal
    def expire(signum, frame):
        raise lib.Fail('budget: boot wall expired', 'timeout')
    def start_capped(server):
        pid = start(server)
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, x['budget']['canary_s'] - 60 if canary else x['depth']['boot_timeout_s'])
        return pid
    def stop_capped(server, grace=30):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return stop(server, min(grace, 30))
    def scoped_journal(since, path, journal_file=None):
        n = journal(since, path, journal_file)
        if n is None: raise lib.Fail('kernel journal unreadable', 'integrity')
        if n: raise lib.Fail('Xid since boot start', 'hazard')
        return n
    lib.Server.start, lib.Server.stop, lib.record_journal = start_capped, stop_capped, scoped_journal


def main():
    argv = sys.argv[1:]
    x = lib.read_experiment(argv[1])
    if any((arm['root'] / 'DRY-STAND-IN').exists() for arm in x['arms'].values()) and any(flag in argv for flag in ('--canary', '--execute')):
        raise lib.Fail('DRY STAND-IN refuses GPU execution', 'integrity')
    depth.boot, depth.evaluate_boot, depth.finish = boot, evaluate, finish
    lib.steady_intervals = gpu_stack10.steady_with_periods
    if '--preflight' in argv:
        fixtures()
        # gpuq hashes model pins; avoid reading multi-GB models in the worker preflight.
        depth.check_inputs(x, full_model_sha=False)
        return 0
    runtime_guards(x, '--canary' in argv)
    rc = harness_main(argv)
    if rc == 0 and x.get('history') and '--canary' in argv:
        out = Path(argv[argv.index('--out') + 1])
        record = (out / '0-long/depth-90112.json').read_bytes()
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            controls = x | {'plan': ['short', 'long', 'long', 'short']}
            for i, arm in enumerate(controls['plan']):
                dest = Path(tmp) / f'{i}-{arm}'; dest.mkdir()
                (dest / 'depth-90112.json').write_bytes(record)
            assert not history_result(controls, tmp)['order_dependent']
            changed = json.loads(record)
            next(c for c in changed['chunks'] if c.get('tokens'))['tokens'][0] += 1
            (Path(tmp) / '3-short/depth-90112.json').write_text(json.dumps(changed))
            must_fail(lambda: history_result(controls, tmp))
        lib.write_json(out / 'history-canary-controls.json', {'pass': True, 'one_real_boot': True})
    if rc == 0 and x.get('history') and '--execute' in argv:
        history_result(x, argv[argv.index('--out') + 1])
    return rc


if __name__ == '__main__':
    try:
        sys.exit(main())
    except lib.Fail as error:
        print(str(error), file=sys.stderr)
        sys.exit(error.rc)
