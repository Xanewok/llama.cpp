#!/usr/bin/env python3
"""CPU-only, prefix-causal lookup replay over stored HQ agent rollouts."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

NS = (32, 24, 16, 12, 8, 6, 4, 3)
ROOT = Path('/home/xanewok/repos/qwen38-eval/results/2026-09-28-dual-5060ti')
TOKENIZER = Path('/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/tokenizer.json')


class Lookup:
    def __init__(self):
        self.history = []
        self.tables = {n: {} for n in NS}

    def append(self, token):
        # Insert the previous suffix only once its continuation is observed.
        h = self.history
        end = len(h)
        for n, table in self.tables.items():
            if end >= n:
                table[tuple(h[end-n:end])] = end
        h.append(token)

    def propose(self, cap=16):
        h = self.history
        for n, table in self.tables.items():
            if len(h) >= n:
                end = table.get(tuple(h[-n:]))
                if end is not None:
                    return n, h[end:end+cap], len(h) - end
        return 0, [], 0


def output_tokens(cur, encode):
    ids, labs = [], []
    def add(text, label):
        part = encode(text)
        ids.extend(part)
        labs.extend([label] * len(part))
    if cur['r']:
        add(''.join(cur['r']).strip('\n') + '\n</think>\n\n', 'r')
    add(''.join(cur['p']), 'p')
    for i, (name, args) in enumerate(cur['t']):
        add(('\n' if i else '') + '<tool_call>\n<function=' + name + '>\n', 's')
        for key, value in args.items():
            add('<parameter=' + key + '>\n', 's')
            add(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), 'a')
            add('\n</parameter>\n', 's')
        add('</function>\n</tool_call>', 's')
    return ids, ''.join(labs)


def extract(job_path):
    from tokenizers import Tokenizer
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    attempt = job_path.parent / job['attempt']
    tok = Tokenizer.from_file(str(TOKENIZER))
    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids if text else []
    lookup = Lookup()
    rows = []
    cur = {'r': [], 'p': [], 't': [], 'invalid': False}
    excluded_calls = excluded_tokens = 0
    for path in sorted(attempt.glob('codex-state/sessions/**/rollout-*.jsonl')):
        for line in path.open():
            event = json.loads(line)
            p = event.get('payload', {})
            if event['type'] == 'response_item':
                kind = p.get('type')
                incoming = None
                if kind == 'message' and p.get('role') in ('system', 'developer', 'user'):
                    incoming = '<|im_start|>' + p['role'] + '\n' + ''.join(c.get('text', '') for c in p['content']) + '<|im_end|>\n'
                elif kind == 'function_call_output':
                    value = p['output']
                    text = value if isinstance(value, str) else json.dumps(value)
                    incoming = '<|im_start|>user\n<tool_response>\n' + text + '\n</tool_response><|im_end|>\n'
                elif kind == 'reasoning':
                    cur['r'].extend(c['text'] for c in p.get('content') or [])
                elif kind == 'message' and p.get('role') == 'assistant':
                    cur['p'].extend(c.get('text', '') for c in p['content'])
                elif kind == 'function_call':
                    try:
                        cur['t'].append((p['name'], json.loads(p['arguments'])))
                    except json.JSONDecodeError:
                        cur['invalid'] = True
                if incoming is not None:
                    for token in encode(incoming):
                        lookup.append(token)
            elif event['type'] == 'token_usage_record':
                if cur['invalid']:
                    excluded_calls += 1
                    excluded_tokens += p['usage']['output_tokens']
                    cur = {'r': [], 'p': [], 't': [], 'invalid': False}
                    continue
                ids, labs = output_tokens(cur, encode)
                matches, runs, available = [], [], []
                for pos, token in enumerate(ids):
                    n, proposal, distance = lookup.propose()
                    accepted = 0
                    for a, b in zip(proposal, ids[pos:pos+16]):
                        if a != b:
                            break
                        accepted += 1
                    matches.append(n)
                    runs.append(accepted)
                    available.append(min(distance, 16))
                    lookup.append(token)
                rows.append({'labs': labs, 'match': matches, 'run': runs, 'available': available,
                             'usage_out': p['usage']['output_tokens']})
                cur = {'r': [], 'p': [], 't': [], 'invalid': False}
    probe = job.get('probe', {})
    metrics = probe.get('metrics_delta', {})
    drafts = metrics.get('drafts', 0)
    k = round(metrics.get('draft_tokens', 0) / drafts) if drafts else 5
    tau = 1 + probe.get('mtp', {}).get('accepted_per_draft', 3)
    step_ms = 1000 * metrics.get('decode_seconds', 0) / drafts if drafts else 33
    return {'campaign': job_path.parent.name, 'task': job['task'], 'rows': rows,
            'k': k, 'tau': tau, 'step_ms': step_ms, 'decode_s': metrics.get('decode_seconds', 0),
            'excluded_calls': excluded_calls, 'excluded_tokens': excluded_tokens}


def gate(match, available, nmin, cap):
    # No access to future tokens, observed acceptance, or region labels.
    return match >= nmin and available >= cap


def alpha_for_tau(tau, k):
    lo, hi = 0., 1.
    for _ in range(50):
        mid = (lo + hi) / 2
        expected = sum(mid**i for i in range(k+1))
        if expected < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@njit(cache=True)
def expected_core(matches, runs, available, k, alpha, nmin, cap, ratio, correlated, overhead):
    size = len(runs)
    base, hybrid = [0.] * (size + 1), [0.] * (size + 1)
    look_rounds, wrong_rounds = [0.] * (size + 1), [0.] * (size + 1)
    for pos in range(size-1, -1, -1):
        survival = 1.
        b, h, lr, wr = 1., 1. + overhead, 0., 0.
        for step in range(k):
            at = min(pos + step, size - 1)
            # Sensitivity only: copy-span MTP conditional acceptance is unknown on HQ.
            a = (.975 if runs[at] else alpha) if correlated else alpha
            mass = survival * (1-a)
            dest = min(pos + step + 1, size)
            b += mass * base[dest]
            h += mass * hybrid[dest]
            lr += mass * look_rounds[dest]
            wr += mass * wrong_rounds[dest]
            survival *= a
        dest = min(pos + k + 1, size)
        b += survival * base[dest]
        h += survival * hybrid[dest]
        lr += survival * look_rounds[dest]
        wr += survival * wrong_rounds[dest]
        if matches[pos] >= nmin and available[pos] >= cap:
            accepted = min(cap, runs[pos])
            dest = min(pos + accepted + 1, size)
            h = ratio + overhead + hybrid[dest]
            lr = 1 + look_rounds[dest]
            wr = (accepted == 0) + wrong_rounds[dest]
        base[pos], hybrid[pos], look_rounds[pos], wrong_rounds[pos] = b, h, lr, wr
    return base[0], hybrid[0], look_rounds[0], wrong_rounds[0]


def expected_cost(row, k, alpha, nmin, cap, ratio, correlated=False, overhead=0.):
    return expected_core(np.asarray(row['match'], dtype=np.int32),
                         np.asarray(row['run'], dtype=np.int32),
                         np.asarray(row['available'], dtype=np.int32),
                         k, alpha, nmin, cap, ratio, correlated, overhead)


def summarize(data):
    for task in data:
        task['arrays'] = [(np.asarray(r['match'], dtype=np.int32),
                           np.asarray(r['run'], dtype=np.int32),
                           np.asarray(r['available'], dtype=np.int32)) for r in task['rows']]
        task['alphas'] = {}
        for correlated in (False, True):
            # Match task-level MTP rounds to the measured tau in both sensitivities.
            lo, hi = 0., 1.
            target_rounds = sum(len(r['labs']) for r in task['rows']) / task['tau']
            for _ in range(22):
                mid = (lo + hi) / 2
                rounds = 0.
                for row, arrays in zip(task['rows'], task['arrays']):
                    rounds += expected_core(*arrays, task['k'], mid, 99, 16, 1., correlated, 0.)[0]
                if rounds > target_rounds:
                    lo = mid
                else:
                    hi = mid
            task['alphas'][correlated] = (lo + hi) / 2
    results = []
    for cap in (8, 12, 16):
        for nmin in (4, 8, 12, 16, 24, 32):
            for marginal_ms in (1.4, 2.5):
                for correlated in (False, True):
                    groups = {}
                    for task in data:
                        alpha = task['alphas'][correlated]
                        # Charging the measured combined MTP step is conservative: skip savings unmeasured.
                        ratio = 1 + max(0, cap-task['k']) * marginal_ms / task['step_ms']
                        sums = [0.] * 4
                        for row, arrays in zip(task['rows'], task['arrays']):
                            val = expected_core(*arrays, task['k'], alpha, nmin, cap, ratio, correlated, 0.)
                            for i in range(4):
                                sums[i] += val[i]
                            missing = max(0, row['usage_out']-len(row['labs'])) / task['tau']
                            sums[0] += missing
                            sums[1] += missing
                        missing = task.get('excluded_tokens', 0) / task['tau']
                        sums[0] += missing
                        sums[1] += missing
                        if not sums[0]:
                            continue
                        entry = groups.setdefault(task['campaign'], [0., 0., 0., 0., 0.])
                        scale = task['decode_s'] / sums[0]
                        entry[0] += task['decode_s']
                        entry[1] += sums[1] * scale
                        entry[2] += sums[2]
                        entry[3] += sums[3]
                        entry[4] += 1
                    campaigns = {key: {'saved_pct': 100*(1-v[1]/v[0]),
                                        'speed_pct': 100*(v[0]/v[1]-1),
                                        'lookup_rounds': v[2], 'wrong_rounds': v[3]}
                                 for key, v in groups.items() if v[0]}
                    total_b = sum(v[0] for v in groups.values())
                    total_h = sum(v[1] for v in groups.values())
                    results.append({'nmin': nmin, 'cap': cap, 'marginal_ms': marginal_ms,
                                    'correlated': correlated, 'speed_pct': 100*(total_b/total_h-1),
                                    'saved_pct': 100*(1-total_h/total_b), 'campaigns': campaigns})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('CUDA_VISIBLE_DEVICES must be empty')
    here = Path(__file__).resolve().parent
    cache = here / 'cache'
    cache.mkdir(exist_ok=True)
    if args.extract:
        jobs = [str(p) for pattern in ('campaign-hq15-*', 'campaign-hyperqwen15-*')
                for c in sorted(ROOT.glob(pattern)) if 'aborted' not in c.name
                for p in sorted(c.glob('job-*.json')) if json.loads(p.read_text()).get('status') == 'passed'
                and (args.refresh or not (cache / (c.name + '-' + json.loads(p.read_text())['task'].replace('/', '_') + '.json')).exists())]
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
            for task in pool.map(extract, jobs):
                key = task['campaign'] + '-' + task['task'].replace('/', '_')
                (cache / (key + '.json')).write_text(json.dumps(task))
                print('extracted', key, sum(len(r['labs']) for r in task['rows']), flush=True)
    data = [json.loads(p.read_text()) for p in sorted(cache.glob('*.json'))]
    results = summarize(data)
    inventory = {'tasks': len(data), 'campaigns': len({t['campaign'] for t in data}),
                 'calls': sum(len(t['rows']) for t in data),
                 'stored_tokens': sum(len(r['labs']) for t in data for r in t['rows']),
                 'usage_tokens': sum(r['usage_out'] for t in data for r in t['rows']),
                 'excluded_calls': sum(t.get('excluded_calls', 0) for t in data),
                 'excluded_tokens': sum(t.get('excluded_tokens', 0) for t in data)}
    (here / 'estimate.json').write_text(json.dumps({'inventory': inventory, 'results': results}, indent=2) + '\n')
    print(json.dumps(inventory), flush=True)
    for result in results:
        print(result['nmin'], result['cap'], result['marginal_ms'], result['correlated'], round(result['speed_pct'], 2))


if __name__ == '__main__':
    main()
