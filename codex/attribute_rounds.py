#!/usr/bin/env python3
"""Summarize recorded host scopes; these intervals are not GPU idle time."""
import argparse
import collections
import json
from pathlib import Path
import statistics

from attribute_catchup import union


def analyze(path, rows, skip):
    selected = []
    values = collections.defaultdict(list)
    counts = collections.Counter()
    for line in path.open():
        r = json.loads(line)
        ph = r['phases']
        if [p['phase'] for p in ph] != ['V', 'C', 'D1', 'D2', 'D3', 'D4', 'D5']:
            continue
        if any(p['n_tokens'] != rows for p in ph[2:]) or ph[0]['n_tokens'] > 6 * rows:
            continue
        selected.append(r)
        if len(selected) <= skip:
            continue
        c, d = ph[1:3]
        a, b = c['end_ns'], d['begin_ns']
        sample = [(max(a, e['begin_ns']), min(b, e['end_ns'])) for e in r['events']
                  if e['kind'] == 'sampling' and e['end_ns'] > a and e['begin_ns'] < b]
        sampling = union(sample)
        values['C_end_to_D1_begin_us'].append((b - a) / 1000)
        values['sampling_inside_host_interval_us'].append(sampling / 1000)
        values['other_inside_host_interval_us'].append((b - a - sampling) / 1000)
        for p in ph[1:]:
            cs = {v['kind']: v for v in p['counters']}
            for k in ('graph_hit', 'graph_miss', 'graph_stale_alloc', 'graph_incompatible'):
                counts[p['phase'] + ':' + k] += cs.get(k, {}).get('count', 0)
            for k in ('llama_build', 'meta_build', 'sampling', 'set_inputs', 'cuda_properties', 'sync'):
                values[p['phase'] + ':' + k + '_us'].append(cs.get(k, {}).get('host_us', 0))
    n = len(selected) - skip
    if n <= 0:
        raise ValueError('no selected rounds')
    return {'artifact': str(path.resolve()), 'rounds_M': n, 'skip_complete_rounds': skip,
            'unit': 'host wall and scope unions per complete V C D^5 round; missing counters are zero',
            'graph_counts_M': dict(counts),
            'median_M': {k: round(statistics.median(v), 3) for k, v in values.items()}}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('trace', type=Path)
    p.add_argument('--rows', type=int, default=2)
    p.add_argument('--skip', type=int, default=3)
    args = p.parse_args()
    print(json.dumps(analyze(args.trace, args.rows, args.skip), indent=2))
