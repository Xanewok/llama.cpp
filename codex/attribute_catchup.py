#!/usr/bin/env python3
"""Read recorded CUPTI launch blocks; report C -> D1 without running CUDA."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import statistics
import sys


def union(intervals):
    end = total = 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def analyze(pack, path):
    sys.path.insert(0, str(pack))
    import attrib_pack as ap

    tr = ap.Trace(path, need_runtime=True)
    blocks = ap.llama_blocks(tr, 14)
    vi = [i for i, b in enumerate(blocks) if b['kind'] == 'V']
    selected = []
    for a, b in zip(vi, vi[1:]):
        group = blocks[a:b]
        if len(group) != 7 or group[0]['host_s'] < tr.t0 + 750_000_000:
            continue
        if any(not x['dev'][0] for x in group):
            continue
        if any('gated_delta_net' in tr.names[k[3]] for x in group[1:] for k in x['kernels']):
            raise ValueError('target kernel in inferred draft phase')
        selected.append(group)
    if len(selected) < 30:
        raise ValueError('fewer than 30 complete V C D^5 rounds')
    values = collections.defaultdict(list)
    gaps = []
    runtime_starts = [r[0] for r in tr.runtime]
    for group in selected:
        c, d = group[1:3]
        a = max(k[1] for k in c['dev'][0])
        b = min(k[0] for k in d['dev'][0])
        gaps.append((a, b))
        c_start = min(k[0] for k in c['dev'][0])
        d_launch = d['host_s']
        first_input = c.get('h2d', (d_launch,))[0]
        values['C_gpu_span_us'].append((a - c_start) / 1000)
        values['C_to_D1_gpu_gap_us'].append((b - a) / 1000)
        values['C_last_launch_to_D1_first_launch_us'].append((d_launch - c['host_e']) / 1000)
        values['pre_input_exposed_us'].append(max(0, first_input - a) / 1000)
        values['input_to_launch_exposed_us'].append(max(0, d_launch - max(a, first_input)) / 1000)
        values['launch_to_gpu_us'].append(max(0, b - max(a, d_launch)) / 1000)
        iv = collections.defaultdict(list)
        lo = max(0, bisect.bisect_left(runtime_starts, a) - 1)
        hi = bisect.bisect_left(runtime_starts, b)
        for s, e, name, _, _ in tr.runtime[lo:hi]:
            if e > a:
                iv[name].append((max(a, s), min(b, e)))
        for key in ('cudaStreamSynchronize', 'cudaEventSynchronize', 'cudaGraphInstantiateWithFlags',
                    'cudaStreamBeginCapture', 'cudaStreamEndCapture'):
            values[key + '_inside_gap_us'].append(union(iv[key]) / 1000)
    result = {'artifact': str(path.resolve()), 'rounds_M': len(selected),
            'selection': 'seven launch blocks between V blocks; skip first 750 ms; no duration trim',
            'median_M': {k: round(statistics.median(v), 3) for k, v in values.items()}}
    db = ap.open_ro(path)
    sampled = db.execute("SELECT 1 FROM sqlite_master WHERE name='COMPOSITE_EVENTS'").fetchone()
    db.close()
    if sampled:
        import host_sampling as hs
        tids = hs.decode_threads(tr, gaps[0][0], gaps[-1][1])
        events, chains, _, _, _ = hs.load_samples(path, tids, gaps[0][0], gaps[-1][1])
        ts = collections.defaultdict(list)
        counts = collections.Counter()
        gap_starts = [a for a, _ in gaps]
        for eid, t, tid in events:
            ts[tid].append(t)
            i = bisect.bisect_right(gap_starts, t) - 1
            if i >= 0 and t < gaps[i][1]:
                top, _, _, _ = hs.classify_chain(chains[eid])
                counts[top[0]] += 1
        spacing = hs.sample_spacing_ms(ts)
        result['cpu_sampling_M'] = {
            'spacing_us': spacing * 1000, 'gap_samples': sum(counts.values()),
            'unit': 'sample-count estimate of mean CPU running time per selected C->D1 gap; not exact scopes',
            'top': [{'symbol': k, 'samples': n, 'estimated_us_per_gap': n * spacing * 1000 / len(gaps)}
                    for k, n in counts.most_common(20)]}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('sqlite', type=Path)
    p.add_argument('--pack', type=Path, default=Path('/home/xanewok/repos/attrib-pack'))
    args = p.parse_args()
    print(json.dumps(analyze(args.pack, args.sqlite), indent=2))
