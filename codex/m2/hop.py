#!/usr/bin/env python3
"""Per draft-step host hop from round traces, per depth window (depth-*.json mono_ns of the run dir).
period = begin(D_i+1) - begin(D_i); sync0 = device-0 sync wait counted in D_i; hop = period - sync0.
DC rounds (chain): wall and sync of the one chain decode. Usage: hop.py RUN_DIR [max_lines]"""
import json, sys, statistics, itertools, collections, glob, os

run = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10**7
wins = {}
for f in glob.glob(os.path.join(run, 'depth-*.json')):
    m = json.load(open(f))['mono_ns']
    wins[int(os.path.basename(f)[6:-5])] = (m[0], m[1])

def depth_of(ns):
    for d, (a, b) in wins.items():
        if a <= ns <= b:
            return d
    return None

acc = collections.defaultdict(lambda: collections.defaultdict(list))
for line in itertools.islice(open(os.path.join(run, 'round-trace.jsonl')), lim):
    try:
        r = json.loads(line)
    except Exception:
        continue
    ph = r.get('phases', [])
    if not ph:
        continue
    dep = depth_of(ph[0]['begin_ns'])
    if dep is None:
        continue
    for p in ph:
        if p['phase'] == 'DC':
            sync = sum(c['host_us'] for c in p['counters'] if c['kind'] == 'sync' and c['device'] == 0)
            k = (dep, 'DC', p['n_tokens'])
            acc[k]['dc_wall'].append((p['end_ns'] - p['begin_ns']) / 1e3)
            acc[k]['dc_sync0'].append(sync)
    d = [p for p in ph if p['phase'].startswith('D') and p['phase'][1:].isdigit()]
    if len(d) < 3:
        continue
    nt = d[0]['n_tokens']
    if any(p['n_tokens'] != nt for p in d):
        continue
    k = (dep, 'D', nt)
    for a, b in zip(d, d[1:]):
        per = (b['begin_ns'] - a['begin_ns']) / 1e3
        sync = sum(c['host_us'] for c in a['counters'] if c['kind'] == 'sync' and c['device'] == 0)
        acc[k]['period'].append(per)
        acc[k]['sync0'].append(sync)
        acc[k]['hop'].append(per - sync)
for key, m in sorted(acc.items()):
    n = max(len(v) for v in m.values())
    print(f'depth={key[0]} {key[1]} rows={key[2]} n={n}', {k: round(statistics.median(v), 1) for k, v in m.items()})
