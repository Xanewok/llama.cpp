#!/usr/bin/env python3
"""runs.json + exported sqlite files -> attrib_pack.py pack manifest.

llama: <window>.sqlite by name. hq: nsys repeat ranges, whatever the report suffix scheme. The prefill window gets the file
whose kernels contain the prefill marker kernels (content, not name); decode windows get the remaining files in range
order (trailing integer of the file name, then mtime). A window with no file keeps a nonexistent path, so the pack
rejects it and says why instead of silently shrinking; the pack itself rejects a file whose content is the wrong kind.
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from attrib_pack import MARKERS  # noqa: E402

WINDOW_TOKENS = {'prefill-32k': 32768, 'decode-8k': 8192, 'decode-8k-b': 8192, 'decode-90k': 90000}


def has_prefill_marker(path):
    try:
        db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        names = [r[0] for r in db.execute('select distinct s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                                          'join StringIds s on s.id = k.demangledName')]
        db.close()
    except sqlite3.DatabaseError:
        return False
    return any(MARKERS['hq'].search(n) for n in names)


def range_order(p):
    m = re.search(r'(\d+)$', p.stem)
    return (int(m.group(1)) if m else 0, p.stat().st_mtime, p.name)


def assign_hq(windows, out_dir):
    found = sorted(out_dir.glob('*.sqlite'), key=range_order)
    pre = [p for p in found if has_prefill_marker(p)]
    dec = [p for p in found if p not in pre]
    files = {}
    for n in windows:
        pool = pre if n.startswith('prefill') else dec
        files[n] = pool.pop(0) if pool else out_dir / f'missing-{n}.sqlite'
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['llama', 'hq'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--min-rounds', type=int, default=40)
    a = ap.parse_args()
    runs = json.loads((a.out_dir / 'runs.json').read_text())
    windows = [w['name'] for w in runs['windows']]
    files = {n: a.out_dir / f'{n}.sqlite' for n in windows} if a.engine == 'llama' else assign_hq(windows, a.out_dir)
    traces = []
    for w in runs['windows']:
        n = w['name']
        spec = {'id': f'{a.engine}-{n}', 'sqlite': str(files[n])}
        if n.startswith('prefill'):
            spec.update(kind='prefill', tokens=WINDOW_TOKENS[n])
        else:
            spec.update(kind='decode', depth=WINDOW_TOKENS[n], k=a.k, group='depth')
            tw = w.get('twin') or {}
            if a.engine == 'llama':
                if w.get('tok_per_round'):
                    spec['tok_per_round'] = w['tok_per_round']
                if tw.get('ms_per_round') and not tw.get('cache_miss'):
                    spec['unprofiled'] = {'ms': tw['ms_per_round'], 'source': 'llama twin request: same prompt and seed, no profiler, '
                                          'server predicted_ms / (draft_n / k) (inferred rounds)'}
            elif tw.get('ms_per_step'):
                spec['unprofiled'] = {'ms': tw['ms_per_step'], 'source': 'hq twin request: median SSE inter-chunk time, '
                                      'no profiler (inferred: one chunk per engine step)'}
        traces.append(spec)
    a.manifest.write_text(json.dumps({'title': a.title, 'model': a.model, 'engine': a.engine, 'k': a.k,
                                      'min_rounds': a.min_rounds, 'traces': traces}, indent=1))


if __name__ == '__main__':
    main()
