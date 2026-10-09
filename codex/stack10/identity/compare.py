"""Compare native token IDs and draft counters in saved np1 completion records."""
import argparse
import hashlib
import json
from pathlib import Path


COUNTERS = ('predicted_n', 'draft_n', 'draft_n_accepted', 'prompt_n', 'cache_n')


def read_record(path):
    path = Path(path)
    if path.stat().st_size > 200 * 1024 * 1024:
        raise ValueError(f'record exceeds 200 MiB: {path}')
    return json.loads(path.read_text())


def tokens(record):
    return [t for c in record['chunks'] for t in (c.get('tokens') or [])]


def counters(record):
    final = next(c for c in reversed(record['chunks']) if c.get('stop') is True)
    return {k: final['timings'].get(k) for k in COUNTERS}


def compare(reference, candidate):
    a, b = tokens(reference), tokens(candidate)
    first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    counter_first = next((i for i, (x, y) in enumerate(zip(reference['chunks'], candidate['chunks']))
                          if {k: x.get('timings', {}).get(k) for k in ('draft_n', 'draft_n_accepted')} !=
                          {k: y.get('timings', {}).get(k) for k in ('draft_n', 'draft_n_accepted')}), None)
    return {
        'request_equal': reference['request'] == candidate['request'],
        'tokens_equal': a == b,
        'final_counters_equal': counters(reference) == counters(candidate),
        'first_token_index0': first,
        'first_tokens': [a[first] if first < len(a) else None, b[first] if first < len(b) else None] if first is not None else None,
        'first_draft_counter_chunk0': counter_first,
        'first_draft_counters': [{k: record['chunks'][counter_first]['timings'].get(k) for k in ('predicted_n', 'draft_n', 'draft_n_accepted')}
                                 for record in (reference, candidate)] if counter_first is not None else None,
        'reference_counters': counters(reference),
        'candidate_counters': counters(candidate),
    }


def matches(row):
    return row['request_equal'] and row['tokens_equal'] and row['final_counters_equal'] and row['first_draft_counter_chunk0'] is None


def analyze(experiment, out):
    import tomllib
    x = tomllib.loads(Path(experiment).read_text())
    spec = x['diagnostic']
    out = Path(out)
    rows = []
    stock_records = []
    for i, arm in enumerate(x['plan']):
        path = out / f'{i}-{arm}' / 'depth-32768.json'
        if not path.exists():
            continue
        record = read_record(path)
        reference = read_record(spec['reference'])
        archived_stock = read_record(spec['archived_stock'])
        row = {'boot': i, 'arm': arm, 'reference': compare(reference, record), 'archived_stock': compare(archived_stock, record)}
        rows.append(row)
        if arm == 'stock':
            stock_records.append(record)
    complete = len(rows) == len(x['plan'])
    repeat_equal = len(stock_records) == 2 and matches(compare(*stock_records))
    reproduced = repeat_equal and all(matches(r['archived_stock']) and not matches(r['reference']) for r in rows if r['arm'] == 'stock')
    result = {'complete': complete, 'stock_repeat_equal': repeat_equal, 'failure_reproduced': reproduced,
              'restoring_arms': [r['arm'] for r in rows if r['arm'] != 'stock' and matches(r['reference'])], 'boots': rows}
    (out / 'identity-diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'boots'}))
    return 0 if complete and reproduced else 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    result = compare(read_record(args.reference), read_record(args.candidate))
    result['files'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.reference, args.candidate)}
    print(json.dumps(result, indent=2))
