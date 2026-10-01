#!/usr/bin/env python3
"""CPU analysis of complete, filtered-target request captures."""
import argparse
import bisect
import json
import math
from pathlib import Path

import catch_replay as replay
from experiment import atomic
from runaways import spans


def token_bytes(tokens):
    pieces = [bytes.fromhex(t['piece_hex']) for t in tokens]
    ends, size = [], 0
    for piece in pieces:
        size += len(piece)
        ends.append(size)
    return b''.join(pieces), ends


def find_onset(tokens):
    raw, ends = token_bytes(tokens)
    text = raw.decode('utf-8', errors='strict')
    hits = spans(text)
    if not hits:
        return None
    ids = [t['id'] for t in tokens]
    candidates = []
    for hit in hits:
        start = len(text[:hit['start']].encode('utf-8'))
        index = bisect.bisect_right(ends, start)
        stop = bisect.bisect_left(ends, len(text[:hit['end']].encode('utf-8'))) + 1
        for period in range(1, min(256, (stop - index) // 4) + 1):
            a = index
            for i in range(index + period, stop + 1):
                if i == stop or ids[i] != ids[i - period]:
                    byte_start = ends[a - 1] if a else 0
                    if i - a >= 4 * period and ends[i - 1] - byte_start >= 4096:
                        candidates.append((a, period, i, hit))
                    a = i - period + 1
    if candidates:
        a, period, stop, hit = min(candidates, key=lambda c: c[:3])
        while a > 0 and ids[a - 1] == ids[a - 1 + period]:
            a -= 1
        while stop < len(ids) and ids[stop] == ids[stop - period]:
            stop += 1
        return dict(onset_token_M=a, period_tokens_M=period, period_ids=ids[a:a + period],
                    end_token_M=stop, detector=hit, exact_token_period=True)
    # Low diversity does not prove an exact model-token period.
    hit = min(hits, key=lambda h: h['start'])
    index = bisect.bisect_right(ends, len(text[:hit['start']].encode('utf-8')))
    return dict(onset_token_M=index, period_tokens_M=None, period_ids=[],
                detector=hit, exact_token_period=False)


def distribution(token):
    row = token.get('target')
    if row is None or row.get('invalid_distribution'):
        raise ValueError('missing target at token ' + str(token['index']))
    top = row['top']
    p = {e['id']: e['p'] for e in top}
    if len(p) != len(top) or any(not math.isfinite(v) or v <= 0 for v in p.values()):
        raise ValueError('invalid target probabilities')
    if row['omitted_mass'] > 1e-6 or abs(sum(p.values()) - 1) > 2e-6:
        raise ValueError('truncated or unnormalized target support')
    return p


def cut_size_is_known(header):
    e = header['effective']
    chain = e.get('samplers', [])
    if isinstance(chain, str):
        chain = chain.split(';')
    suffix = chain[chain.index('min_p') + 1:] if 'min_p' in chain else ['unknown']
    return (e.get('temperature', 1) > 0 and e.get('xtc_probability', 0) == 0
            and e.get('mirostat', 0) == 0 and e.get('adaptive_target', -1) < 0
            and not e.get('grammar') and not header['parsed'].get('grammar')
            and e.get('min_p', 0) > 0 and all(s in ('xtc', 'temperature') for s in suffix))


def metrics(header, token, repeat_id=None):
    p = distribution(token)
    sampled = p.get(token['id'], 0)
    if sampled <= 0:
        raise ValueError('emitted token is outside captured target support')
    known = cut_size_is_known(header)
    return dict(index_M=token['index'], id=token['id'], sampled_target_p_M=sampled,
                top20=token['target']['top'], entropy_nats_D=-sum(v * math.log(v) for v in p.values()),
                support_size_M=len(p), min_p_cut_size_D=len(p) if known else None,
                min_p_cut_size_reason='final support equals min-p support' if known else 'downstream mask/order not certified',
                sampled_rank_D=sorted(p, key=lambda k: (-p[k], k)).index(token['id']) + 1,
                repeat_id=repeat_id, repeat_target_p_M=p.get(repeat_id, 0) if repeat_id is not None else None,
                stage_survival_D={s: True for s in ('top-k', 'top-p', 'min-p')} if known else None,
                earlier_stage_ranks=None, earlier_stage_margins=None)


def analyze(path, out, match=None, window=512):
    header, prompt, tokens = replay.load(path)
    found = find_onset(tokens)
    reference = match or found
    rounds = {e['round']: e for e in map(json.loads, Path(path).read_text().splitlines()) if e['event'] == 'round'}
    rows = []
    for token in tokens:
        index = token['index']
        repeated = None
        if reference and reference.get('period_ids'):
            ids = reference['period_ids']
            repeated = ids[(index - reference['onset_token_M']) % len(ids)]
        row = metrics(header, token, repeated)
        proposal = rounds.get(token.get('round'), {})
        row.update(round=token.get('round'), draft_ids=proposal.get('draft', []), draft_q=proposal.get('draft_q', []))
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per-token.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    center = reference['onset_token_M'] if reference else None
    trajectory = [] if center is None else rows[max(0, center - window):min(len(rows), center + window + 1)]
    def averages(group):
        if not group:
            return None
        return dict(n_M=len(group), repeat_p_mean_D=sum(r['repeat_target_p_M'] for r in group) / len(group)
                    if group[0]['repeat_target_p_M'] is not None else None,
                    entropy_mean_nats_D=sum(r['entropy_nats_D'] for r in group) / len(group))
    before = averages([r for r in trajectory if r['index_M'] < center]) if center is not None else None
    after = averages([r for r in trajectory if r['index_M'] >= center]) if center is not None else None
    result = dict(capture=str(Path(path).resolve()), prompt_tokens_M=len(prompt), emitted_tokens_M=len(tokens),
                  onset=found, matching_reference=match, before=before, after=after,
                  entry=rows[center] if center is not None and center < len(rows) else None,
                  trajectory=trajectory, diagnostic='inspect entry p/rank and phase-aligned probability trajectory; no automatic causal verdict')
    atomic(out / 'analysis.json', result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--control', type=Path, action='append', default=[])
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    if not args.out.resolve().is_relative_to(root):
        ap.error('output must stay inside this worktree')
    result = analyze(args.capture, args.out)
    for i, path in enumerate(args.control):
        analyze(path, args.out / f'control-{i}', match=result['onset'])
    print(json.dumps({k: result[k] for k in ('onset', 'before', 'after')}))


if __name__ == '__main__':
    main()
