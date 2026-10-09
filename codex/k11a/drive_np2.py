#!/usr/bin/env python3
"""attrib-llama np2 client: two slots decode concurrently while one nsys window is open (attrib-pack llama_drive.py is np1 only).

run:      per window <name> (decode-8k | decode-32k | decode-90k), slot s reads corpus tokens from s * agg.OFFSET (no shared
          prefix); both slots prefill un-profiled, then a profiled replay of both slots at once, then an un-profiled twin pair.
manifest: runs.json + exported <window>.sqlite -> attrib_pack.py manifest (same schema as gpu/make_manifest.py).
Contract: the window opens DELAY s after BOTH slots produced a token and nsys stop is CALLED >= 2 s before the first slot ends,
so the window only holds two-slot decode; a request ends only at n_predict. tok_per_round is the SUM over slots (one batched
round serves both), twin ms/round is measured inside the twin's overlap window only (a slot that ends early leaves the other
decoding alone, which is not the np2 round).
Exit: 0 ok, 2 request/nsys failure.
"""
import argparse
import json
import shlex
import sys
import threading
import time
import urllib.request
from pathlib import Path

GPU = Path('/home/xanewok/repos/attrib-pack/gpu')
sys.path.insert(0, str(GPU))
sys.path.insert(0, str(GPU / 'inputs'))
import agg  # noqa: E402  (frozen item-30 client: prompts, stream, overlap metrics)
import llama_drive as ld  # noqa: E402  (nsys start/stop, timing summary)

DEPTHS = {'decode-8k': 8192, 'decode-32k': 32768, 'decode-90k': 90000}
SLOTS = (0, 1)


def live_stream(url, prompt, slot, seed, n_predict, out):
    """agg.stream, but stamps land in out['stamps'] as they arrive (the capture thread waits on the first token of both slots)."""
    body = {'prompt': prompt, 'n_predict': n_predict, 'ignore_eos': True, 'cache_prompt': True, 'id_slot': slot, 'stream': True,
            **agg.sampling(seed)}
    req = urllib.request.Request(url + '/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    out.update(slot=slot, seed=seed, t_submit=time.monotonic(), stamps=[], timings={})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode(errors='replace').strip()
                if not line.startswith('data: '):
                    continue
                c = json.loads(line[6:])
                if 'error' in c:
                    raise RuntimeError(str(c['error']))
                if c.get('tokens_predicted'):
                    out['stamps'].append((time.monotonic(), c['tokens_predicted']))
                if c.get('stop'):
                    out['timings'] = c.get('timings', {})
                    break
    except Exception as e:  # noqa: BLE001
        out['error'] = repr(e)


def pair(url, prompts, depth, seed, n_predict):
    outs = [{} for _ in SLOTS]
    th = [threading.Thread(target=live_stream, args=(url, prompts[(s, depth)], s, seed + s, n_predict, outs[s]), daemon=True) for s in SLOTS]
    for t in th:
        t.start()
    return outs, th


def overlap_ms_per_round(outs, per_stream):
    """Per slot: overlap wall ms / (tokens inside the overlap / that slot's tokens per round)."""
    w0 = max(o['stamps'][0][0] for o in outs)
    w1 = min(o['stamps'][-1][0] for o in outs)
    vals = []
    for o, p in zip(outs, per_stream):
        if w1 <= w0 or not p.get('tok_per_round'):
            return None
        toks = agg.n_at(o['stamps'], w1) - agg.n_at(o['stamps'], w0)
        vals.append((w1 - w0) * 1000 / (toks / p['tok_per_round']))
    return {'per_slot': vals, 'mean': sum(vals) / len(vals), 'overlap_s': w1 - w0}


def run_window(url, session, out, name, depth, prompts, k, delay, window, n_predict, seed, twin_n, sample_args):
    pre = [agg.prefill(url, prompts[(s, depth)], s, seed + s) for s in SLOTS]
    state = {'err': None, 'opened': None, 'closing': None, 'closed': None}
    outs, th = pair(url, prompts, depth, seed, n_predict)

    def capture():
        try:
            end = time.monotonic() + 300
            while not all(first_stamp(o) for o in outs) and not any(o.get('error') for o in outs) and time.monotonic() < end:
                time.sleep(0.05)
            if not all(first_stamp(o) for o in outs):
                raise RuntimeError('a slot never produced a token')
            time.sleep(delay)
            ld.nsys_start(session, out / name, sample_args)
            state['opened'] = time.monotonic()
            time.sleep(window)
            state['closing'] = time.monotonic()
            ld.nsys_stop(session)
            state['closed'] = time.monotonic()
        except Exception as e:  # noqa: BLE001
            state['err'] = repr(e)

    cap = threading.Thread(target=capture, daemon=True)
    cap.start()
    for t in th:
        t.join(timeout=3600)
    cap.join(timeout=900)
    per = [ld.summarize_timings(o.get('timings'), k) for o in outs]
    ends = [o['stamps'][-1][0] if o.get('stamps') else None for o in outs]
    res = {'name': name, 'kind': 'decode', 'slots': len(SLOTS), 'depth': depth,
           'prefill_requests': [{k_: p.get(k_) for k_ in ('prompt_n', 'cache_n', 'prompt_ms', 'wall_s')} for p in pre],
           # the streams' own errors first: a dead server also makes the capture thread fail generically
           'capture_error': '; '.join([f'slot {s}: {o["error"]}' for s, o in zip(SLOTS, outs) if o.get('error')]
                                      + ([f'capture: {state["err"]}'] if state['err'] else [])) or None,
           'window_open_after_both_first_tokens_s': (state['opened'] - max(o['stamps'][0][0] for o in outs)) if state['opened'] and all(first_stamp(o) for o in outs) else None,
           'stop_called_before_first_slot_end_s': (min(e for e in ends if e) - state['closing']) if state['closing'] and all(ends) else None,
           'nsys_stop_s': (state['closed'] - state['closing']) if state['closed'] and state['closing'] else None,
           'per_slot': per, 'replay_prompt_n': [o.get('timings', {}).get('prompt_n') for o in outs]}
    if all(p.get('tok_per_round') for p in per):
        res['tok_per_round_sum'] = sum(p['tok_per_round'] for p in per)
    if not all(ends) or not state['closing'] or state['closing'] > min(e for e in ends if e) - 2.0:
        res['capture_error'] = (res['capture_error'] or '') + ' window did not close at least 2 s before the first slot ended'
    try:
        touts, tth = pair(url, prompts, depth, seed, twin_n)
        for t in tth:
            t.join(timeout=3600)
        tper = [ld.summarize_timings(o.get('timings'), k) for o in touts]
        res['twin'] = {'per_slot': tper, 'replay_prompt_n': [o.get('timings', {}).get('prompt_n') for o in touts],
                       'cache_miss': any((o.get('timings', {}).get('prompt_n') or 0) >= agg.MISS_TOKENS for o in touts),
                       'overlap': overlap_ms_per_round(touts, tper) if all(first_stamp(o) for o in touts) else None}
    except Exception as e:  # noqa: BLE001
        res['twin'] = {'error': repr(e)}
    return res


def first_stamp(o):
    return bool(o.get('stamps'))


def cmd_run(a):
    url = f'http://127.0.0.1:{a.port}'
    names = a.windows.split(',')
    depths = dict(DEPTHS)
    for kv in filter(None, a.depths.split(',')):
        n, v = kv.split('=')
        depths[n] = int(v)
    a.out.mkdir(parents=True, exist_ok=True)
    sample_args = shlex.split(a.nsys_sample_args) or None
    runs = {'k': a.k, 'slots': len(SLOTS), 'windows': [], 'depths': {n: depths[n] for n in names}}
    if sample_args:
        runs['nsys_sample_args'] = sample_args

    def save():
        (a.out / 'runs.json').write_text(json.dumps(runs, indent=1))

    prompts, meta = agg.build_prompts(url, len(SLOTS), [depths[n] for n in names])
    runs['prompt_meta'] = meta
    save()
    for s in SLOTS:
        warm = agg.call(url, '/completion', {'prompt': 'Write a Rust function that parses a CSV line.', 'n_predict': 64, 'ignore_eos': True,
                                             'id_slot': s, **agg.sampling(7 + s)})
        runs.setdefault('warmup', []).append(ld.summarize_timings(warm.get('timings'), a.k))
    save()
    rc = 0
    for i, n in enumerate(names):
        print(f'[{time.strftime("%T")}] window {n} (2 slots)', flush=True)
        try:
            res = run_window(url, a.session, a.out, n, depths[n], prompts, a.k, a.delay, a.window, a.n_predict, 1729 + 10 * i, a.twin_n, sample_args)
        except Exception as e:  # noqa: BLE001
            res = {'name': n, 'kind': 'decode', 'capture_error': repr(e)}
        runs['windows'].append(res)
        save()
        print(json.dumps(res), flush=True)
        if res.get('capture_error'):
            rc = 2
            break
    sys.exit(rc)


def cmd_manifest(a):
    runs = json.loads((a.out_dir / 'runs.json').read_text())
    traces = []
    for w in runs['windows']:
        n = w['name']
        spec = {'id': f'llama-{n}-np2', 'sqlite': str(a.out_dir / f'{n}.sqlite'), 'kind': 'decode', 'depth': w['depth'], 'k': a.k, 'group': 'depth'}
        if w.get('tok_per_round_sum'):
            spec['tok_per_round'] = w['tok_per_round_sum']
        ov = (w.get('twin') or {}).get('overlap')
        if ov and not w['twin'].get('cache_miss'):
            spec['unprofiled'] = {'ms': ov['mean'], 'source': 'np2 twin pair: same prompts and seeds, no profiler; overlap-window wall ms / '
                                  '(overlap tokens / slot tokens-per-round), mean of 2 slots (inferred rounds)'}
        traces.append(spec)
    a.manifest.write_text(json.dumps({'title': a.title, 'model': a.model, 'engine': 'llama', 'k': a.k, 'min_rounds': a.min_rounds, 'traces': traces}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--port', type=int, required=True)
    r.add_argument('--session', required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--windows', default='decode-8k,decode-32k,decode-90k')
    r.add_argument('--k', type=int, required=True)
    r.add_argument('--delay', type=float, default=1.5)
    r.add_argument('--window', type=float, default=5.0)
    r.add_argument('--n-predict', type=int, default=2000)
    r.add_argument('--twin-n', type=int, default=800)
    r.add_argument('--depths', default='', help='CPU rehearsal only: name=prompt_tokens,... overrides')
    r.add_argument('--nsys-sample-args', default='')
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser('manifest')
    m.add_argument('--out-dir', type=Path, required=True)
    m.add_argument('--model', required=True)
    m.add_argument('--k', type=int, required=True)
    m.add_argument('--title', required=True)
    m.add_argument('--manifest', type=Path, required=True)
    m.add_argument('--min-rounds', type=int, default=40)
    m.set_defaults(fn=cmd_manifest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
