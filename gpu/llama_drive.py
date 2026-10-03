#!/usr/bin/env python3
"""attrib-llama client: warm up one booted llama-server (under an nsys launch session), then take the capture windows.

Windows (each is one request, one nsys start/stop pair, one <name>.nsys-rep):
  prefill-32k  cold 32768-token prompt (stream 1 of the frozen corpus), n_predict 1; window = whole request
  decode-8k    8192-token prompt (stream 0); un-profiled prefill request, then a profiled replay whose window opens
               DELAY s after the first token and lasts WINDOW s, then an un-profiled twin replay (same prompt and seed)
  decode-90k   90000-token prompt (stream 0, nested on the 8k prefix); same three requests
Contract: the capture window only ever holds the phase it names (prefill window closes before decode; decode windows open
after the first token and `nsys stop` is CALLED before the request ends: its return includes report generation and is not
compared), the profiled request runs to its end so the server reports draft counters, the twin gives the unprofiled
ms/round for the profiler-overhead line, and every window is recorded in runs.json before the next one starts.
Exit: 0 ok, 2 request/nsys failure, 5 prefill was not cold.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'inputs'))
import agg  # noqa: E402  (frozen item-30 client: prompt builder, sampling)

NSYS = 'nsys'
WINDOWS = {
    'prefill-32k': ('prefill', 32768, 1),
    'decode-8k': ('decode', 8192, 0),
    'decode-90k': ('decode', 90000, 0),
}


def nsys(*args, timeout=300):
    return subprocess.run([NSYS, *args], capture_output=True, text=True, timeout=timeout)


def nsys_start(session, out):
    r = nsys('start', f'--session={session}', '--sample=none', '--cpuctxsw=none', '-o', str(out), timeout=60)
    if r.returncode:
        raise RuntimeError(f'nsys start failed: {r.stdout[-300:]} {r.stderr[-300:]}')


def nsys_stop(session):
    r = nsys('stop', f'--session={session}', timeout=600)
    if r.returncode:
        raise RuntimeError(f'nsys stop failed: {r.stdout[-300:]} {r.stderr[-300:]}')


def make_prompt(url, tokens, stream, depth, overhead):
    chunk = agg.call(url, '/detokenize', {'tokens': tokens[stream * agg.OFFSET: stream * agg.OFFSET + depth - overhead]})['content']
    return agg.call(url, '/apply-template', {'messages': [{'role': 'user', 'content': chunk + '\n\n' + agg.QUESTION}]})['prompt']


def build(url, names):
    corpus = (HERE / 'inputs' / 'corpus.txt').read_text()
    tokens = agg.call(url, '/tokenize', {'content': corpus})['tokens']
    need = max(s * agg.OFFSET + d for _, d, s in (WINDOWS[n] for n in names))
    while len(tokens) < need:
        tokens += tokens
    empty = agg.call(url, '/apply-template', {'messages': [{'role': 'user', 'content': '\n\n' + agg.QUESTION}]})['prompt']
    overhead = len(agg.call(url, '/tokenize', {'content': empty})['tokens'])
    return {n: make_prompt(url, tokens, WINDOWS[n][2], WINDOWS[n][1], overhead) for n in names}


def summarize_timings(t, k):
    if not t:
        return {}
    out = {key: t.get(key) for key in ('prompt_n', 'cache_n', 'prompt_ms', 'prompt_per_second', 'predicted_n', 'predicted_ms',
                                       'predicted_per_second', 'draft_n', 'draft_n_accepted')}
    if t.get('draft_n') and k:
        rounds = t['draft_n'] / k
        out['rounds_est'] = rounds
        out['tok_per_round'] = t['predicted_n'] / rounds
        out['ms_per_round'] = t['predicted_ms'] / rounds if t.get('predicted_ms') else None
    return out


def stream_request(url, body, on_first_token):
    """Stream /completion; call on_first_token() once at the first content chunk; return (t_submit, t_first, final chunk)."""
    req = urllib.request.Request(url + '/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t_submit, t_first, final = time.monotonic(), None, {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data: '):
                continue
            c = json.loads(line[6:])
            if 'error' in c:
                raise RuntimeError(str(c['error']))
            if t_first is None and c.get('content'):
                t_first = time.monotonic()
                on_first_token()
            if c.get('stop'):
                final = c
                break
    return t_submit, t_first, final


def run_decode(url, session, out, name, prompt, k, delay, window, n_predict, seed, twin_n=1000):
    state = {'err': None, 'opened': None, 'closing': None, 'closed': None}
    pre = agg.prefill(url, prompt, 0, seed)  # un-profiled: leaves the prompt and its checkpoint cached for the replays

    def capture():
        try:
            time.sleep(delay)
            nsys_start(session, out / name)
            state['opened'] = time.monotonic()
            time.sleep(window)
            state['closing'] = time.monotonic()  # before the call: nsys stop returns after the report is written
            nsys_stop(session)
            state['closed'] = time.monotonic()
        except Exception as e:  # noqa: BLE001
            state['err'] = repr(e)

    th = []

    def first():
        t = threading.Thread(target=capture, daemon=True)
        t.start()
        th.append(t)

    body = {'prompt': prompt, 'n_predict': n_predict, 'ignore_eos': True, 'cache_prompt': True, 'id_slot': 0, 'stream': True,
            **agg.sampling(seed)}
    t_submit, t_first, final = stream_request(url, body, first)
    t_end = time.monotonic()
    for t in th:
        t.join(timeout=900)
    res = {'name': name, 'kind': 'decode', 'prefill_request': {k_: pre.get(k_) for k_ in ('prompt_n', 'cache_n', 'prompt_ms', 'wall_s')},
           'ttft_s': (t_first - t_submit) if t_first else None, 'capture_error': state['err'],
           'window_open_after_first_token_s': (state['opened'] - t_first) if state['opened'] and t_first else None,
           'stop_called_before_request_end_s': (t_end - state['closing']) if state['closing'] else None,
           'nsys_stop_s': (state['closed'] - state['closing']) if state['closed'] and state['closing'] else None,
           **summarize_timings(final.get('timings'), k)}
    if not final or not state['closing'] or state['closing'] > t_end - 2.0:
        res['capture_error'] = (res['capture_error'] or '') + ' window did not close at least 2 s before the request ended'
    # Twin: same prompt and seed, no profiler, prompt cached; its ms/round is the unprofiled reference.
    try:
        _, _, tfinal = stream_request(url, {**body, 'n_predict': twin_n}, lambda: None)
        res['twin'] = summarize_timings(tfinal.get('timings'), k)
        if (res['twin'].get('prompt_n') or 0) >= agg.MISS_TOKENS:
            res['twin']['cache_miss'] = True
    except Exception as e:  # noqa: BLE001
        res['twin'] = {'error': repr(e)}
    return res


def run_prefill(url, session, out, name, prompt, depth, k, seed):
    nsys_start(session, out / name)
    t0 = time.monotonic()
    err = None
    try:
        r = agg.call(url, '/completion', {'prompt': prompt, 'n_predict': 1, 'cache_prompt': True, 'id_slot': 0,
                                          **agg.sampling(seed)})
    finally:
        try:
            nsys_stop(session)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    res = {'name': name, 'kind': 'prefill', 'wall_s': time.monotonic() - t0, 'tokens': depth, 'capture_error': err,
           **summarize_timings(r.get('timings'), k)}
    res['cold'] = (res.get('prompt_n') or 0) >= 0.95 * depth
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--session', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows', default='prefill-32k,decode-8k,decode-90k')
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--delay', type=float, default=1.5)
    ap.add_argument('--window', type=float, default=4.0)
    ap.add_argument('--n-predict', type=int, default=4000)
    ap.add_argument('--twin-n', type=int, default=1000)
    ap.add_argument('--tokens', default='', help='CPU rehearsal only: name=prompt_tokens,... overrides')
    a = ap.parse_args()
    for kv in filter(None, a.tokens.split(',')):
        n, v = kv.split('=')
        WINDOWS[n] = (WINDOWS[n][0], int(v), WINDOWS[n][2])
    url = f'http://127.0.0.1:{a.port}'
    names = a.windows.split(',')
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {'k': a.k, 'windows': []}

    def save():
        (a.out / 'runs.json').write_text(json.dumps(runs, indent=1))

    prompts = build(url, names)
    runs['prompt_tokens'] = {n: len(agg.call(url, '/tokenize', {'content': p})['tokens']) for n, p in prompts.items()}
    save()
    # Warm-up: JIT/graph capture for the real shapes must not land in a window. Short, with drafts.
    warm = agg.call(url, '/completion', {'prompt': 'Write a Rust function that parses a CSV line.', 'n_predict': 64, 'ignore_eos': True, 'id_slot': 0, **agg.sampling(7)})
    runs['warmup'] = summarize_timings(warm.get('timings'), a.k)
    save()
    rc = 0
    for i, n in enumerate(names):
        kind, depth, _ = WINDOWS[n]
        seed = 1729 + i
        print(f'[{time.strftime("%T")}] window {n}', flush=True)
        try:
            if kind == 'decode':
                res = run_decode(url, a.session, a.out, n, prompts[n], a.k, a.delay, a.window, a.n_predict, seed, a.twin_n)
            else:
                res = run_prefill(url, a.session, a.out, n, prompts[n], depth, a.k, seed)
        except Exception as e:  # noqa: BLE001
            res = {'name': n, 'kind': kind, 'capture_error': repr(e)}
        runs['windows'].append(res)
        save()
        print(json.dumps(res), flush=True)
        if res.get('capture_error'):
            rc = 2
            break
        if kind == 'prefill' and not res.get('cold'):
            print('REJECTED: prefill window was not cold (prompt_n < 95% of the prompt)', flush=True)
            rc = 5
            break
    sys.exit(rc)


if __name__ == '__main__':
    main()
