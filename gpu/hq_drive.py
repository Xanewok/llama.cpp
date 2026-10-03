#!/usr/bin/env python3
"""attrib-hq client: profiler windows on one booted vLLM (HQ) server that runs under `nsys profile
--capture-range=cudaProfilerApi --capture-range-end=repeat-shutdown:N:sync`. Window i is nsys range i.

  prefill-32k  prefix cache reset, 32768-token prompt (stream 1 of the corpus), max_tokens 1; /start_profile before the
               request, /stop_profile after it returns
  decode-*     prefix cache reset, an un-profiled twin request (cold prefill + TWIN_N tokens, gives the unprofiled ms/step),
               then the profiled request on the same prompt and seed (prompt cached): /start_profile DELAY s after its
               first token, /stop_profile WINDOW s later, request dropped
Contract: ranges are closed by this client (the server's max_iterations is only a safety cap); every read has a timeout,
because the last range shuts the server down mid-stream and mid-reply (step0's profile request hung 600 s on exactly
that), so the last /stop_profile is capped at LAST_STOP_S and its connection error is accepted only for the last window;
the twin runs BEFORE the window because nothing can run after the last range. runs.json is rewritten after every window.
Twin ms/step is the median inter-chunk time of the twin's steady part (one SSE chunk per engine step: inferred, not
checked against the server's step counter). Exit: 0 ok, 2 profiler/request failure.
"""
import argparse
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFSET = 100000  # tokens between stream starts; prefill and decode prompts share no prefix
WINDOWS = {'prefill-32k': ('prefill', 32768, 1), 'decode-8k': ('decode', 8192, 0), 'decode-8k-b': ('decode', 8192, 0),
           'decode-90k': ('decode', 90000, 0)}
SAMPLING = {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95, 'seed': 1729}  # min_p 0.05 comes from the server's generation config
TWIN_N = 400
LAST_STOP_S = 90
STOP_S = 600


def post(url, body=None, timeout=60):
    req = urllib.request.Request(url, b'' if body is None else json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def start_profile(base):
    t = time.monotonic()
    post(base + '/start_profile', timeout=120)
    return time.monotonic() - t


def stop_profile(base, timeout):
    # sync mode: the call returns after nsys has written the range's report
    t = time.monotonic()
    try:
        post(base + '/stop_profile', timeout=timeout)
    except OSError:  # the last range shuts the server down before the reply; main() rejects it for any other window
        return None
    return time.monotonic() - t


def prompt_tokens(base, model, text, stream, depth):
    tokens = json.loads(post(base + '/tokenize', {'model': model, 'prompt': text, 'add_special_tokens': False}))['tokens']
    while len(tokens) < stream * OFFSET + depth:
        tokens += tokens
    return tokens[stream * OFFSET: stream * OFFSET + depth]


def stream_chunks(base, model, toks, max_tokens, on_chunk):
    """POST a streaming completion; on_chunk(index, t) -> True stops reading. Returns (ttft_s, chunk times)."""
    body = {'model': model, 'prompt': toks, 'max_tokens': max_tokens, 'ignore_eos': True, 'stream': True, **SAMPLING}
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)  # covers a 90k cold prefill
    conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
    t_submit = time.monotonic()
    resp = conn.getresponse()
    times, done = [], threading.Event()
    last = [time.monotonic()]

    def watchdog():  # once streaming, a 30 s gap means the server is gone; closing unblocks the read
        while not done.wait(2):
            if times and time.monotonic() - last[0] > 30:
                conn.close()
                return

    try:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status}: {resp.read(300)!r}')
        threading.Thread(target=watchdog, daemon=True).start()
        for raw in resp:
            line = raw.decode(errors='replace').strip()
            if line == 'data: [DONE]':
                break
            if not line.startswith('data: '):
                continue
            c = json.loads(line[6:])
            if c.get('choices') and c['choices'][0].get('text'):
                last[0] = time.monotonic()
                times.append(last[0])
                if on_chunk(len(times) - 1, last[0]):
                    break
    finally:
        done.set()
        conn.close()
    return (times[0] - t_submit) if times else None, times


def run_prefill(base, model, toks, name, last):
    post(base + '/reset_prefix_cache')
    res = {'name': name, 'kind': 'prefill', 'tokens': len(toks)}
    res['start_s'] = start_profile(base)
    t0 = time.monotonic()
    try:
        r = json.loads(post(base + '/v1/completions', {'model': model, 'prompt': toks, 'max_tokens': 1, **SAMPLING}, timeout=900))
        res['usage'] = r.get('usage')
    finally:
        res['request_s'] = time.monotonic() - t0
        res['stop_s'] = stop_profile(base, LAST_STOP_S if last else STOP_S)
    return res


def run_decode(base, model, toks, name, delay, window, last):
    post(base + '/reset_prefix_cache')
    res = {'name': name, 'kind': 'decode', 'tokens': len(toks)}
    ttft, times = stream_chunks(base, model, toks, TWIN_N, lambda i, t: False)
    steady = [b - a for a, b in zip(times[10:], times[11:])]
    res['twin'] = {'ttft_s': ttft, 'chunks': len(times), 'ms_per_step': 1000 * statistics.median(steady) if len(steady) >= 50 else None}
    state = {}
    t_first = []

    def on_chunk(i, now):
        if not t_first:
            t_first.append(now)
        if now - t_first[0] >= delay and 'start_s' not in state:
            state['start_s'] = start_profile(base)
            state['opened'] = time.monotonic()
        if 'opened' in state and time.monotonic() - state['opened'] >= window:
            state['stop_s'] = stop_profile(base, LAST_STOP_S if last else STOP_S)
            return True
        return False

    ttft, times = stream_chunks(base, model, toks, 1500, on_chunk)
    res.update(ttft_s=ttft, chunks=len(times), start_s=state.get('start_s'), stop_s=state.get('stop_s'))
    if 'stop_s' not in state:
        res['error'] = 'request ended before the window closed'
        if 'start_s' in state:
            stop_profile(base, 60)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows', default='prefill-32k,decode-8k,decode-90k')
    ap.add_argument('--delay', type=float, default=1.5)
    ap.add_argument('--window', type=float, default=3.5)
    ap.add_argument('--corpus', type=Path, default=HERE / 'inputs' / 'hq-corpus.txt')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = json.loads(urllib.request.urlopen(a.url + '/v1/models', timeout=30).read())['data'][0]['id']
    text = a.corpus.read_text()
    runs = {'model': model, 'windows': []}

    def save():
        (a.out / 'runs.json').write_text(json.dumps(runs, indent=1))

    rc = 0
    names = a.windows.split(',')
    for name in names:
        kind, depth, stream = WINDOWS[name]
        last = name == names[-1]
        print(f'[{time.strftime("%T")}] window {name}', flush=True)
        try:
            toks = prompt_tokens(a.url, model, text, stream, depth)
            res = (run_prefill(a.url, model, toks, name, last) if kind == 'prefill'
                   else run_decode(a.url, model, toks, name, a.delay, a.window, last))
        except Exception as e:  # noqa: BLE001
            res = {'name': name, 'kind': kind, 'error': repr(e)}
        if 'stop_s' in res and res['stop_s'] is None and not last:
            res['error'] = 'server closed the connection on /stop_profile before the last window'
        runs['windows'].append(res)
        save()
        print(json.dumps(res), flush=True)
        if res.get('error'):
            rc = 2
            break
    sys.exit(rc)


if __name__ == '__main__':
    main()
