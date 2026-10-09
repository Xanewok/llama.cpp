#!/usr/bin/env python3
"""np2 joint-replay smoke client: the drive_np2.py request sequence up to the joint cache-hit replay (the stack10 crash).

Per boot: warm both slots (64 tokens each, one at a time), 8k prefill of both slots, then --reps joint replays, slot 0 sent first.
Exit 0 always; replays.json holds the stage reached and per-rep errors and token counts, the shell judges them with the server state.
"""
import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'k11a'))
import drive_np2 as d  # noqa: E402

agg = d.agg
DEPTH = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reps', type=int, required=True)
    ap.add_argument('--n-predict', type=int, default=64)
    a = ap.parse_args()
    url = f'http://127.0.0.1:{a.port}'
    a.out.mkdir(parents=True, exist_ok=True)
    res = {'stage': 'start', 'n_predict': a.n_predict, 'reps': []}

    def save():
        (a.out / 'replays.json').write_text(json.dumps(res, indent=1))

    try:
        prompts, _ = agg.build_prompts(url, 2, [DEPTH])
        res['stage'] = 'warmup'
        save()
        for s in d.SLOTS:
            agg.call(url, '/completion', {'prompt': 'Write a Rust function that parses a CSV line.', 'n_predict': 64, 'ignore_eos': True,
                                          'id_slot': s, **agg.sampling(7 + s)})
        res['stage'] = 'prefill'
        save()
        res['prefill'] = [{k: p.get(k) for k in ('prompt_n', 'cache_n', 'prompt_ms')} for p in
                          (agg.prefill(url, prompts[(s, DEPTH)], s, 1729 + s) for s in d.SLOTS)]
        for rep in range(a.reps):
            res['stage'] = f'replay-{rep}'
            save()
            outs = [{} for _ in d.SLOTS]
            th = [threading.Thread(target=d.live_stream, args=(url, prompts[(s, DEPTH)], s, 1729 + s, a.n_predict, outs[s]), daemon=True)
                  for s in d.SLOTS]
            for t in th:
                t.start()
            for t in th:
                t.join(timeout=600)
            res['reps'].append({'rep': rep, 'error': [o.get('error') for o in outs],
                                'tokens': [o['stamps'][-1][1] if o.get('stamps') else 0 for o in outs],
                                'prompt_n': [o.get('timings', {}).get('prompt_n') for o in outs],
                                'cache_n': [o.get('timings', {}).get('cache_n') for o in outs]})
            save()
            if any(o.get('error') for o in outs):
                break
        res['stage'] = 'done'
    except Exception as e:  # noqa: BLE001
        res['client_exception'] = repr(e)
    save()


if __name__ == '__main__':
    main()
