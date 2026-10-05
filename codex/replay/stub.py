#!/usr/bin/env python3
"""CPU-only stand-in for the HTTP endpoints used by sweep.py."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--arm', choices=('base', 'subset'), required=True)
    args = ap.parse_args()
    corpus = json.loads((DATA / 'prefixes.json').read_text())
    bounds = json.loads((DATA / 'prefix_bounds.json').read_text())
    lookup = {json.dumps(p['messages'], sort_keys=True): (i, p) for i, p in enumerate(corpus['prefixes'])}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(dict(status='ok'))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/apply-template':
                i, prefix = lookup[json.dumps(body['messages'], sort_keys=True)]
                self.reply(dict(prompt=f'{i}:{bounds["prefixes"][prefix["id"]]["rendered_prompt_tokens_M"]}'))
            elif self.path == '/tokenize':
                i, length = map(int, body['content'].split(':'))
                self.reply(dict(tokens=[i + 1] * length))
            elif self.path == '/completion':
                length = body['n_predict']
                n = len(body['prompt'])
                rate = {5: 100, 6: 110, 7: 105}[args.k] * (1.1 if args.arm == 'subset' else 1.0)
                if os.environ.get('STUB_SHORT') == '1' and length > 8:
                    length = 15
                draft = args.k * max(1, length // (args.k + 1))
                t = dict(predicted_n=length, predicted_ms=(length - 1) / rate * 1000,
                         predicted_per_second=rate, draft_n=draft, draft_n_accepted=int(draft * 0.75),
                         cache_n=0, prompt_n=n, prompt_ms=10)
                settings = dict(seed=body['seed'], temperature=1.0, top_k=20, top_p=0.95, min_p=0.05,
                                ignore_eos=False, n_predict=body['n_predict'], backend_sampling=True,
                                **{'speculative.types': 'none,draft-mtp'})
                if os.environ.get('STUB_DRIFT') == '1' and length > 8:
                    settings['top_p'] = 0.9
                chunks = [dict(content='', stop=False, timings=None),
                          dict(content=f'sample k={args.k} seed={body["seed"]}', stop=False, timings=t),
                          dict(content='', stop=True, truncated=False, generation_settings=settings, timings=t)]
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for row in chunks:
                    self.wfile.write(('data: ' + json.dumps(row) + '\n\n').encode())
                self.wfile.flush()
            else:
                self.send_error(404)

    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
