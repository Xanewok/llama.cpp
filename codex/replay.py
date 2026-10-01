#!/usr/bin/env python3
"""Use an env-gated capture server; preserve prompt/generated history roles."""
import argparse
import hashlib
import json
import urllib.request
import urllib.error
from pathlib import Path


def text(tokens):
    return b''.join(bytes.fromhex(t['piece_hex']) for t in tokens).decode('utf-8', errors='replace')


def load(path):
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not events or events[0].get('version') != 1:
        raise ValueError('unsupported capture')
    header = events[0]
    tokens = [e for e in events if e['event'] == 'token']
    if [e['index'] for e in tokens] != list(range(len(tokens))):
        raise ValueError('missing or duplicate generated token')
    if not events[-1]['event'] == 'end' or events[-1]['emitted'] != len(tokens):
        raise ValueError('incomplete response capture')
    if any(e['event'] in ('context_shift', 'checkpoint_restore', 'control') for e in events):
        raise ValueError('capture has state transitions unsupported by replay')
    prompt = [e['ids'] for e in events if e['event'] == 'prompt']
    if len(prompt) != 1:
        raise ValueError('missing evaluated prompt')
    effective = header['effective']
    if effective.get('mirostat', 0) or effective.get('adaptive_target', -1) >= 0 or effective.get('xtc_probability', 0):
        raise ValueError('sample-dependent sampler cannot be rebuilt by history')
    if header.get('replay_force'):
        raise ValueError('select an original capture, not a replay')
    return header, prompt[0], tokens


def body(path, position, length, branch_seed=None, tolerance=1e-6, verify=True, allow_drift=False):
    header, prompt, tokens = load(path)
    if not 0 < position < len(tokens):
        raise ValueError('position must be inside generated history and positive')
    if length <= 0:
        raise ValueError('length must be positive')
    # Start from the parsed request to retain fields omitted by effective serialization.
    request = dict(header['parsed'])
    request.update({key: value for key, value in header['effective'].items()
                    if key not in ('chat_format', 'preserved_tokens', 'logit_bias', 'grammar_triggers')})
    request['logit_bias'] = [[e['token'], False if e['bias'] is None else e['bias']] for e in header['effective']['logit_bias']]
    request['__capture_preserved'] = header['effective']['preserved_tokens']
    # These serialized fields are namespaced/debug values, not transport controls.
    for key in list(request):
        if key.startswith('speculative.'):
            request.pop(key)
    request['prompt'] = prompt
    request.update(seed=header['target_seed'], stream=False, return_tokens=True,
                   cache_prompt=False, n_cmpl=1, n_predict=position + length,
                   max_tokens=position + length, max_completion_tokens=position + length)
    request['__capture_force'] = [t['id'] for t in tokens[:position]]
    request['__capture_budget'] = header['budget']
    request['__capture_tolerance'] = tolerance
    request['__capture_allow_drift'] = allow_drift
    if verify:
        expected = {}
        for token in tokens[:position + 1]:
            if 'target' not in token:
                raise ValueError('capture lacks target distribution at ' + str(token['index']))
            if token['target']['omitted_mass'] > tolerance:
                raise ValueError('top-20 does not cover full target support')
            expected[str(token['index'])] = token['target']
        request['__capture_expected'] = expected
    if branch_seed is not None:
        if not 0 <= branch_seed < 2**32 - 1:
            raise ValueError('branch seed must be explicit and finite')
        request['__capture_branch_seed'] = branch_seed
    return request


def post(url, path, data, timeout=1800):
    if not url.startswith(('http://127.0.0.1:', 'http://localhost:')):
        raise ValueError('only local server URLs are accepted')
    req = urllib.request.Request(url + path, json.dumps(data).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 503:
            raise
        try:
            value = json.load(error)
        except ValueError:
            raise error
    if 'error' in value:
        raise RuntimeError(value['error'])
    return value


def request_capture(url, capture_dir, request, timeout=1800):
    before = set(Path(capture_dir).glob('request-*.jsonl'))
    result = post(url, '/completion', request, timeout)
    created = set(Path(capture_dir).glob('request-*.jsonl')) - before
    if len(created) != 1:
        raise RuntimeError('replay requires an exclusive single-slot server')
    path = created.pop()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if not events or events[-1]['event'] != 'end':
        raise RuntimeError('replay was interrupted')
    wanted = request.get('__capture_expected', {})
    validations = [e for e in events if e['event'] == 'validation']
    if {str(e['index']) for e in validations} != set(wanted) or any(not e['equal'] and not request.get('__capture_allow_drift') for e in validations):
        raise RuntimeError('missing or failed distribution validation')
    force = request['__capture_force']
    emitted = [e for e in events if e['event'] == 'token']
    if [e['id'] for e in emitted[:len(force)]] != force:
        raise RuntimeError('teacher forcing did not retain exact token IDs')
    return result, path, emitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--position', type=int, required=True)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--url', default='http://127.0.0.1:18127')
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--allow-numeric-drift', action='store_true', help='Record row differences for a deliberately changed arm; requires an independent strict fidelity check.')
    args = parser.parse_args()
    if args.n < 1:
        parser.error('--n must be positive')
    requests = [body(args.capture, args.position, args.length, args.seed + i, args.tolerance, allow_drift=args.allow_numeric_drift) for i in range(args.n)]
    if args.plan:
        print(json.dumps(dict(continuations_D=args.n, length_D=args.length,
                              force_tokens_D=args.position, verify_rows_D=args.position + 1)))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    validation = body(args.capture, args.position, 1, tolerance=args.tolerance, allow_drift=args.allow_numeric_drift)
    _, validation_path, _ = request_capture(args.url, args.capture_dir, validation)
    (args.out / 'validation.json').write_text(json.dumps({'capture': str(validation_path)}))
    for i, request in enumerate(requests):
        result, path, tokens = request_capture(args.url, args.capture_dir, request)
        suffix = tokens[args.position:]
        (args.out / f'branch-{i}.json').write_text(json.dumps(dict(
            seed=request['__capture_branch_seed'], capture=str(path), token_ids=[t['id'] for t in suffix],
            text=text(suffix), final=result), indent=2))
    (args.out / 'SOURCE.json').write_text(json.dumps(dict(capture=str(args.capture.resolve()),
        sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(), position=args.position,
        n=args.n, length=args.length, tolerance=args.tolerance), indent=2))


if __name__ == '__main__':
    main()
