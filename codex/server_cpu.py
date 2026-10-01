#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.request

root = pathlib.Path(__file__).resolve().parent.parent
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
model = os.environ.get('MODEL', '/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf')
binary = root / 'build-cpu/bin/s4-cpu-srv'
subset = root / 'codex/cpu-subset.json'
subset.write_text(json.dumps(list(range(248319, -1, -10))))
port = 18174

def request(path, data=None):
    data = None if data is None else json.dumps(data).encode()
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/{path}', data, {'Content-Type': 'application/json'}), timeout=240) as response:
        return json.load(response)

results = []
reference = None
for k, sub in [(0, False)] + [(k, sub) for k in (3, 4, 5) for sub in (False, True)]:
    tag = f'k{k}-subset{int(sub)}'
    command = ['nice', '-n', '19', str(binary), '-m', model, '-ngl', '0', '-t', '8', '-tb', '8',
               '-c', '2048', '-np', '1', '-b', '128', '-ub', '128', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
               '--host', '127.0.0.1', '--port', str(port), '--metrics', '--backend-sampling']
    if k:
        command += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(k), '--spec-draft-n-min', '0', '--spec-draft-p-min', '0',
                    '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling', '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16']
    if sub:
        command += ['--spec-draft-vocab', str(subset)]
    with open(root / f'codex/server-{tag}.log', 'w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=root)
        try:
            deadline = time.monotonic() + 120
            while True:
                assert process.poll() is None, f'{tag}: server exited'
                try:
                    if request('health').get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                assert time.monotonic() < deadline, f'{tag}: health timeout'
                time.sleep(0.2)
            data = request('completion', {'prompt': 'Solve SEND + MORE = MONEY. Explain the first constraint briefly.\n',
                                         'temperature': 0, 'seed': 1729, 'n_predict': 48, 'cache_prompt': False, 'return_tokens': True})
            digest = hashlib.sha256(data['content'].encode()).hexdigest()
            results.append({'tag': tag, 'sha256': digest, 'response': data})
            (root / 'codex/server-results.json').write_text(json.dumps(results, indent=2))
            if reference is None:
                reference = data['content']
            assert data['content'] == reference, f'{tag}: greedy output differs from non-spec'
            if k:
                assert data['timings'].get('draft_n', 0) > 0, f'{tag}: no speculative tokens'
            print(f'M: {tag}: greedy identity, {data["timings"]["predicted_n"]} tokens, sha256={digest}', flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
