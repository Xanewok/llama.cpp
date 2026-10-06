#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'logs/reserve-server'
MODEL = '/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf'
PORT = 18372


def request(path, data=None, timeout=10):
    encoded = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}' + path, encoded,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert 'GGML_SCHED_NO_REALLOC:BOOL=ON' in (ROOT / 'build-cpu/CMakeCache.txt').read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / 'build-cpu/bin/gdn-reserve-server'), '-m', MODEL, '-c', '4096', '-np', '2',
               '-ngl', '0', '-t', '8', '-tb', '8', '-b', '64', '-ub', '16', '-fa', 'on',
               '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '--spec-draft-n-min', '0',
               '--spec-draft-p-min', '0', '--spec-draft-sampling', 'probabilistic',
               '--kv-unified', '--host', '127.0.0.1', '--port', str(PORT), '--log-verbosity', '4']
    (OUT / 'command.json').write_text(json.dumps(command))
    started = time.monotonic()
    with (OUT / 'server.log').open('wb') as log:
        env = dict(os.environ, LD_PRELOAD=str(ROOT / 'build-cpu/bin/reserve-probe.so'))
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(240):
                assert process.poll() is None, 'server exited before ready'
                try:
                    request('/health')
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise TimeoutError('server ready budget')
            data = {'prompt': 'List the integers from 1 to 50, one per line.\n1\n2\n3\n',
                    'n_cmpl': 2, 'n_predict': 32, 'temperature': 0.8, 'seed': 1729, 'cache_prompt': True}
            (OUT / 'request.json').write_text(json.dumps(data))
            try:
                result = request('/completion', data, timeout=900)
            except urllib.error.HTTPError as error:
                (OUT / 'error.txt').write_bytes(error.read())
                raise
            (OUT / 'response.json').write_text(json.dumps(result, indent=2))
            assert process.poll() is None, 'server aborted'
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    text = (OUT / 'server.log').read_text(errors='replace')
    assert re.search(r'n_rs_seq\s*=\s*[1-9]', text), 'rollback not enabled'
    assert 'copying state to child' in text or 'copying shared prompt' in text, 'seq_cp not engaged'
    assert not re.search(r'GGML_ASSERT|Aborted|failed to allocate', text), 'allocation failed'
    print(json.dumps({'NO_REALLOC': 'PASS', 'seq_cp': 'PASS', 'n_parallel': 2,
                      'extra_row_probe_lines_M': text.count('GDN_GETTER_EXTRA'),
                      'wall_s_M': time.monotonic() - started, 'exit': process.returncode}))


if __name__ == '__main__':
    main()
