#!/usr/bin/env python3
"""CPU identity and strict-replay smoke using the read-only capture CPU build."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error

import catch
import catch_replay as replay
import onset

OUT = catch.ROOT / os.environ.get('CPU_CATCH_OUT', 'build-catch-cpu')
SOURCE = catch.REFERENCE / 'build-cpu/bin'
MODEL = '/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf'
URL = 'http://127.0.0.1:18129'


def boot(label, enabled, spec):
    directory = OUT / label
    directory.mkdir()
    captures = directory / 'captures'
    captures.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               LD_LIBRARY_PATH=str(OUT / 'bin'), GGML_BACKEND_DL_PATH=str(OUT / 'bin'))
    for key in ('LLAMA_CAPTURE_DIR', 'LLAMA_CAPTURE_PROBS', 'LLAMA_CAPTURE_LIMIT'):
        env.pop(key, None)
    if enabled:
        env.update(LLAMA_CAPTURE_DIR=str(captures), LLAMA_CAPTURE_PROBS='1', LLAMA_CAPTURE_LIMIT='32')
    cmd = ['nice', '-n', '19', str(OUT / 'bin/catch-cpu-srv'), '-m', MODEL, '-ngl', '0',
           '-np', '1', '-c', '4096', '-b', '512', '-ub', '128', '-t', '4', '-tb', '4', '-fa', 'on',
           '--host', '127.0.0.1', '--port', '18129', '--fit', 'off', '--backend-sampling', '--no-context-shift']
    if spec:
        cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '--spec-draft-n-min', '0',
                '--spec-draft-p-min', '0', '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling']
    else:
        cmd += ['--spec-type', 'none']
    log = (directory / 'server.log').open('w')
    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(480):
            if child.poll() is not None:
                raise RuntimeError('CPU server died: ' + str(directory))
            try:
                replay.post(URL, '/tokenize', dict(content='x'), timeout=2)
                return child, log, captures
            except (OSError, urllib.error.HTTPError):
                time.sleep(.25)
        raise RuntimeError('CPU boot timed out')
    except BaseException:
        catch.study.stop(child)
        log.close()
        raise


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU smoke requires CUDA_VISIBLE_DEVICES empty')
    if not OUT.resolve().is_relative_to(catch.ROOT) or OUT.resolve() == catch.ROOT:
        raise RuntimeError('CPU output must stay inside this worktree')
    OUT.mkdir(exist_ok=False)
    binary = OUT / 'bin'
    binary.mkdir()
    shutil.copy2(SOURCE / 'llama-server', binary / 'catch-cpu-srv')
    for path in SOURCE.glob('*.so*'):
        if 'cuda' in path.name:
            raise RuntimeError('CPU build contains a CUDA backend')
        shutil.copy2(path, binary / path.name, follow_symlinks=True)
    timings, identities, validations = [], {}, []
    start = time.monotonic()
    body = dict(prompt='Write a Python list of integers from zero through thirty. Output only the list.',
                seed=1234, temperature=1, top_k=20, top_p=.95, min_p=.05, ignore_eos=True,
                cache_prompt=False, stream=False, return_tokens=True, n_predict=16,
                repeat_penalty=1, presence_penalty=0, frequency_penalty=0, dry_multiplier=0)
    for spec in (False, True):
        for enabled in (False, True):
            label = f'{"mtp3" if spec else "no-spec"}-{"on" if enabled else "off"}'
            child, log, directory = boot(label, enabled, spec)
            try:
                result = replay.post(URL, '/completion', body, timeout=600)
                identities[label] = result['tokens']
                timings.append(dict(label=label, timings_M=result['timings']))
                if enabled:
                    path, = directory.glob('request-*.jsonl')
                    header, _, tokens = replay.load(path)
                    assert len(tokens) == 16
                    onset.analyze(path, OUT / (label + '-analysis'))
                    if spec:
                        events = [json.loads(l) for l in path.read_text().splitlines()]
                        assert any(e['event'] == 'round' and e['draft_q'] for e in events)
                    for variant in ('served', 'temp07', 'presence'):
                        position = 8
                        request = catch.branch_request(path, position, 8, 9001, variant, 1e-6)
                        _, branch, emitted = replay.request_capture(URL, directory, request, timeout=600)
                        assert [t['id'] for t in emitted[:position]] == [t['id'] for t in tokens[:position]]
                        events = [json.loads(l) for l in branch.read_text().splitlines()]
                        assert [e['at'] for e in events if e['event'] == 'branch'] == [position]
                        checked = [e for e in events if e['event'] == 'validation']
                        changed = sum(not e['equal'] for e in checked)
                        assert len(checked) == 9
                        if variant == 'served':
                            assert changed == 0
                        elif variant == 'temp07':
                            assert changed > 0
                        else:
                            assert events[0]['effective']['presence_penalty'] == 1
                            assert events[0]['effective']['repeat_last_n'] == 102400
                        validations.append(dict(label=label, variant=variant, rows_M=len(checked), drift_rows_M=changed))
                    if not spec:
                        bad = replay.body(path, 8, 1)
                        bad['__capture_expected']['0']['top'][0]['p'] += .1
                        before = set(directory.glob('request-*.jsonl'))
                        try:
                            replay.post(URL, '/completion', bad, timeout=600)
                        except RuntimeError as error:
                            assert 'capture replay distribution mismatch' in str(error)
                        else:
                            raise AssertionError('corrupted replay probability accepted')
                        failed, = set(directory.glob('request-*.jsonl')) - before
                        assert not any(json.loads(l)['event'] == 'branch' for l in failed.read_text().splitlines())
            finally:
                catch.study.stop(child)
                log.close()
        prefix = 'mtp3' if spec else 'no-spec'
        assert identities[prefix + '-off'] == identities[prefix + '-on'], 'capture changed seeded tokens'
    evidence = dict(model=MODEL, threads_D=4, capture_identity_tokens_M={p: len(identities[p + '-on']) for p in ('no-spec', 'mtp3')},
                    strict_replay_positions_M=[8], negative_control='corrupt probability rejected by server',
                    elapsed_seconds_M=time.monotonic() - start, timings=timings, validations=validations,
                    binary_sha256=catch.study.digest(binary / 'catch-cpu-srv'))
    catch.study.atomic(catch.ROOT / 'codex/CPU-CATCH.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
