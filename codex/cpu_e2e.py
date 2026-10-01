#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error

import replay

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / os.environ.get('CAPTURE_CPU_OUT', 'build-capture-test')
MODEL = '/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf'
URL = 'http://127.0.0.1:18127'


def boot(label, capture, spec=False):
    directory = OUT / label
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               TMPDIR=str(ROOT / 'build-tmp'))
    env.pop('LLAMA_CAPTURE_DIR', None)
    if capture:
        (directory / 'captures').mkdir()
        env.update(LLAMA_CAPTURE_DIR=str(directory / 'captures'), LLAMA_CAPTURE_PROBS='1', LLAMA_CAPTURE_LIMIT='48')
    cmd = ['nice', '-n', '19', str(OUT / 'capture-cpu-srv'), '-m', MODEL, '-ngl', '0', '-np', '1',
           '-c', '4096', '-b', '512', '-ub', '128', '-t', '4', '-tb', '4', '-fa', 'on',
           '--seed', '1234', '--host', '127.0.0.1', '--port', '18127', '--jinja', '--fit', 'off', '--backend-sampling']
    if spec:
        cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '4', '--spec-draft-n-min', '0',
                '--spec-draft-p-min', '0', '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling']
    else:
        cmd += ['--spec-type', 'none']
    log = (directory / 'server.log').open('w')
    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            if child.poll() is not None:
                raise RuntimeError('CPU server died: ' + str(directory / 'server.log'))
            try:
                replay.post(URL, '/tokenize', {'content': 'x'})
                return child, log, directory
            except (OSError, urllib.error.HTTPError):
                time.sleep(.25)
        raise RuntimeError('CPU server boot timeout')
    except BaseException:
        stop(child, log)
        raise


def stop(child, log):
    child.terminate()
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)
    log.close()


def main():
    OUT.mkdir(exist_ok=False)
    shutil.copy2(ROOT / 'build-cpu/bin/llama-server', OUT / 'capture-cpu-srv')
    env_path = str(ROOT / 'build-cpu/bin')
    os.environ['LD_LIBRARY_PATH'] = env_path
    simple = dict(prompt='Count from one to ten, with one number per line.', seed=1234, temperature=1,
                  top_k=20, top_p=.95, min_p=.05, cache_prompt=False, return_tokens=True, n_predict=24)
    results = {}
    for label, enabled in [('off', False), ('on', True)]:
        child, log, directory = boot(label, enabled)
        try:
            result = replay.post(URL, '/completion', simple)
            results[label] = result['tokens']
        finally:
            stop(child, log)
    assert results['off'] == results['on'], 'capture changed sampled tokens'
    spec_identity = {}
    for label, enabled in [('off-mtp', False), ('on-mtp', True)]:
        child, log, directory = boot(label, enabled, spec=True)
        try:
            spec_identity[label] = replay.post(URL, '/completion', simple)['tokens']
        finally:
            stop(child, log)
    assert spec_identity['off-mtp'] == spec_identity['on-mtp'], 'capture changed speculative sampled tokens'
    child, log, directory = boot('mtp', True, spec=True)
    start = time.monotonic()
    try:
        tool = dict(type='function', function=dict(name='emit_rows', description='Emit integer rows.',
            parameters=dict(type='object', properties=dict(rows=dict(type='array', items=dict(type='integer'))),
                            required=['rows'], additionalProperties=False)))
        request = dict(messages=[dict(role='user', content='Call emit_rows with the list of integers from zero to one hundred. Do not explain.')],
                       tools=[tool], tool_choice='auto', chat_template_kwargs=dict(enable_thinking=False),
                       seed=1234, temperature=1, top_k=20, top_p=.95, min_p=.05,
                       cache_prompt=False, max_tokens=200, return_tokens=True)
        # Force entry into the tool grammar; the remaining rows are sampled by the real chain.
        trigger = replay.post(URL, '/tokenize', dict(content='<tool_call>', add_special=False, parse_special=True))['tokens']
        assert len(trigger) == 1
        request['logit_bias'] = [[trigger[0], 100]]
        response = replay.post(URL, '/v1/chat/completions', request)
        path, = (directory / 'captures').glob('request-*.jsonl')
        header, prompt, tokens = replay.load(path)
        assert header['request']['tools'] == [tool]
        assert header['effective']['grammar_lazy'] and header['effective']['grammar_triggers']
        assert '<tool_call>' in replay.text(tokens), 'lazy grammar never triggered'
        assert len(tokens) == 48, 'environment cap did not override request'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert any(e['event'] == 'round' and e['draft'] and e['draft_q'] for e in events), 'no probabilistic drafts'
        position = 12
        validation = replay.body(path, position, 1)
        _, validation_path, _ = replay.request_capture(URL, directory / 'captures', validation)
        branches = []
        for seed in (9001, 9002):
            _, branch_path, emitted = replay.request_capture(URL, directory / 'captures', replay.body(path, position, 8, seed))
            assert len(emitted) == position + 8
            assert any(e['event'] == 'branch' for e in map(json.loads, branch_path.read_text().splitlines()))
            branches.append(str(branch_path))
        broken = replay.body(path, position, 1)
        broken['__capture_expected']['0']['top'][0]['p'] += .25
        before_negative = set((directory / 'captures').glob('request-*.jsonl'))
        try:
            replay.request_capture(URL, directory / 'captures', broken)
        except (RuntimeError, urllib.error.HTTPError) as error:
            negative = str(error)
            assert 'capture replay distribution mismatch' in negative
            negative_path, = set((directory / 'captures').glob('request-*.jsonl')) - before_negative
            negative_events = list(map(json.loads, negative_path.read_text().splitlines()))
            assert any(e['event'] == 'validation' and e['equal'] is False for e in negative_events)
            assert not any(e['event'] == 'branch' for e in negative_events)
        else:
            raise AssertionError('corrupt distribution passed replay')
        evidence = dict(capture_token_identity_M=len(results['on']), spec_capture_token_identity_M=len(spec_identity['on-mtp']), generated_tokens_M=len(tokens),
                        prompt_tokens_M=len(prompt), lazy_grammar_triggered_M=True,
                        replay_rows_M=position + 1, branches_M=2, branch_length_M=8,
                        negative_control=negative, validation=str(validation_path), branches=branches,
                        elapsed_seconds_M=time.monotonic()-start)
        (ROOT / 'codex/CPU-EVIDENCE.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(evidence))
    finally:
        stop(child, log)


if __name__ == '__main__':
    main()
