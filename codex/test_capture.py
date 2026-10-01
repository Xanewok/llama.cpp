#!/usr/bin/env python3
"""CPU contract checks, including sentinels that forbid plan-mode execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import parent_gpu
import replay


def fixture(path):
    row = dict(top=[dict(id=7, p=1.0)], omitted_mass=0.0, budget_state=0)
    effective = dict(logit_bias=[dict(token=99, bias=None)], preserved_tokens=[100],
                     mirostat=0, adaptive_target=-1, xtc_probability=0)
    events = [dict(event='request', version=1, parsed=dict(chat_format=42, grammar_type='tool_calls',
                    grammar='root ::= "x"', grammar_lazy=True, grammar_triggers=[dict(type=0, value='<tool_call>')]),
                   effective=effective, target_seed=123, budget=dict(tokens=0, start=[1], end=[[2]], forced=[2]),
                   replay_force=[]), dict(event='prompt', ids=[1, 2, 3])]
    for i in range(3):
        events.append(dict(event='token', index=i, id=7, piece_hex='78', target=row))
    events.append(dict(event='end', emitted=3))
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    return events


def main():
    with tempfile.TemporaryDirectory(dir=parent_gpu.ROOT / 'build-tmp') as tmp:
        path = Path(tmp) / 'capture.jsonl'
        events = fixture(path)
        request = replay.body(path, 2, 8, 9001)
        assert request['prompt'] == [1, 2, 3] and request['__capture_force'] == [7, 7]
        assert request['__capture_expected']['2']['budget_state'] == 0
        assert request['chat_format'] == 42 and request['grammar_type'] == 'tool_calls'
        assert request['logit_bias'] == [[99, False]] and request['__capture_preserved'] == [100]
        assert request['__capture_budget']['tokens'] == 0
        assert request['n_predict'] == 10 and request['__capture_branch_seed'] == 9001
        assert replay.text([dict(piece_hex='c3'), dict(piece_hex='b3')]) == '\u00f3'
        events.insert(-1, dict(event='checkpoint_restore'))
        path.write_text(''.join(json.dumps(e) + '\n' for e in events))
        try:
            replay.load(path)
        except ValueError as error:
            assert 'unsupported' in str(error)
        else:
            raise AssertionError('checkpoint restore was silently ignored')
        def forbidden(*_args, **_kwargs):
            raise AssertionError('plan tried to run a process, HTTP request, or GPU query')
        with patch.dict(os.environ, RUN_PARENT_GPU='0', PLAN='1'), \
             patch('subprocess.Popen', forbidden), patch('subprocess.run', forbidden), \
             patch('urllib.request.OpenerDirector.open', forbidden), \
             patch('sys.argv', ['parent_gpu.py']), contextlib.redirect_stdout(io.StringIO()) as output:
            parent_gpu.main()
        plan = json.loads(output.getvalue())
        assert plan['response_cap_D'] == 40000 and plan['arms'] == ['S0', 'S4', 'no-spec']
        with patch.dict(os.environ, RUN_PARENT_GPU='0', PLAN='0'):
            try:
                parent_gpu.execution_gate()
            except RuntimeError:
                pass
            else:
                raise AssertionError('GPU gate accepted one flag')
        binary = Path(tmp) / 'bin'
        binary.mkdir()
        (binary / 'llama-server').write_bytes(b'server')
        (binary / 'capture-srv').write_bytes(b'server')
        (binary / 'libggml.so').write_bytes(b'library')
        (binary / 'SHA256SUMS').write_text(''.join(f'{parent_gpu.sha(p)}  {p.name}\n'
            for p in sorted(binary.iterdir())))
        parent_gpu.verify_frozen(binary, 'capture-srv')
        (binary / 'libggml.so').write_bytes(b'broken')
        try:
            parent_gpu.verify_frozen(binary, 'capture-srv')
        except RuntimeError as error:
            assert 'hash mismatch' in str(error)
        else:
            raise AssertionError('changed frozen library passed')
    print('PASS: replay roles, budget/grammar transport, UTF-8 bytes, unsupported transitions, plan sentinels, GPU gates, frozen negative control')


if __name__ == '__main__':
    main()
