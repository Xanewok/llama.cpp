#!/usr/bin/env python3
"""CPU-only controls; never execute a server, GPU tool, or HTTP request."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import gpu_ab as ab


def fails(function):
    try:
        function()
    except RuntimeError:
        return
    raise AssertionError('negative control escaped')


def main():
    with tempfile.TemporaryDirectory(dir=ab.ROOT / 'codex') as directory:
        root = Path(directory)
        source = root / 'source'
        (source / 'bin').mkdir(parents=True)
        (source / 'SOURCE').write_text('a' * 40 + '\n')
        binary = source / 'bin' / 'llama-server'
        binary.write_bytes(b'CPU fixture; never executed')
        binary.chmod(0o755)
        library = source / 'bin' / 'libfixture.so'
        library.write_bytes(b'library fixture')
        (source / 'bin' / 'SHA256SUMS').write_text(f'{ab.sha(binary)}  llama-server\n{ab.sha(library)}  libfixture.so\n')
        executable, frozen = ab.freeze(source, root / 'frozen', 'llama-server')
        assert executable.name == 'draft-loop-ab'
        assert not (executable.parent / 'llama-server').exists()
        ab.verify(executable.parent.parent, frozen)
        executable.write_bytes(b'changed')
        fails(lambda: ab.verify(executable.parent.parent, frozen))
        library.write_bytes(b'changed')
        fails(lambda: ab.freeze(source, root / 'broken', 'llama-server'))
    choices = [{'message': {'content': 'same', 'tool_calls': [{'id': 'random', 'function': {'name': 'f', 'arguments': '{}'}}]}, 'finish_reason': 'stop'}]
    counters = dict(zip(ab.COUNTERS, [20, 60, 40]))
    records = [dict(arm=arm, boot=boot, k=3, seed=1, kind=kind, counters=copy.deepcopy(counters),
                    timings={'predicted_ms': 600 if arm == 'up' else 580}, choices=copy.deepcopy(choices))
               for kind in ('tool', 'think') for boot, arm in enumerate(ab.ORDER)]
    assert 'patch-up -1.000000' in ab.summarize(records, [3], [1])
    changed = copy.deepcopy(records)
    changed[1]['choices'][0]['message']['content'] = 'different'
    fails(lambda: ab.summarize(changed, [3], [1]))
    changed = copy.deepcopy(records)
    changed[1]['counters'][ab.COUNTERS[2]] += 1
    fails(lambda: ab.summarize(changed, [3], [1]))
    fails(lambda: ab.summarize(records[1:], [3], [1]))
    fails(lambda: ab.arm_env('{"CUDA_VISIBLE_DEVICES":"0"}'))
    with patch.object(sys, 'argv', ['gpu_ab.py']), contextlib.redirect_stdout(io.StringIO()):
        with patch.object(ab.subprocess, 'run', side_effect=AssertionError('plan ran a command')):
            with patch.object(ab.subprocess, 'Popen', side_effect=AssertionError('plan started a process')):
                with patch.object(ab.urllib.request, 'urlopen', side_effect=AssertionError('plan used HTTP')):
                    ab.main()
    print('M: frozen-copy, renamed-executable, paired-summary and read-only-plan checks passed')
    print('M: 6 negative controls rejected changed bytes, completion, counters, missing boots and protected env')


if __name__ == '__main__':
    main()
