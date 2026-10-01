#!/usr/bin/env python3
"""Exercise a live reasoning-budget transition with the same CPU harness."""
import json
import os
import shutil
from pathlib import Path

import cpu_e2e
import replay


def main():
    root = cpu_e2e.ROOT
    out = root / os.environ.get('CAPTURE_BUDGET_OUT', 'build-capture-budget')
    out.mkdir(exist_ok=False)
    cpu_e2e.OUT = out
    shutil.copy2(root / 'build-cpu/bin/llama-server', out / 'capture-cpu-srv')
    os.environ['LD_LIBRARY_PATH'] = str(root / 'build-cpu/bin')
    child, log, directory = cpu_e2e.boot('budget', True)
    try:
        request = dict(prompt='Count from one to one hundred.\n<think>', generation_prompt='<think>',
                       reasoning_budget_tokens=2, reasoning_budget_start_tag='<think>',
                       reasoning_budget_end_tags=['</think>'], reasoning_budget_message='[budget ended]',
                       seed=1234, temperature=1, top_k=20, top_p=.95, min_p=.05,
                       cache_prompt=False, n_predict=12, return_tokens=True)
        replay.post(cpu_e2e.URL, '/completion', request)
        source, = (directory / 'captures').glob('request-*.jsonl')
        _, _, tokens = replay.load(source)
        states = sorted(set(t['target']['budget_state'] for t in tokens))
        assert 1 in states and 2 in states and 4 in states, f'budget did not traverse counting/forcing/done: {states}'
        _, validation, _ = replay.request_capture(cpu_e2e.URL, directory / 'captures', replay.body(source, 6, 1))
        evidence = json.loads((root / 'codex/CPU-EVIDENCE.json').read_text())
        evidence.update(budget_states_transition_M=states, budget_replay_rows_M=7,
                        budget_source_capture=str(source), budget_validation_capture=str(validation))
        (root / 'codex/CPU-EVIDENCE.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(dict(states_M=states, replay_rows_M=7, capture=str(source))))
    finally:
        cpu_e2e.stop(child, log)


if __name__ == '__main__':
    main()
