import copy
import hashlib
import json
from pathlib import Path
import sys
import tomllib

sys.path.insert(0, '/home/xanewok/repos/qwen38-eval')
from harness import depth, lib

root = Path(__file__).resolve().parent.parent
real = Path('/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6')
fixtures = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')
x = tomllib.loads((root / 'codex/depth-experiment.toml').read_text())
depth.fixture_test(fixtures)
record = json.loads((real / 'depth-8192.json').read_text())
metric = lib.steady_intervals(record, 40, 5)
identity = lib.token_identity(record)
assert len(identity[0]) >= 256
corrupt = copy.deepcopy(record)
for chunk in corrupt['chunks']:
    if chunk.get('tokens'):
        chunk['tokens'][0] += 1
        break
try:
    refs = {}
    depth.settle(refs, 'cross-arm', 'up', 8192, identity)
    depth.settle(refs, 'cross-arm', 'patch', 8192, lib.token_identity(corrupt))
except lib.Fail as error:
    assert error.cls == 'integrity' if hasattr(error, 'cls') else 'identity' in str(error)
else:
    raise AssertionError('cross-arm token mutation escaped')
log = (real / 'server.log').read_text(errors='replace')
lib.verify_log_patterns(log, x['arms']['up']['canary_log'], 'recorded stack6')
try:
    lib.verify_log_patterns(log, x['arms']['patch']['canary_log'], 'unpatched negative control')
except lib.Fail:
    pass
else:
    raise AssertionError('unpatched log passed indexed engagement gate')
# The new log signature has no GPU recording yet. Use the emitted format as an additional synthetic positive.
lib.verify_log_patterns(log + '\ngdn_state_indexed: device=0 graph=123 tokens=6 gathers_skipped=1\n'
                            + 'gdn_state_indexed: device=1 graph=124 tokens=6 gathers_skipped=1\n',
                        x['arms']['patch']['canary_log'], 'synthetic indexed lines')
selected = None
with (real / 'round-trace.jsonl').open() as stream:
    for line in stream:
        row = json.loads(line)
        if any(p['phase'] == 'V' and p['n_tokens'] == 6 for p in row['phases']):
            selected = row
            break
assert selected
out = root / 'codex/parser-fixtures'
out.mkdir(exist_ok=True)
trace = out / 'stack6-round.jsonl'
trace.write_text(json.dumps(selected) + '\n')
lo = selected['phases'][0]['begin_ns']
counters = lib.trace_window(trace, lo, lo + 1, ['graph_hit', 'graph_miss', 'gdn_gather_skipped'])
assert counters['rounds'] == 1 and counters['gdn_gather_skipped']['count'] == 0
assert counters['graph_hit']['count'] > 0
(out / 'SOURCE.json').write_text(json.dumps(dict(source=str(real / 'round-trace.jsonl'), round=selected['round'], extracted_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()), indent=2) + '\n')
(root / 'codex/cpu-results/parser-results.json').write_text(json.dumps(dict(record=str(real / 'depth-8192.json'), metrics_M=metric, token_count_M=len(identity[0]), counters_M=counters, negative_controls='PASS', indexed_log_positive='synthetic only; real engagement required by canary'), indent=2) + '\n')
print('M: real SSE, timings, native tokens, identity, trace, memory and journal parsers passed; negative identity and indexed-engagement controls passed')
