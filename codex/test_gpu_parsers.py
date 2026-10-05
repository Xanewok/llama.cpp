import copy
import json
from pathlib import Path
from harness import depth, lib

root = Path(__file__).resolve().parent
x = lib.read_experiment(root / 'gdn-state-movers.experiment.toml')
fixtures = Path(x['depth']['fixtures'])
depth.fixture_test(fixtures)
record_path = Path('/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack4-depth-j5q/full-a1/0-up/depth-8192.json')
record = json.loads(record_path.read_text())
result, identity = depth.gate(x['depth'], record, depth.first_prefill(8192))
results = {}
depth.settle(results, 'cross-arm', 'up', 8192, identity)
depth.settle(results, 'cross-arm', 'patch', 8192, identity)
changed = copy.deepcopy(identity)
changed = list(changed)
changed[0] = list(changed[0])
changed[0][0] += 1
try:
    depth.settle(results, 'cross-arm', 'patch', 8192, changed)
except lib.Fail as error:
    assert error.cls == 'integrity'
else:
    raise AssertionError('cross-arm identity negative control escaped')
short = copy.deepcopy(record)
short['chunks'] = [{}, {'timings': None}]
try:
    depth.gate(x['depth'], short, depth.first_prefill(8192))
except (lib.Fail, ValueError):
    pass
else:
    raise AssertionError('empty stream passed gates')
print(json.dumps({'recorded_native_artifact': str(record_path), 'label': 'M', 'steady_intervals': result['intervals_M'], 'native_tokens': len(identity[0]), 'cross_arm_identity_negative_control': 'PASS', 'empty_stream_negative_control': 'PASS'}, indent=2))
