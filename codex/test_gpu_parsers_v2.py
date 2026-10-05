import copy
import contextlib
import io
import json
from pathlib import Path

from harness import depth, lib
from gdn_conv_gpu import bench_engagement, bench_results, engagement

root = Path(__file__).resolve().parent
x = lib.read_experiment(root / 'gdn-conv-v2.experiment.toml')
fixtures = Path(x['depth']['fixtures'])
fixture_output = io.StringIO()
with contextlib.redirect_stdout(fixture_output):
    depth.fixture_test(fixtures)
fixture_result = json.loads(fixture_output.getvalue())
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

def fails(action):
    try:
        action()
    except (lib.Fail, ValueError):
        return
    raise AssertionError('negative control escaped')

fails(lambda: depth.settle(results, 'cross-arm', 'patch', 8192, changed))
short = copy.deepcopy(record)
short['chunks'] = [{}, {'timings': None}]
fails(lambda: depth.gate(x['depth'], short, depth.first_prefill(8192)))
bench_dir = Path('/home/xanewok/repos/qwen38-eval-worktrees/work/gdn-movers-diag/runs/bench-a2')
real_bench = (bench_dir / 'gpu0-p1.txt').read_text()
bench = bench_results(real_bench, legacy=True)
fails(lambda: bench_results(real_bench.replace('max_abs_cache=0', 'max_abs_cache=1'), legacy=True))
fails(lambda: bench_results(real_bench.replace('us_per_layer_async=19.078', 'us_per_layer_async=nan'), legacy=True))
fails(lambda: bench_results(real_bench.splitlines()[1], legacy=True))
# Real V1 logs carry no proof. Both new proof parsers must reject them.
real_log = (bench_dir / 'gpu0-p1.err').read_text()
fails(lambda: engagement(real_log, 48, 6, [0]))
fails(lambda: bench_engagement(real_log, 48, 6, 0))
positive = ''.join(f'ssm_conv_state_fusion: engaged conv_output_raw-{layer} device={dev} graph={100+dev} tokens=6 channels=5120 seqs=1 slots=6 nodes=20\n'
                   for dev in (0, 1) for layer in range(48))
positive += ''.join(f'ssm_conv_state_fusion: graph device={dev} graph={100+dev} tokens=6 conv=48 fused=48\n' for dev in (0, 1))
assert len(engagement(positive, 48, 6, [0, 1])) == 2
fails(lambda: engagement(positive.replace('conv_output_raw-47', 'conv_output_raw-46'), 48, 6, [0, 1]))
fails(lambda: engagement(positive + 'ssm_conv_state_fusion: rejected snap device=0 graph=100 tokens=6: external consumer\n', 48, 6, [0, 1]))
fails(lambda: engagement(positive.replace('conv=48 fused=48', 'conv=48 fused=47'), 48, 6, [0, 1]))
fails(lambda: engagement(positive + 'ssm_conv_state_fusion: graph device=0 graph=999 tokens=6 conv=1 fused=0\n', 48, 6, [0, 1]))
one_device = '\n'.join(l for l in positive.splitlines() if 'device=0 ' in l) + '\n'
# Meta separates CUDA graphs at AllReduce boundaries; the device must still cover every layer.
fragmented = ''.join(f'ssm_conv_state_fusion: engaged conv_output_raw-{layer} device={dev} graph={1000+dev*100+layer} tokens=6 channels=5120 seqs=1 slots=6 nodes=20\n'
                     for dev in (0, 1) for layer in range(48))
fragmented += ''.join(f'ssm_conv_state_fusion: graph device={dev} graph={1000+dev*100+layer} tokens=6 conv=1 fused=1\n' for dev in (0, 1) for layer in range(48))
assert len(engagement(fragmented, 48, 6, [0, 1], fragmented=True)) == 96
fails(lambda: engagement(fragmented.replace('conv_output_raw-47', 'conv_output_raw-46'), 48, 6, [0, 1], fragmented=True))
bench_positive = ''.join('BENCH_MODE ' + mode + '\n' + (one_device if mode != 'sep-unfused' else '')
                         for mode in ('sep-unfused', 'sep-fused', 'galloc-model', 'galloc-pinned'))
assert len(bench_engagement(bench_positive, 48, 6, 0)) == 3
print(json.dumps({'label': 'M', 'shared_fixture_tests': fixture_result, 'recorded_native_artifact': str(record_path),
                  'steady_intervals_M': result['intervals_M'], 'native_tokens_M': len(identity[0]),
                  'recorded_bench_artifact': str(bench_dir / 'gpu0-p1.txt'),
                  'real_bench_modes_M': len(bench), 'identity_negative_control': 'DETECTED',
                  'empty_stream_negative_control': 'DETECTED', 'bench_corruption_controls': 'DETECTED',
                  'real_V1_engagement_missing': 'DETECTED', 'partial_engagement_controls': 'DETECTED',
                  'positive_engagement_fixture': 'synthetic; no V2 GPU execution yet'}, indent=2))
