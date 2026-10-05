import sys
import tomllib
from pathlib import Path

sys.path.insert(0, '/home/xanewok/repos/qwen38-eval')
from harness import lib

root = Path(__file__).resolve().parent
recorded = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/server.log').read_text()
config = tomllib.loads((root / 'depth-experiment.toml').read_text())
patterns = config['arms']['patch']['canary_log'][2:]
for pattern in patterns:
    try:
        lib.verify_log_patterns(recorded, [pattern], 'recorded baseline')
    except lib.Fail:
        pass
    else:
        raise AssertionError('recorded baseline must fail a new-path engagement gate')

# These positive format checks are synthetic; the real positive check belongs to the live canary.
for device in (0, 1):
    sample = f'q8_reuse: device={device} graph=1 rows=6 skipped=1 quantized=1 bytes=128\nadd_rms_norm: device={device} rows=6 columns=5120\n'
    relevant = [p for p in patterns if f'device={device}' in p]
    lib.verify_log_patterns(sample, relevant, 'format')
    try:
        lib.verify_log_patterns(sample.replace('skipped=1', 'skipped=0'), relevant, 'zero engagement')
    except lib.Fail:
        pass
    else:
        raise AssertionError('zero skips accepted')
print('M: recorded baseline rejects new engagement gates; synthetic positive/zero-count format checks PASS')
