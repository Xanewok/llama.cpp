#!/usr/bin/env python3
import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
EVAL = Path('/home/xanewok/repos/qwen38-eval')
sys.path.insert(0, str(EVAL))
from harness import depth, lib

HERE = Path(__file__).resolve().parent
RECORDED = Path('/home/xanewok/repos/qwen38-eval-worktrees/wt/gdn-gather-inplace/codex/gpu-results/depth/full-a1')
FIXTURES = EVAL / 'orchestration/gpuq/fixtures'


def main():
    depth.fixture_test(FIXTURES)
    records = [json.loads((RECORDED / arm / 'depth-8192.json').read_text()) for arm in ['0-up', '1-patch']]
    identity = [lib.token_identity(record) for record in records]
    assert identity[0] == identity[1]
    results = {}
    depth.settle(results, 'cross-arm', 'up', 8192, identity[0])
    try:
        depth.settle(results, 'cross-arm', 'patch', 8192, ([999999], 1, 0))
    except lib.Fail:
        pass
    else:
        raise AssertionError('identity negative control escaped')
    for record in records:
        assert lib.steady_intervals(record, 40, 5)
    counts = lib.trace_window(RECORDED / '1-patch/round-trace.jsonl', *records[1]['mono_ns'],
                              ['graph_hit', 'graph_miss', 'gdn_gather_skipped'])
    assert counts['rounds'] > 40 and counts['gdn_gather_skipped']['count'] > 0
    log = (RECORDED / '1-patch/server.log').read_text(errors='replace')
    patterns = [r'gdn_state_indexed: device=0 graph=\d+ tokens=6 gathers_skipped=[1-9]\d*\b',
                r'gdn_state_indexed: device=1 graph=\d+ tokens=6 gathers_skipped=[1-9]\d*\b']
    lib.verify_log_patterns(log, patterns, 'recorded indexed engagement')
    try:
        lib.verify_log_patterns(log.replace('gdn_state_indexed:', 'disabled:'), patterns, 'negative control')
    except lib.Fail:
        pass
    else:
        raise AssertionError('engagement negative control escaped')
    spec = importlib.util.spec_from_file_location('sanitizer_parsers', HERE / 'memcheck/memcheck.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.selftest()
    mutant = (HERE / "logs/negative-row0.log").read_text()
    assert module.parse_cases(mutant, "GATED_DELTA_NET_INDEXED")["fail"] == 13
    spec = importlib.util.spec_from_file_location('canary_control', HERE / 'canary_identity.py')
    canary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canary)
    out = HERE / 'fixtures/canary-control'
    (out / '0-patch').mkdir(parents=True, exist_ok=True)
    target = out / '0-patch/depth-8192.json'
    target.write_text(json.dumps(records[1]))
    canary.check(out)
    changed = copy.deepcopy(records[1])
    next(chunk for chunk in changed['chunks'] if chunk.get('tokens'))['tokens'][0] = 999999
    target.write_text(json.dumps(changed))
    try:
        canary.check(out)
    except lib.Fail:
        pass
    else:
        raise AssertionError('canary identity negative control escaped')
    target.write_text(json.dumps(records[1]))
    canary.check(out)
    print('depth identity, steady, trace and engagement: real GPU artifacts and negative controls PASS')


if __name__ == '__main__':
    main()
