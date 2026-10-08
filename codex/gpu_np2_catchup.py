"""Use the existing frozen-build harness and stack7 budget/cleanup guards."""
import os
from pathlib import Path
import re
import sys

STACK7 = Path('/home/xanewok/repos/qwen38-eval-worktrees/wt/stack7/codex')
sys.path.insert(0, str(STACK7))
import gpu_stack7 as s7

EVAL = Path('/home/xanewok/repos/qwen38-eval')
GOOD = EVAL / 'results/2026-10-08-bg-exact-j5q/canary-a1/cand-np2/server.log'
BAD = EVAL / 'results/2026-10-07-stack7-np2-real-j5q/canary-a2/0-patch/server.log'


def check_patch_log(text):
    s7.lib.verify_log_patterns(text, [r'n_slots = 2\b', r'model loaded\b'], 'np2 patch boot')
    if re.search(r'backend sampling with SPLIT_MODE_TENSOR needs[^\n]*using CPU|backend sampling is not compatible', text):
        raise s7.lib.Fail('patch target sampler fell back to the full-row path', 'integrity')


def fixtures():
    s7.fixture_checks()
    check_patch_log(GOOD.read_text())
    try:
        check_patch_log(BAD.read_text())
    except s7.lib.Fail:
        pass
    else:
        raise AssertionError('recorded target sampler refusal escaped')
    print('recorded GPU np2 boot/refusal parser PASS; no model or GPU execution')


def install(argv):
    s7.install_runtime_guards(argv)
    original = s7.depth.evaluate_boot

    def evaluate(x, dest, arm, results, canary=False):
        failures = original(x, dest, arm, results, canary)
        if arm == 'patch':
            try:
                check_patch_log((dest / 'server.log').read_text())
            except (OSError, s7.lib.Fail) as error:
                failures.append(s7.lib.Fail(str(error), 'integrity'))
        return failures

    s7.depth.evaluate_boot = evaluate


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixtures']:
        fixtures()
    else:
        if '--execute' in sys.argv or '--canary' in sys.argv:
            if os.environ.get('GPUQ_ACTIVE') != '1':
                raise SystemExit('execution requires gpuq')
        install(sys.argv[1:])
        raise SystemExit(s7.harness_main())
