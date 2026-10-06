#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, '/home/xanewok/repos/qwen38-eval')
from harness import lib

BASELINE = Path('/home/xanewok/repos/qwen38-eval-worktrees/wt/gdn-gather-inplace/codex/gpu-results/depth/full-a1/0-up/depth-8192.json')


def check(out):
    expected = lib.token_identity(json.loads(BASELINE.read_text()))
    actual = lib.token_identity(json.loads((out / '0-patch/depth-8192.json').read_text()))
    good = expected == actual
    (out / 'canary-identity.json').write_text(json.dumps({'recorded_baseline': str(BASELINE), 'identity': 'PASS' if good else 'FAIL'}))
    if not good:
        raise lib.Fail('cross-arm identity failed against the recorded stack6 canary baseline', 'integrity')
    print('canary recorded stack6 identity: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args, _ = parser.parse_known_args()
    try:
        check(args.out)
    except lib.Fail as error:
        print(str(error), flush=True)
        sys.exit(error.rc)
    except (OSError, ValueError) as error:
        print('cross-arm identity failed: ' + str(error), flush=True)
        sys.exit(6)
