#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
cd "$(dirname "$0")/.."
"$HOME/repos/qwen38-eval-worktrees/bin/cpugate" ./codex/gate.sh \
    env PYTHONPATH="$PWD/codex" python3 -m unittest test_spec_tf test_estimator_io test_import_streams test_gpu_payload
