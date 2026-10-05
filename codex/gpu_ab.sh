#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
if [ "${1:-}" = --parser-test ]; then
    export PYTHONPATH=/home/xanewok/repos/qwen38-eval
    exec nice -n 19 python3 -c 'from harness.depth import fixture_test; fixture_test("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")'
fi
if [ "${1:-}" != --preflight ] && [ "${GPUQ_ACTIVE:-}" != 1 ]; then
    echo 'GPU queue required' >&2
    exit 2
fi
exec bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$here/depth-experiment.toml" "$@"
