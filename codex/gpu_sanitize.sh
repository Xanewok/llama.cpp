#!/usr/bin/env bash
# Parent only. Without --run, only CPU selftests execute; no GPU tool or CUDA binary is started.
# Run: gpu_sanitize.sh --run --stage san|repro|synccheck|racecheck|build [--canary] [--budget N] [--out DIR]
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ ${1:-} == --run ]]; then
    shift
    exec nice -n 19 python3 "$ROOT/codex/scope_gpu.py" --run --arch 120 --local "$@"
fi
nice -n 19 python3 "$ROOT/codex/scope_gpu.py" --selftest
exec nice -n 19 python3 "$ROOT/codex/test_sanitize.py"
