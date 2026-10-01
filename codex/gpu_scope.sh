#!/usr/bin/env bash
# Parent only. Without --run, no GPU tools or CUDA binaries are executed.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ ${1:-} == --run ]]; then
    exec nice -n 19 python3 "$ROOT/codex/scope_gpu.py" --run --arch 120 --local "$@"
fi
exec nice -n 19 python3 "$ROOT/codex/scope_gpu.py" --selftest
