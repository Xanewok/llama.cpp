#!/usr/bin/env bash
# Parent only. Example: bash codex/arch_bench.sh --run --arch 75 --gpus 0
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export SCOPE_GPUS=${SCOPE_GPUS:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ ${1:-} == --run ]]; then
    exec nice -n 19 python3 "$ROOT/codex/scope_gpu.py" "$@"
fi
exec nice -n 19 python3 "$ROOT/codex/scope_gpu.py" --selftest
