#!/usr/bin/env bash
# Parent only. Default mode plans without querying GPUs or starting any process.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PLAN=${PLAN:-1} RUN_PARENT_GPU=${RUN_PARENT_GPU:-0}
exec nice -n 19 python3 "$ROOT/codex/parent_gpu.py" "$@"
