#!/usr/bin/env bash
# Parent-only: up/patch/patch/up, long SSE windows and cached agent follow-ups, <=2400 s per K.
# Plan by default. Parent execution also needs RUN_GPU=1, selected CUDA_VISIBLE_DEVICES and XID_SINCE.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
exec nice -n 19 python3 "$(dirname "$0")/gpu_harness.py" ab "$@"
