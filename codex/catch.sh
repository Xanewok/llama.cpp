#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 python3 "$(dirname "$0")/catch.py" "$@"
