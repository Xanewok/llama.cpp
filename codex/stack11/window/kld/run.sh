#!/usr/bin/env bash
set -euo pipefail
# The gpuq daemon PATH has no CUDA.
export PATH=/usr/local/cuda-13.2/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
exec nice -n 19 python3 "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/runner.py" "$@"
