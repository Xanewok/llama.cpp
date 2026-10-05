#!/usr/bin/env bash
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/xanewok/repos/qwen38-eval
exec nice -n 19 python3 "$(dirname "$0")/fa_memcheck.py" "$@"
