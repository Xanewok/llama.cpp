#!/usr/bin/env bash
# gpuq entry for mtp-fuse-depth-j5q: harness/run.sh with the adapter in place of `python -m harness depth`.
set -euo pipefail
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/xanewok/repos/qwen38-eval
here=$(cd "$(dirname "$0")" && pwd)
case " $* " in *" --preflight "*|*" --fixtures "*) exec nice -n 19 python3 -P "$here/mtp_fuse_gpu.py" "$@" ;; esac
exec python3 -P "$here/mtp_fuse_gpu.py" "$@"
