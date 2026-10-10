#!/usr/bin/env bash
set -euo pipefail
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} == preflight ]]; then
    export CUDA_VISIBLE_DEVICES=''
    exec nice -n 19 taskset -c 6-11,18-23 python3 "$HERE/kld_item_v2.py" "$@"
fi
[[ ${GPUQ_ACTIVE:-0} == 1 ]] || { echo 'GPU entry requires gpuq'; exit 8; }
exec python3 "$HERE/kld_item_v2.py" "$@"
