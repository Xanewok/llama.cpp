#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/xanewok/repos/qwen38-eval
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
mode=${1:---preflight}
case "$mode" in --preflight|--canary|--execute) ;; *) exit 2 ;; esac
shift || true
if [[ "$mode" = --preflight ]]; then
    exec nice -n 19 python3 "$here/gdn_conv_gpu.py" bench "$here/frozen/gdn-conv-v2-proof" "$mode" "$@"
fi
[[ ${GPUQ_ACTIVE:-0} = 1 ]] || { echo 'GPU execution requires gpuq' >&2; exit 8; }
exec python3 "$here/gdn_conv_gpu.py" bench "$here/frozen/gdn-conv-v2-proof" "$mode" "$@"
