#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
mode=${1:---preflight}
case "$mode" in
    --preflight|--capture|--canary|--execute) ;;
    *) echo "usage: $0 --preflight|--capture|--canary|--execute [harness options]" >&2; exit 2 ;;
esac
shift || true
if [[ "$mode" = --canary || "$mode" = --execute ]]; then
    [[ ${GPUQ_ACTIVE:-0} = 1 ]] || { echo 'GPU execution requires gpuq' >&2; exit 8; }
fi
export PYTHONPATH=/home/xanewok/repos/qwen38-eval
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ "$mode" = --preflight || "$mode" = --capture ]]; then
    exec nice -n 19 python3 "$here/gdn_conv_gpu.py" depth "$here/gdn-conv-v2.experiment.toml" "$mode" "$@"
fi
exec python3 "$here/gdn_conv_gpu.py" depth "$here/gdn-conv-v2.experiment.toml" "$mode" "$@"
