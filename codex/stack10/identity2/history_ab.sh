#!/usr/bin/env bash
set -euo pipefail
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "$HERE/../identity/snapshots/harness-385c353bf0ab87df00dc6b9773c80ae4827f0dec25cf821cb88c76e57d3393a9" && pwd)
export HARNESS_CODE_ROOT="$CODE" HARNESS_DATA_ROOT=/home/xanewok/repos/qwen38-eval PYTHONPATH="$CODE:/home/xanewok/repos/qwen38-eval"
saved=${1:?saved harness/run.sh}
shift
[[ $saved == "$CODE/harness/run.sh" ]] || { echo 'unexpected harness snapshot'; exit 6; }
if [[ " $* " == *' --preflight '* ]]; then
    export CUDA_VISIBLE_DEVICES=''
    exec nice -n 19 taskset -c 6-11,18-23 python3 "$HERE/history_runner.py" "$@"
fi
[[ ${GPUQ_ACTIVE:-0} == 1 ]] || { echo 'GPU entry requires gpuq'; exit 8; }
exec python3 "$HERE/history_runner.py" "$@"
