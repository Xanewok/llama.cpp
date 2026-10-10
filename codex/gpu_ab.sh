#!/usr/bin/env bash
set -euo pipefail
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd)
saved=${1:?saved harness/run.sh}; shift
code=$(cd "$(dirname "$saved")/.." && pwd)
data=$(cat "$code/source-root.txt")
export HARNESS_CODE_ROOT="$code" HARNESS_DATA_ROOT="$data" PYTHONPATH="$code:$data"
case " $* " in
    *' --preflight '*|*' --capture '*)
        export CUDA_VISIBLE_DEVICES=''
        exec nice -n 19 taskset -c 6-11,18-23 python3 -P "$HERE/vanilla_runner.py" "$@" ;;
esac
[[ ${GPUQ_ACTIVE:-0} == 1 ]] || { echo 'GPU entry requires gpuq'; exit 8; }
exec python3 -P "$HERE/vanilla_runner.py" "$@"
