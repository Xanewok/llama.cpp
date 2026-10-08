#!/usr/bin/env bash
# GPU entry for the parent queue only; no server is started by preflight or fixture replay.
set -euo pipefail
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
saved=${1:?saved harness/run.sh}; shift
code=$(cd "$(dirname "$saved")/.." && pwd)
data=$(cat "$code/source-root.txt")
export HARNESS_CODE_ROOT="$code" HARNESS_DATA_ROOT="$data" PYTHONPATH="$code:$data"
case " $* " in
    *' --preflight '*|*' --capture '*|*' --fixtures '*)
        export CUDA_VISIBLE_DEVICES=''
        exec nice -n 19 python3 -P "$ROOT/codex/gpu_stack10.py" "$@" ;;
esac
[[ ${GPUQ_ACTIVE:-0} == 1 ]] || { echo 'GPU entry requires gpuq'; exit 8; }
exec python3 -P "$ROOT/codex/gpu_stack10.py" "$@"
