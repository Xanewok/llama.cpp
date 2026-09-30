#!/usr/bin/env bash
# Parent-only. Plan mode does not inspect or execute a GPU backend.
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
execute=0
for arg in "$@"; do [[ $arg != --execute ]] || execute=1; done
if (( execute )); then
    [[ ${RUN_GPU_AB:-0} == 1 && -n ${CUDA_VISIBLE_DEVICES:-} ]] || { echo 'Parent must set RUN_GPU_AB=1 and CUDA_VISIBLE_DEVICES'; exit 4; }
else
    export CUDA_VISIBLE_DEVICES=""
fi
exec nice -n 19 python3 "$(dirname "$0")/gpu_ab.py" "$@"
