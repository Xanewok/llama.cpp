#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
while true; do
    status=$(gpuq status --brief)
    printf '%s\n' "$status"
    if [[ "$status" != *"RUN "* ]]; then
        break
    fi
    sleep 30
done
exec /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate "$@"
