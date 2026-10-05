#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$(dirname "$0")/.."
export PATH="$PWD/codex/gate-bin:$PATH"
unset GGML_BACKEND_PATH
export LD_LIBRARY_PATH="$PWD/build-cpu/bin"
cp build-cpu/bin/test-backend-ops build-cpu/bin/gdn-backend-cpu
export LD_PRELOAD="$PWD/build-cpu/bin/backend-thread-cap.so"
# The test executable otherwise sets its thread counts from host hardware_concurrency().
nice -n 19 /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate build-cpu/bin/gdn-backend-cpu test -b CPU -o 'GATED_DELTA_NET.*' -j 1 > codex/cpu-results/backend-eval.log 2>&1
nice -n 19 /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate build-cpu/bin/gdn-backend-cpu support -b CPU -o 'GATED_DELTA_NET_INDEXED.*' --output csv > codex/cpu-results/backend-support.csv 2> codex/cpu-results/backend-support.log
