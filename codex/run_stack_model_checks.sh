#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export LD_LIBRARY_PATH="$PWD/build-stack-cpu/bin"
unset GGML_BACKEND_PATH LD_PRELOAD
for np in 1 2; do
    /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate python3 codex/timed_cpu.py "codex/logs/stack-state-movers-np$np.json" \
        build-stack-cpu/bin/test-recurrent-state-rollback --state-movers \
        -m /mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf -ngl 0 -t 8 -tb 8 -c 4096 -b 64 -ub 16 -np "$np" \
        > "codex/logs/stack-state-movers-np$np.log" 2>&1
done
export LD_LIBRARY_PATH="$PWD/build-cpu/bin"
/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate python3 codex/timed_cpu.py codex/logs/reserve-server.json \
    python3 codex/reserve_cpu.py > codex/logs/reserve-server.log 2>&1
