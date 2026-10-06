#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset GGML_BACKEND_PATH LD_PRELOAD
model=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
for arm in patch base; do
    if [[ $arm == patch ]]; then
        bindir="$PWD/build-cpu/bin"
    else
        bindir=/home/xanewok/repos/qwen38-eval-worktrees/wt/gdn-indexed-base/build-cpu/bin
    fi
    export LD_LIBRARY_PATH="$bindir"
    for np in 1 2; do
        set +e
        /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate python3 codex/timed_cpu.py "codex/logs/model-$arm-np$np.json" \
            build-cpu/bin/test-recurrent-state-rollback -m "$model" -ngl 0 -t 8 -tb 8 -c 4096 -b 64 -ub 16 -np "$np" \
            > "codex/logs/model-$arm-np$np.log" 2>&1
        rc=$?
        set -e
        printf '%s np%s rc=%s\n' "$arm" "$np" "$rc"
    done
done
export LD_LIBRARY_PATH="$PWD/build-cpu/bin"
/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate python3 codex/timed_cpu.py codex/logs/reserve-server.json \
    python3 codex/reserve_cpu.py > codex/logs/reserve-server.log 2>&1
