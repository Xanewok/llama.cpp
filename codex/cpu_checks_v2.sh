#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset LD_PRELOAD LD_LIBRARY_PATH
model=${MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
for np in 1 2; do
    for arm in base patch; do
        lib="$PWD/build-cpu/bin"
        if [[ $arm = base ]]; then lib="$PWD/build-cpu/base:$lib"; fi
        /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate env LD_LIBRARY_PATH="$lib" \
            /usr/bin/time -p build-cpu/bin/test-recurrent-state-rollback --state-movers \
            -m "$model" -ngl 0 -t 4 -tb 4 -c 4096 -b 64 -ub 16 -np "$np" \
            > "codex/cpu-state-v2-$arm-np$np.log" 2> "codex/cpu-state-v2-$arm-np$np.time"
    done
done
nice -n 19 python3 codex/check_cpu_identity_v2.py > codex/cpu-identity-v2.json
