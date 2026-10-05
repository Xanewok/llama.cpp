#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset LD_PRELOAD LD_LIBRARY_PATH
model=${MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON
nice -n 19 cmake --build build-cpu --target test-backend-ops test-recurrent-state-rollback -j8
nice -n 19 c++ -shared -fPIC codex/thread_cap.cpp -o build-cpu/thread-cap.so
cp build-cpu/bin/test-backend-ops build-cpu/bin/cpu-ops
LD_PRELOAD="$root/build-cpu/thread-cap.so" nice -n 19 build-cpu/bin/cpu-ops -b CPU -o SSM_CONV_STATE_FUSION > codex/cpu-ops.log 2>&1
nice -n 19 python3 codex/build_cpu_base.py > codex/cpu-base-build.log 2>&1
args=(--state-movers -m "$model" -ngl 0 -t 4 -tb 4 -c 256 -b 64 -ub 16)
nice -n 19 build-cpu/bin/test-recurrent-state-rollback "${args[@]}" > codex/cpu-state-movers-patch.log 2>&1
LD_LIBRARY_PATH="$root/build-cpu/base:$root/build-cpu/bin" nice -n 19 build-cpu/bin/test-recurrent-state-rollback "${args[@]}" > codex/cpu-state-movers-base.log 2>&1
nice -n 19 python3 codex/check_cpu_identity.py
