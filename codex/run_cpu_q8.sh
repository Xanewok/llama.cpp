#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
export PATH="$PWD/codex/bin:$PATH" CCACHE_READONLY=1 CCACHE_TEMPDIR="$PWD/build-cpu/ccache-tmp"
mkdir -p "$CCACHE_TEMPDIR" codex/cpu-results
gate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
cpu=/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate
model=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
nice -n 19 "$gate" cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
nice -n 19 "$gate" cmake --build build-cpu --target server-context -j4
nice -n 19 "$gate" c++ -std=c++17 -Iggml/include -Iggml/src/ggml-cuda codex/cache-contract.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml-base -o build-cpu/bin/q8-cache-contract
nice -n 19 build-cpu/bin/q8-cache-contract > codex/cache-contract.log
if nice -n 19 build-cpu/bin/q8-cache-contract stale > codex/cache-mutant.log 2>&1; then
    echo 'negative control did not fail' >&2; exit 1
else
    test "$?" = 1
fi
nice -n 19 "$gate" c++ -std=c++17 -O2 -I. -Iinclude -Iggml/include -Icommon -Itools/server -Itools/mtmd -Ivendor codex/identity-cpu.cpp build-cpu/tools/server/libserver-context.a build-cpu/common/libllama-common-base.a -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama-common -lmtmd -lllama -lggml -lggml-base -pthread -ldl -o build-cpu/bin/q8-identity-cpu
export CPU_TEMP=0 CPU_UBATCH=128 LLAMA_ARG_LOG_VERBOSITY=4
for meta in 0 1; do
    for arm in off on; do
        if [ "$arm" = off ]; then export GGML_CUDA_DISABLE_Q8_REUSE=1; else unset GGML_CUDA_DISABLE_Q8_REUSE; fi
        "$cpu" build-cpu/bin/q8-identity-cpu "$model" "codex/cpu-results/meta-$meta-$arm.bin" "$meta" 3 1729 16 2 0 > "codex/cpu-results/meta-$meta-$arm.log" 2>&1
    done
    cmp "codex/cpu-results/meta-$meta-off.bin" "codex/cpu-results/meta-$meta-on.bin"
done
