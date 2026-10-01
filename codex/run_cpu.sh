#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$PWD/build-cpu/tmp"
mkdir -p "$TMPDIR" codex
cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF > codex/configure-cpu.log 2>&1
cmake --build build-cpu --target test-backend-ops -j8 > codex/build-cpu.log 2>&1
cp build-cpu/bin/test-backend-ops build-cpu/bin/swift-fa-cpu
export LLAMA_TEST_SWIFT_FA=1
set +e
build-cpu/bin/swift-fa-cpu test -b CPU -o FLASH_ATTN_EXT -j 1 > codex/cpu-results.log 2>&1
positive_rc=$?
set -e
printf "CPU suite exit: %s\n" "$positive_rc"
set +e
LLAMA_TEST_SWIFT_FA_CORRUPT=1 build-cpu/bin/swift-fa-cpu test -b CPU -o FLASH_ATTN_EXT -p 'nr23=\[6,1\],kv=4096,nb=5,.*type_K=q8_0.*pattern=causal,compare_rows=0,used_kv=4096$' -j 1 > codex/negative-control.log 2>&1
rc=$?
set -e
if [[ $rc != 1 ]] || ! grep -q 'FAIL' codex/negative-control.log; then
    echo "FAIL: negative control (exit $rc)"
    exit 2
fi
printf 'Negative control passed.\n'
exit "$positive_rc"
