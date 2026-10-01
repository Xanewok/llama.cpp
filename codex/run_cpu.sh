#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$PWD/build-cpu/tmp"
mkdir -p "$TMPDIR"
MODEL=${MODEL:-/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
nice -n 19 cmake -S . -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DGGML_CPU_REPACK=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF
nice -n 19 cmake --build build-cpu -j8 --target llama-server test-sampling test-sampling-prefilter test-llama-archs
cp build-cpu/bin/llama-server build-cpu/bin/s4-cpu-srv
nice -n 19 g++ -std=c++17 -O2 -Icommon -Iinclude -Iggml/include -Ivendor codex/check.cpp \
    build-cpu/bin/libllama-common.so build-cpu/bin/libllama.so build-cpu/bin/libggml.so \
    build-cpu/bin/libggml-cpu.so build-cpu/bin/libggml-base.so build-cpu/common/libllama-common-base.a \
    -Wl,-rpath,"$PWD/build-cpu/bin" -o build-cpu/bin/s4-check
for mode in law decode tp; do
    args=()
    [[ $mode == law ]] || args=("$mode")
    nice -n 19 stdbuf -oL build-cpu/bin/s4-check "$MODEL" "${args[@]}" > "codex/$mode.log" 2>&1
done
status=0
S4_CORRUPT_Q=1 nice -n 19 build-cpu/bin/s4-check "$MODEL" > codex/negative-control.log 2>&1 || status=$?
[[ $status == 1 ]] && rg -q 'FAIL: law oracle failed' codex/negative-control.log
nice -n 19 build-cpu/bin/test-sampling > codex/test-sampling.log 2>&1
nice -n 19 build-cpu/bin/test-sampling-prefilter "$MODEL" > codex/prefilter.log 2>&1
nice -n 19 build-cpu/bin/test-llama-archs --tp-sampling -s 42 > codex/tp-sampling.log 2>&1
nice -n 19 g++ -std=c++17 -O2 -Iinclude -Isrc -Iggml/include codex/head.cpp \
    build-cpu/bin/libllama.so build-cpu/bin/libggml.so build-cpu/bin/libggml-base.so \
    -Wl,-rpath,"$PWD/build-cpu/bin" -o build-cpu/bin/s4-head
for rows in 4 5 6; do
    for mode in single tp; do
        S4_ROWS=$rows nice -n 19 build-cpu/bin/s4-head "$MODEL" "$mode" 3 deferred > "codex/head-$rows-$mode.log" 2>&1
    done
done
nice -n 19 python3 codex/server_cpu.py > codex/server-identity.log 2>&1
