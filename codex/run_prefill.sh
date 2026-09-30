#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 TMPDIR="$PWD/build-tmp"
nice -n 19 c++ -O3 -std=c++17 -Icommon -Iinclude -Iggml/include -Isrc -Ivendor codex/prefill.cpp \
    -Wl,-rpath,"$PWD/build-cpu/bin" build-cpu/bin/libllama-common.so.0.5.0 build-cpu/bin/libllama.so.0.5.0 \
    build-cpu/bin/libggml.so.0.25.3 build-cpu/bin/libggml-cpu.so.0.25.3 build-cpu/bin/libggml-base.so.0.25.3 \
    build-cpu/common/libllama-common-base.a -o build-cpu/bin/verify-prefill
nice -n 19 build-cpu/bin/verify-prefill "${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}" "$@"
