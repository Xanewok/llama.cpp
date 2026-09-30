#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" CCACHE_DISABLE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODEL=${1:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF > codex/cmake-cpu.log 2>&1
nice -n 19 cmake --build build-cpu --target llama-server test-sampling test-llama-archs -j8 > codex/build-cpu-final.log 2>&1
nice -n 19 c++ -std=c++17 -O2 -Iinclude -Icommon -Isrc -Iggml/include -Ivendor codex/test-draft-chain.cpp \
    -Lbuild-cpu/bin -Wl,-rpath,'$ORIGIN/../build-cpu/bin' -lllama-common -lllama -lggml -lggml-base -o codex/test-draft-chain
nice -n 19 codex/test-draft-chain "$MODEL" single "${SINGLE_SEEDS:-100}" > codex/cpu-single.log 2>&1
nice -n 19 codex/test-draft-chain "$MODEL" tp "${TP_SEEDS:-40}" > codex/cpu-tp.log 2>&1
nice -n 19 codex/test-draft-chain "$MODEL" single 2 wrong-pick > codex/cpu-wrong-pick.log 2>&1
nice -n 19 codex/test-draft-chain "$MODEL" tp 2 wrong-pick > codex/cpu-tp-wrong-pick.log 2>&1
if nice -n 19 codex/test-draft-chain "$MODEL" single 1 wrong-embedding > codex/cpu-wrong-embedding.log 2>&1; then
    echo 'Negative control unexpectedly passed' >&2
    exit 1
fi
nice -n 19 build-cpu/bin/test-sampling > codex/test-sampling.log 2>&1
nice -n 19 build-cpu/bin/test-llama-archs --tp-sampling -s 42 > codex/test-tp-sampling.log 2>&1
nice -n 19 build-cpu/bin/test-llama-archs --input-upload -s 42 > codex/test-input-upload.log 2>&1
nice -n 19 codex/test-draft-chain "$MODEL" single 2 normal 2 > codex/cpu-two-slots.log 2>&1
nice -n 19 codex/test-draft-chain "$MODEL" tp 2 normal 2 > codex/cpu-tp-two-slots.log 2>&1
nice -n 19 codex/test-draft-chain "$MODEL" tp 2 parallel 2 > codex/cpu-tp-parallel.log 2>&1
nice -n 19 cp build-cpu/bin/llama-server build-cpu/bin/chain-cpu
nice -n 19 python3 codex/cpu_server.py "$MODEL" > codex/cpu-server.log 2>&1
SLOTS=2 nice -n 19 python3 codex/cpu_server.py "$MODEL" > codex/cpu-server-two-slots.log 2>&1
