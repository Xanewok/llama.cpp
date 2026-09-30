#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CUDA_VISIBLE_DEVICES="" TMPDIR="$ROOT/build-tmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$TMPDIR"
test "$(git -C "$ROOT/llama.cpp-stack2" rev-parse HEAD)" = 0dba27b9002b8347c6f14a9f4a1651b3478065ef
test -z "$(git -C "$ROOT/llama.cpp-stack2" status --porcelain --untracked-files=no)"
nice -n 19 cmake -S "$ROOT/llama.cpp-stack2" -B "$ROOT/build-cuda" \
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
    -DCMAKE_CUDA_COMPILER_FORCED=ON \
    -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
    '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' \
    -DGGML_CCACHE=OFF -DCMAKE_CUDA_COMPILER_LAUNCHER= -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
nice -n 19 cmake --build "$ROOT/build-cuda" -j8 --target llama-server
