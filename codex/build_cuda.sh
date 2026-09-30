#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" CCACHE_DISABLE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# CMake 3.28 runs its CUDA ABI binary for native architecture discovery even when an architecture is specified.
# Skip that compiler probe; the full target build still checks compilation and linking.
nice -n 19 cmake -B build-cuda-no-probe -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
    -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER_FORCED=ON \
    -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_OPENSSL=OFF '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' > codex/cmake-cuda-no-probe.log 2>&1
nice -n 19 cmake --build build-cuda-no-probe --target llama-server -j8 > codex/build-cuda-no-probe.log 2>&1
