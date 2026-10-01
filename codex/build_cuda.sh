#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$PWD/build-cuda/tmp"
mkdir -p "$TMPDIR" codex
nice -n 19 cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CCACHE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF > codex/configure-cuda.log 2>&1
nice -n 19 cmake --build build-cuda --target test-backend-ops -j8 > codex/build-cuda.log 2>&1
sha256sum tests/test-backend-ops.cpp ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/fattn-common.cuh ggml/src/ggml-cuda/fattn-mma-f16.cuh ggml/src/ggml-cuda/fattn-vec.cuh ggml/src/ggml-cuda/mma.cuh ggml/src/ggml-cuda/convert.cu build-cuda/bin/test-backend-ops build-cuda/bin/lib*.so* > codex/cuda-build.sha256
