#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
OUT=${1:?out}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
mkdir -p "$OUT"
export BW_MODEL=${BW_MODEL:-$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
# Configure and compile only on the Mac. No backend can use a GPU.
nice -n 19 cmake -S codex/stack7-test -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CCACHE=ON \
    -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$OUT/cmake.log" 2>&1
nice -n 19 cmake --build build-cpu --target bw-np2-driver -j4 > "$OUT/build.log" 2>&1
nice -n 19 python3 codex/mseq/matrix.py "$OUT/matrix" > "$OUT/matrix.log" 2>&1
