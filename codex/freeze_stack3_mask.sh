#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR"
# CMake must not run its CUDA device probe.
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
cmake -B build-cuda "${flags[@]}" > codex/cmake-cuda.log 2>&1
cmake --build build-cuda --target llama-server -j"${FREEZE_JOBS:-8}" > codex/build-cuda.log 2>&1
[[ ! -e codex/frozen/stack3-mask ]] || { echo "freeze destination already exists" >&2; exit 1; }
mkdir -p codex/frozen/stack3-mask/bin
cp build-cuda/bin/llama-server codex/frozen/stack3-mask/bin/round-trace
for path in build-cuda/bin/*.so*; do
    cp -L "$path" "codex/frozen/stack3-mask/bin/$(basename "$path")"
done
git rev-parse HEAD > codex/frozen/stack3-mask/SOURCE
(cd codex/frozen/stack3-mask/bin; sha256sum round-trace *.so* > SHA256SUMS)
(cd codex/frozen/stack3-mask; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
