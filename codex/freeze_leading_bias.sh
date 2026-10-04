#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available < 10 * 1024 * 1024 )); then
    printf 'Skip freeze: M MemAvailable=%s KiB, below required 10 GiB.\n' "$available"
    exit 0
fi
printf 'Freeze start: M MemAvailable=%s KiB.\n' "$available"
[[ ! -e codex/frozen/leading-bias ]] || { echo 'Freeze destination already exists' >&2; exit 1; }
mkdir -p build-cuda-leading-bias/tmp
export TMPDIR="$PWD/build-cuda-leading-bias/tmp" CUDA_CACHE_PATH="$PWD/build-cuda-leading-bias/tmp/cuda-cache" CCACHE_DIR="$PWD/build-cuda-leading-bias/ccache"
# A compiler build is allowed; CMake must not execute its CUDA device probe.
cmake -B build-cuda-leading-bias -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release > codex/cmake-leading-bias-cuda.log 2>&1
cmake --build build-cuda-leading-bias --target llama-server -j6 > codex/build-leading-bias-cuda.log 2>&1
mkdir -p codex/frozen/leading-bias/bin
cp build-cuda-leading-bias/bin/llama-server codex/frozen/leading-bias/bin/round-trace
for library in build-cuda-leading-bias/bin/*.so*; do
    cp -L "$library" "codex/frozen/leading-bias/bin/$(basename "$library")"
done
for binary in codex/frozen/leading-bias/bin/round-trace codex/frozen/leading-bias/bin/*.so*; do
    patchelf --set-rpath '$ORIGIN:/usr/local/cuda-13.2/lib64' "$binary"
done
git rev-parse HEAD > codex/frozen/leading-bias/SOURCE
printf 'CUDA 13.2; NCCL OFF; arch 120; Release; -j6; CUDA_VISIBLE_DEVICES empty; no GPU execution\n' >> codex/frozen/leading-bias/SOURCE
(cd codex/frozen/leading-bias/bin && sha256sum round-trace *.so* > SHA256SUMS)
(cd codex/frozen/leading-bias && sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
