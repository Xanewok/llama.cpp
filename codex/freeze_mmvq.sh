#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR"
dest=${1:-codex/frozen/mmvq6}
[[ $dest == codex/frozen/* && ! -e $dest ]] || { echo "invalid or existing destination" >&2; exit 1; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked source is dirty" >&2; exit 1; }
available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ $available -ge 8388608 ]] || { echo "less than 8 GiB available RAM; retry later" >&2; exit 1; }
free -g > "$TMPDIR/freeze-memory.log"
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
    -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON > "$TMPDIR/cmake-cuda.log" 2>&1
# One compiler worker bounds peak compiler memory; the parent can raise this to six.
jobs=${FREEZE_JOBS:-1}
[[ $jobs =~ ^[1-6]$ ]] || { echo "FREEZE_JOBS must be 1..6" >&2; exit 1; }
cmake --build build-cuda --target llama-server bench-mmvq-shapes test-backend-ops -j "$jobs" > "$TMPDIR/build-cuda.log" 2>&1
[[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -ge 8388608 ]] || { echo "RAM floor crossed; inspect build log" >&2; exit 1; }
mkdir -p "$dest/bin"
cp build-cuda/bin/llama-server "$dest/bin/round-trace"
cp build-cuda/bin/bench-mmvq-shapes "$dest/bin/bench-mmvq-shapes"
cp build-cuda/bin/test-backend-ops "$dest/bin/cuda-op-check"
for path in build-cuda/bin/*.so*; do
    cp -L "$path" "$dest/bin/$(basename "$path")"
done
for path in "$dest"/bin/*; do
    patchelf --set-rpath '$ORIGIN' "$path"
done
{ git rev-parse HEAD; echo "CUDA 13.2; sm_120a (CMake maps 120); CCCL v3.4.3; NCCL OFF; Release"; } > "$dest/SOURCE"
(cd "$dest/bin"; sha256sum * > SHA256SUMS)
(cd "$dest"; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
