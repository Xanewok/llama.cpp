#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.2
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked tree is dirty" >&2; exit 1; }
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR"
dest=codex/frozen/stack3-rebase
# CMake must not run its CUDA device probe; no network: the prebuilt web UI download stays off.
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
cmake -B build-cuda "${flags[@]}" > build-tmp/cmake-cuda.log 2>&1
nice -n 19 cmake --build build-cuda --target llama-server -j"${FREEZE_JOBS:-8}" > build-tmp/build-cuda.log 2>&1
[[ ! -e $dest ]] || { echo "freeze destination already exists" >&2; exit 1; }
mkdir -p "$dest/bin"
cp build-cuda/bin/llama-server "$dest/bin/round-trace"
for path in build-cuda/bin/*.so*; do
    cp -L "$path" "$dest/bin/$(basename "$path")"
done
git rev-parse HEAD > "$dest/SOURCE"
(cd "$dest/bin"; sha256sum round-trace *.so* > ../PRE-PATCHELF-SHA256SUMS)
# The build RUNPATH names this build tree; the frozen copy must resolve only beside itself (and the CUDA runtime).
for f in "$dest"/bin/round-trace "$dest"/bin/*.so*; do
    case $(basename "$f") in
        libggml-cuda.so*) patchelf --set-rpath '$ORIGIN:/usr/local/cuda-13.2/lib64' "$f" ;;
        *)                patchelf --set-rpath '$ORIGIN' "$f" ;;
    esac
done
(cd "$dest/bin"; sha256sum round-trace *.so* > SHA256SUMS)
(cd "$dest"; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
