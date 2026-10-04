#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.2
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR"
dest=codex/frozen/mask-np2
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
       -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
       -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3
       -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 -DFETCHCONTENT_FULLY_DISCONNECTED=ON)
cmake -B build-cuda-cccl343 "${flags[@]}" > build-tmp/cmake-cuda-cccl343.log 2>&1
[[ ${CONFIGURE_ONLY:-0} = 1 ]] && exit 0
nice -n 19 cmake --build build-cuda-cccl343 --target llama-server -j8 > build-tmp/build-cuda-cccl343.log 2>&1
[[ ! -e $dest ]] || { echo "freeze destination already exists" >&2; exit 1; }
mkdir -p "$dest/bin"
cp build-cuda-cccl343/bin/llama-server "$dest/bin/mask-np2-srv"
for path in build-cuda-cccl343/bin/*.so*; do
    cp -L "$path" "$dest/bin/$(basename "$path")"
done
{ git rev-parse HEAD; echo "CUDA 13.2; arch 120a; NCCL off; CCCL v3.4.3 from local source; compile only"; } > "$dest/SOURCE"
for path in "$dest"/bin/mask-np2-srv "$dest"/bin/*.so*; do
    case $(basename "$path") in
        libggml-cuda.so*) patchelf --set-rpath '$ORIGIN:/usr/local/cuda-13.2/lib64' "$path" ;;
        *)                patchelf --set-rpath '$ORIGIN' "$path" ;;
    esac
done
(cd "$dest/bin"; sha256sum mask-np2-srv *.so* > SHA256SUMS)
(cd "$dest"; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
