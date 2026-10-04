#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.2
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
[[ $(git branch --show-current) = topic/host-bundle ]]
[[ -z $(git status --porcelain --untracked-files=no | sed '/ codex\//d') ]]
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR" codex/logs
cccl=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
[[ -f $cccl/CMakeLists.txt ]]
dest=codex/frozen/host-bundle
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$cccl" -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
cmake -B build-cuda-host "${flags[@]}" > codex/logs/cuda-configure.log 2>&1
[[ ${CONFIGURE_ONLY:-0} = 1 ]] && exit 0
cmake --build build-cuda-host --target llama-server -j8 > codex/logs/cuda-build.log 2>&1
[[ ! -e $dest ]]
mkdir -p "$dest/bin"
cp build-cuda-host/bin/llama-server "$dest/bin/round-trace"
for path in build-cuda-host/bin/*.so*; do
    cp -L "$path" "$dest/bin/$(basename "$path")"
done
{ git rev-parse HEAD; echo 'CUDA 13.2; CCCL v3.4.3 from local source; NCCL OFF; arch 120a; Release'; } > "$dest/SOURCE"
(cd "$dest/bin"; sha256sum round-trace *.so* > ../PRE-PATCHELF-SHA256SUMS)
for f in "$dest"/bin/round-trace "$dest"/bin/*.so*; do
    case $(basename "$f") in
        libggml-cuda.so*) patchelf --set-rpath '$ORIGIN:/usr/local/cuda-13.2/lib64' "$f" ;;
        *)               patchelf --set-rpath '$ORIGIN' "$f" ;;
    esac
done
(cd "$dest/bin"; sha256sum round-trace *.so* > SHA256SUMS)
(cd "$dest"; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
