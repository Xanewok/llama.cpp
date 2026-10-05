#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.2
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
[[ $(git branch --show-current) = topic/mmvq-pertype ]]
[[ -z $(git status --porcelain --untracked-files=no | sed '/ codex\//d') ]]
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR" codex/logs-pertype
cccl=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
[[ -f $cccl/CMakeLists.txt ]]
dest=codex/frozen/mmvq-pertype
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$cccl" -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
while (( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) < 10485760 )); do sleep 10; done
cmake -B build-cuda-pertype "${flags[@]}" > codex/logs-pertype/cuda-configure.log 2>&1
[[ ${CONFIGURE_ONLY:-0} = 1 ]] && exit 0
while (( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) < 10485760 )); do sleep 10; done
/usr/bin/time -f 'M: full_build_elapsed_s=%e' cmake --build build-cuda-pertype --target llama-server test-backend-ops -j6 > codex/logs-pertype/cuda-build.log 2>&1
[[ ! -e $dest ]]
mkdir -p "$dest/bin"
cp build-cuda-pertype/bin/llama-server "$dest/bin/round-trace"
cp build-cuda-pertype/bin/test-backend-ops "$dest/bin/"
for path in build-cuda-pertype/bin/*.so*; do
    cp -L "$path" "$dest/bin/$(basename "$path")"
done
{ git rev-parse HEAD; echo 'CUDA 13.2; CCCL v3.4.3 from local source; NCCL OFF; arch 120a; Release'; } > "$dest/SOURCE"
(cd "$dest/bin"; sha256sum round-trace test-backend-ops *.so* > ../PRE-PATCHELF-SHA256SUMS)
for f in "$dest"/bin/round-trace "$dest"/bin/test-backend-ops "$dest"/bin/*.so*; do
    case $(basename "$f") in
        libggml-cuda.so*) patchelf --set-rpath '$ORIGIN:/usr/local/cuda-13.2/lib64' "$f" ;;
        *)               patchelf --set-rpath '$ORIGIN' "$f" ;;
    esac
done
(cd "$dest/bin"; sha256sum round-trace test-backend-ops *.so* > SHA256SUMS)
(cd "$dest"; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
