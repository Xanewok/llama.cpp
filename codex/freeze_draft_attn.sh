#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.2
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
[[ $(git branch --show-current) = topic/draft-attn-1row ]]
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR" codex/logs
cccl=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$cccl" -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
python3 codex/ram_build.py cmake -B build-cuda-draft-attn "${flags[@]}" > codex/logs/cuda-configure-draft-attn.log 2>&1
python3 codex/ram_build.py cmake --build build-cuda-draft-attn --target llama-server -j6 > codex/logs/cuda-build-draft-attn.log 2>&1
python3 codex/ram_build.py c++ -O3 -DNDEBUG -DN_THREADS=8 -std=c++17 -Iggml/include tests/test-backend-ops.cpp -Lbuild-cuda-draft-attn/bin -lggml -lggml-base -lggml-cpu -pthread -o build-cuda-draft-attn/bin/test-backend-ops > codex/logs/probe-build.log 2>&1
freeze() {
    local dest=$1 src=$2
    [[ ! -e $dest ]]
    mkdir -p "$dest/bin"
    cp "$src/round-trace" "$dest/bin/round-trace"
    cp build-cuda-draft-attn/bin/test-backend-ops "$dest/bin/"
    for path in "$src"/*.so*; do
        cp -L "$path" "$dest/bin/$(basename "$path")"
    done
    cp "$src/../SOURCE" "$dest/SOURCE"
    printf '%s\n' "probe source: $(git rev-parse HEAD); N_THREADS=8; shared driver for both arms" >> "$dest/SOURCE"
    (cd "$dest/bin"; sha256sum round-trace test-backend-ops *.so* > ../PRE-PATCHELF-SHA256SUMS)
    for f in "$dest"/bin/round-trace "$dest"/bin/test-backend-ops "$dest"/bin/*.so*; do
        case $(basename "$f") in
            libggml-cuda.so*) patchelf --set-rpath '$ORIGIN:/usr/local/cuda-13.2/lib64' "$f" ;;
            *)               patchelf --set-rpath '$ORIGIN' "$f" ;;
        esac
    done
    (cd "$dest/bin"; sha256sum round-trace test-backend-ops *.so* > SHA256SUMS)
    (cd "$dest"; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
}
cp build-cuda-draft-attn/bin/llama-server build-cuda-draft-attn/bin/round-trace
{ git rev-parse HEAD; printf '%s\n' 'CUDA 13.2; CCCL v3.4.3 from local source; NCCL OFF; arch 120a; Release'; printf '%s\n' "${flags[@]}"; } > build-cuda-draft-attn/SOURCE
freeze codex/frozen/draft-attn build-cuda-draft-attn/bin
base=/home/xanewok/repos/qwen38-weekend/frozen/stack3-rebase-cccl343
(cd "$base/build/bin"; sha256sum -c --quiet SHA256SUMS)
[[ $(head -1 "$base/SOURCE") = 98c695111df5b5517245b184ef4e818333cf5776 ]]
cp "$base/SOURCE" build-tmp/SOURCE
mkdir -p build-tmp/base-bin
cp "$base/build/bin/round-trace" build-tmp/base-bin/
for path in "$base/build/bin"/*.so*; do cp -L "$path" build-tmp/base-bin/; done
freeze codex/frozen/draft-attn-base build-tmp/base-bin
