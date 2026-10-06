#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CCACHE_READONLY=1 CCACHE_NOSTATS=1 CCACHE_TEMPDIR="$PWD/build-tmp/ccache"
sha=$(git rev-parse HEAD)
short=$(git rev-parse --short HEAD)
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo 'commit tracked changes before the CUDA configure' >&2
    exit 1
fi
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
       -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
       -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
       -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
       -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
gate() { bash codex/host-gate.sh "$@"; }
buildgate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
gate "$buildgate" cmake -B build-cuda "${flags[@]}" > codex/host-results/configure-cuda.log 2>&1
gate "$buildgate" cmake --build build-cuda --target llama-server -j4 > codex/host-results/build-cuda.log 2>&1
if [[ $(git rev-parse HEAD) != "$sha" ]]; then
    echo 'HEAD changed during the CUDA build; reconfigure before freezing' >&2
    exit 1
fi
if ! strings build-cuda/bin/libggml-base.so | rg -x "$short" > codex/host-results/ggml-stamp.txt; then
    echo 'ggml stamp does not match the configured source' >&2
    exit 1
fi
cp build-cuda/bin/llama-server build-cuda/bin/round-trace
export QWEN38_WT_ROOT="$PWD/codex"
gate /home/xanewok/repos/qwen38-eval-worktrees/bin/wt freeze "$PWD" host-trims --build build-cuda --desc 'stack6 + indexed KV range removal + bulk MTP verify hidden copy; local engagement logs; CUDA 13.2, CCCL343, NCCL OFF' > codex/host-results/freeze.log 2>&1
{
    printf 'commit %s\n' "$sha"
    printf 'configure: cmake -B build-cuda '
    printf '%q ' "${flags[@]}"
    printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
    printf 'CUDA_VISIBLE_DEVICES=""; ggml stamp %s; no CUDA binary executed\n' "$short"
} > codex/frozen/host-trims/RECIPE.txt
test "$(head -1 codex/frozen/host-trims/SOURCE)" = "$sha"
test "$(strings codex/frozen/host-trims/build/bin/libggml-base.so | rg -x "$short")" = "$short"
printf 'PASS CUDA compile/freeze/stamp %s\n' "$sha"
