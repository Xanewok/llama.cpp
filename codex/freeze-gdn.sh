#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
export CCACHE_READONLY=true CCACHE_TEMPDIR="$PWD/build-tmp/ccache"
mkdir -p "$CCACHE_TEMPDIR"
gate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
nice -n 19 "$gate" cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
nice -n 19 "$gate" cmake --build build-cuda --target llama-server test-backend-ops -j4
cp build-cuda/bin/llama-server build-cuda/bin/round-trace
if [[ ${BUILD_ONLY:-0} == 1 ]]; then exit 0; fi
QWEN38_WT_ROOT="$PWD/codex" nice -n 19 /home/xanewok/repos/qwen38-eval-worktrees/bin/wt freeze "$PWD" gdn-gather-inplace --build "$PWD/build-cuda" --desc 'CUDA 13.2, SM120a, CCCL343, NCCL OFF; indexed GDN rollback state'
cp codex/freeze-gdn.sh codex/frozen/gdn-gather-inplace/RECIPE.txt
