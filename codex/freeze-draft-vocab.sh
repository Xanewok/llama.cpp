#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked tree is dirty" >&2; exit 1; }
mkdir -p build-tmp
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > codex/draft-vocab-cmake-cuda.log 2>&1
cmake --build build-cuda --target llama-server -j4 > codex/draft-vocab-build-cuda.log 2>&1
QWEN38_WT_ROOT="$PWD/codex" /home/xanewok/repos/qwen38-eval-worktrees/bin/wt freeze "$PWD" draft-vocab --build "$PWD/build-cuda" --desc "GGUF-carried reduced MTP vocabulary; CUDA 13.2, CCCL v3.4.3, NCCL off, requested arch 120 (CMake selects 120a); built only, never run"
cp codex/frozen/draft-vocab/build/bin/llama-server codex/frozen/draft-vocab/build/bin/round-trace
(cd codex/frozen/draft-vocab/build/bin; sha256sum llama-server round-trace *.so* > SHA256SUMS)
