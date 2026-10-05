#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export CCACHE_READONLY=true CCACHE_TEMPDIR="$PWD/build-cuda/ccache-tmp"
mkdir -p "$CCACHE_TEMPDIR"
gate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
"$gate" cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
    -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER_FORCED=ON \
    -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
    -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON > codex/configure-cuda-head.log 2>&1
"$gate" cmake --build build-cuda --target llama-server test-backend-ops -j4 > codex/build-cuda-head.log 2>&1
"$gate" c++ -std=c++17 -O2 -Iggml/include -Iggml/src codex/conv_state_bench.cpp \
    -Lbuild-cuda/bin -Wl,-rpath,'$ORIGIN' -lggml -lggml-base \
    -o build-cuda/bin/conv-state-bench > codex/build-bench-cuda.log 2>&1
cp build-cuda/bin/llama-server build-cuda/bin/round-trace
QWEN38_WT_ROOT="$PWD/codex" /home/xanewok/repos/qwen38-eval-worktrees/bin/wt freeze "$PWD" gdn-conv-v2-proof \
    --build "$PWD/build-cuda" --desc 'V2 snapshot fusion after concat; small-row concat; CUDA 13.2; NCCL OFF; arch 120a; local CCCL 3.4.3' > codex/freeze-v2.log
# Extend the new freeze before anything pins it; the benchmark resolves only its frozen libraries.
cp build-cuda/bin/conv-state-bench codex/frozen/gdn-conv-v2-proof/build/bin/
(cd codex/frozen/gdn-conv-v2-proof/build/bin && sha256sum $(find . -maxdepth 1 -type f -o -type l | sed 's#^./##' | rg -v '^SHA256SUMS$' | LC_ALL=C sort) > SHA256SUMS)
cp codex/configure-cuda-head.log codex/frozen/gdn-conv-v2-proof/RECIPE.txt
