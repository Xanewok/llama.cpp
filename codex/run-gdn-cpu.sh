#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$(dirname "$0")/.."
export PATH="$PWD/codex/gate-bin:$PATH" CCACHE_READONLY=true CCACHE_TEMPDIR="$PWD/build-tmp/ccache"
mkdir -p "$CCACHE_TEMPDIR" codex/cpu-results
gate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
nice -n 19 "$gate" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON
nice -n 19 "$gate" cmake --build build-cpu --target llama test-backend-ops -j4
nice -n 19 "$gate" g++ -std=c++17 -O2 -g -rdynamic -pthread -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Iggml/include -Isrc -I. codex/gdn-cpu.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml -lggml-cpu -lggml-base -o build-cpu/bin/gdn-cpu
nice -n 19 "$gate" g++ -std=c++17 -O2 -DGGML_SHARED -Iggml/include codex/gdn-op-cpu.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml -lggml-cpu -lggml-base -o build-cpu/bin/gdn-op-cpu
nice -n 19 "$gate" g++ -shared -fPIC codex/backend-thread-cap.cpp -o build-cpu/bin/backend-thread-cap.so
nice -n 19 /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate build-cpu/bin/gdn-op-cpu
nice -n 19 bash codex/run-backend-cpu.sh
ulimit -c 0
nice -n 19 python3 codex/run-cpu.py
