#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CCACHE_READONLY=1 CCACHE_NOSTATS=1 CCACHE_TEMPDIR="$PWD/build-tmp/ccache"
gate() { bash codex/host-gate.sh "$@"; }
buildgate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
gate "$buildgate" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF > codex/host-results/reconfigure-cpu.log 2>&1
gate "$buildgate" cmake --build build-cpu --target llama-server test-kv-mask test-sampling -j4 > codex/host-results/rebuild-cpu.log 2>&1
gate "$buildgate" g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Ivendor -Itools/mtmd -I. codex/host-identity.cpp build-cpu/tools/server/libserver-context.a build-cpu/common/libllama-common-base.a -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o build-cpu/bin/host-identity > codex/host-results/build-identity.log 2>&1
gate "$buildgate" g++ -std=c++17 -O2 -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Iggml/include -Isrc -I. codex/kv-rm-bench.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml -lggml-cpu -lggml-base -o build-cpu/bin/kv-rm-bench > codex/host-results/build-bench.log 2>&1
gate build-cpu/bin/test-kv-mask > codex/host-results/test-kv-mask.log 2>&1
gate build-cpu/bin/test-sampling > codex/host-results/test-sampling.log 2>&1
LD_LIBRARY_PATH="$PWD/codex/base-cpu" gate build-cpu/bin/kv-rm-bench > codex/host-results/bench-base.log 2>&1
gate build-cpu/bin/kv-rm-bench > codex/host-results/bench-patch.log 2>&1
gate python3 codex/host-cpu-identity.py > codex/host-results/cpu-identity.log 2>&1
gate python3 codex/host-attrib.py > codex/host-results/phase-attrib.log 2>&1
gate python3 codex/host-parser-check.py > codex/host-results/parser-check.log 2>&1
gate python3 codex/host-negative.py > codex/host-results/negative-controls.log 2>&1
