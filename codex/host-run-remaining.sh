#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CCACHE_READONLY=1 CCACHE_NOSTATS=1 CCACHE_TEMPDIR="$PWD/build-tmp/ccache"
gate() { bash codex/host-gate.sh "$@"; }
buildgate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
gate "$buildgate" g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Iggml/src -Ivendor -Itools/mtmd -I. codex/host-identity.cpp build-cpu/tools/server/libserver-context.a build-cpu/common/libllama-common-base.a -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o build-cpu/bin/host-identity > codex/host-results/build-identity.log 2>&1
gate python3 codex/host-cpu-identity.py > codex/host-results/cpu-identity.log 2>&1
gate python3 codex/host-draft-identity.py > codex/host-results/draft-identity.log 2>&1
gate python3 codex/host-input-count.py > codex/host-results/input-count.log 2>&1
gate python3 codex/host-attrib.py > codex/host-results/phase-attrib.log 2>&1
gate python3 codex/host-parser-check.py > codex/host-results/parser-check.log 2>&1
gate python3 codex/host-negative.py > codex/host-results/negative-controls.log 2>&1
