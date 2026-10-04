#!/usr/bin/env bash
set -euo pipefail
if [[ ${HOST_DEPTH_NICED:-} != 1 ]]; then
    export HOST_DEPTH_NICED=1
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
mkdir -p build-tmp
export TMPDIR="$PWD/build-tmp"
export CUDA_VISIBLE_DEVICES="" CCACHE_DISABLE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target llama-common -j8
bash codex/build_fixture.sh
exec build-cpu/host-depth "$@"
