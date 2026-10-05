#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/../.."
python3 codex/q8-probe3/prepare_driver.py
mkdir -p codex/q8-probe3/bin
c++ -O3 -DNDEBUG -DN_THREADS=8 -std=c++17 -Iggml/include build-tmp/q8-probe3/driver.cpp \
    -L/home/xanewok/repos/q8-1row/codex/frozen/q8-1row-base/bin -lggml -lggml-base -lggml-cpu -pthread \
    -o codex/q8-probe3/bin/q8-probe-driver
sha256sum codex/q8-probe3/bin/q8-probe-driver > codex/q8-probe3/DRIVER-SHA256SUMS
