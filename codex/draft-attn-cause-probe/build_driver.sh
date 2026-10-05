#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/../.."
python3 codex/draft-attn-cause-probe/prepare_driver.py
mkdir -p codex/draft-attn-cause-probe/bin
c++ -O3 -DN_THREADS=8 -std=c++17 -Iggml/include build-tmp/draft-attn-cause/driver.cpp \
    -L/home/xanewok/repos/draft-attn/codex/frozen/draft-attn-base/bin -lggml -lggml-base -lggml-cpu -pthread \
    -o codex/draft-attn-cause-probe/bin/cause-driver
sha256sum codex/draft-attn-cause-probe/bin/cause-driver > codex/draft-attn-cause-probe/DRIVER-SHA256SUMS
