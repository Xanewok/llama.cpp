#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${1:-$ROOT/build-cpu}
DEST=${2:-$ROOT/build-cpu/verify-probe}
GPU=${3:-0}
SOURCE=${PROBE_SOURCE:-$ROOT/codex/probe.cpp}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/build-cpu/tmp"
mkdir -p "$TMPDIR" "$(dirname "$DEST")"
DEFINES=()
[[ $GPU == 0 ]] || DEFINES=(-DVERIFY_PARENT_GPU)
exec nice -n 19 g++ -std=c++17 -O2 "${DEFINES[@]}" -I"$ROOT/common" -I"$ROOT/include" -I"$ROOT/src" \
    -I"$ROOT/ggml/include" -I"$ROOT/vendor" "$SOURCE" \
    "$BUILD/bin/libllama-server-impl.so" "$BUILD/bin/libllama-common.so" "$BUILD/bin/libllama.so" \
    "$BUILD/bin/libggml.so" "$BUILD/bin/libggml-base.so" \
    "$ROOT/build-cpu/common/libllama-common-base.a" \
    -Wl,-rpath,"$(dirname "$DEST")" -Wl,-rpath,"$BUILD/bin" -Wl,-rpath-link,"$BUILD/bin" -o "$DEST"
