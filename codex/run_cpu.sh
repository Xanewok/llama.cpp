#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="" TMPDIR="$ROOT/build-tmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
SOURCE=${SOURCE:-$ROOT/build-stack-src}
BUILD=${BUILD:-$SOURCE/build-cpu}
mkdir -p "$TMPDIR"
nice -n 19 g++ -std=c++17 -O2 -I"$SOURCE/include" -I"$SOURCE/src" -I"$SOURCE/ggml/include" \
    codex/test-draft-vocab.cpp "$BUILD/bin/libllama.so.0.5.0" "$BUILD/bin/libggml.so.0.25.3" "$BUILD/bin/libggml-base.so.0.25.3" \
    -Wl,-rpath,"$BUILD/bin" -o "$BUILD/bin/test-draft-vocab"
nice -n 19 "$BUILD/bin/test-sampling"
nice -n 19 "$BUILD/bin/test-llama-archs" --input-upload -s 42
nice -n 19 "$BUILD/bin/test-llama-archs" --tp-sampling -s 42
nice -n 19 "$BUILD/bin/test-sampling-prefilter" "$MODEL" --draft-vocab-only --check codex/subset-verify-seeds.txt
for mode in single tp; do
    nice -n 19 "$BUILD/bin/test-draft-vocab" "$MODEL" "$mode" 17 deferred
done
nice -n 19 g++ -std=c++17 -O2 -I"$SOURCE/common" -I"$SOURCE/include" -I"$SOURCE/ggml/include" \
    codex/test-stack-verify.cpp "$BUILD/bin/libllama-common.so.0.5.0" "$BUILD/bin/libllama.so.0.5.0" \
    "$BUILD/bin/libggml.so.0.25.3" "$BUILD/bin/libggml-cpu.so.0.25.3" "$BUILD/bin/libggml-base.so.0.25.3" \
    "$BUILD/common/libllama-common-base.a" -Wl,-rpath,"$BUILD/bin" -o "$BUILD/bin/verify-sampling"
nice -n 19 "$BUILD/bin/verify-sampling" "$MODEL" --decode
