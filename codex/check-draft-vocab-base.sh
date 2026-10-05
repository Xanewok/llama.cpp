#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export DRAFT_VOCAB_TEST_THREADS=4 PYTHONDONTWRITEBYTECODE=1
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
VOCAB=/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json
if [[ ! -d codex/draft-base-src/src ]]; then
    mkdir -p codex/draft-base-src
    git archive 590912a1de2097523db6c8efe185387bb5c7d4fc | tar -x -C codex/draft-base-src
fi
cmake -S codex/draft-base-src -B build-draft-base-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF > codex/draft-base-cmake.log 2>&1
cmake --build build-draft-base-cpu --target llama -j4 > codex/draft-base-build.log 2>&1
for build in build-cpu build-draft-base-cpu; do
    g++ -std=c++17 -O2 -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Iggml/include -Isrc codex/test-draft-vocab.cpp -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama -lggml -lggml-base -o "$build/bin/test-draft-vocab"
    "$build/bin/test-draft-vocab" "$MODEL" cpu 3 identity "codex/$build-identity.bin" > "codex/$build-identity.log" 2>&1
done
cmp codex/build-cpu-identity.bin codex/build-draft-base-cpu-identity.bin
sha256sum codex/build-cpu-identity.bin codex/build-draft-base-cpu-identity.bin > codex/draft-vocab-base-identity-sha256.txt
build-cpu/bin/gguf-draft-vocab "$MODEL" "build-cpu/qwen4b-hq-$BASHPID.gguf" "$VOCAB" embed
build-cpu/bin/test-draft-vocab "$MODEL" cpu 5 gguf "build-cpu/qwen4b-hq-$BASHPID.gguf" > codex/draft-vocab-embedded.log 2>&1
