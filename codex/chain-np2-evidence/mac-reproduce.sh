#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-stack7/tools/local/bin:$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
cd "$HOME/qwen38-chain-np2/src"
OUT=$PWD/codex/chain-np2-evidence/baseline
mkdir -p "$OUT"
nice -n 19 cmake -S codex/stack7-test -B build-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
nice -n 19 cmake --build build-cpu --target bw-np2-driver chain-cpu -j10
cp -R build-cpu/bin "$OUT/bin"
chmod +x "$HOME/qwen38-chain-np2/mac-model-gate.sh"
for meta in 0 1; do
    for chain in 0 1; do
        : > "$OUT/$meta-$chain.trace.jsonl"
        date -u
        env LLAMA_MTP_CHAIN="$chain" LLAMA_MTP_FOLD_C=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
            LLAMA_ROUND_TRACE="$OUT/$meta-$chain.trace.jsonl" LLAMA_MTP_CHAIN_DUMP="$OUT/$meta-$chain.dump.txt" \
            "$HOME/qwen38-chain-np2/mac-model-gate.sh" "$OUT/bin/bw-np2-driver" "$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf" \
            "$OUT/$meta-$chain.json" "$meta" 2 1729 64 > "$OUT/$meta-$chain.log" 2>&1
        date -u
    done
done
printf 'PASS baseline runs completed\n'
