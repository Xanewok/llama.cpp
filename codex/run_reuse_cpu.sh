#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build-tmp"
mkdir -p "$TMPDIR"
if [[ -d codex/local.git ]]; then export GIT_DIR="$PWD/codex/local.git"; fi
if [[ ! -d codex/base-src ]]; then
    mkdir -p codex/base-src
    git archive "$(cat codex/BASE)" | tar -x -C codex/base-src
fi
flags=(-DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF)
cmake -B build-cpu "${flags[@]}" > codex/cmake-cpu-safe.log 2>&1
cmake --build build-cpu --target llama-server test-llama-archs test-sampling -j8 > codex/build-cpu-safe.log 2>&1
cmake -S codex/base-src -B build-base-cpu "${flags[@]}" > codex/cmake-base-safe.log 2>&1
cmake --build build-base-cpu --target llama-server -j8 > codex/build-base-safe.log 2>&1
for build in build-base-cpu build-cpu; do
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Icommon -Iinclude -Iggml/include -Isrc -Ivendor -I. codex/reuse-cpu.cpp "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/reuse-cpu"
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Ivendor -Itools/mtmd -I. codex/reuse-server-cpu.cpp "$build/tools/server/libserver-context.a" "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/reuse-server-cpu"
done
g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Iggml/include -Isrc -I. codex/reuse-contract.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml -lggml-cpu -lggml-base -o build-cpu/bin/reuse-contract
g++ -std=c++17 -O2 -DGGML_SHARED -Iggml/include codex/prepare_cpu_target.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml-base -o build-cpu/bin/prepare-cpu-target
export MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
build-cpu/bin/prepare-cpu-target "$MODEL" build-tmp/reuse-untied-target.gguf > codex/prepare-target-safe.log 2>&1
build-cpu/bin/reuse-contract > codex/contracts-safe.log 2>&1
build-cpu/bin/test-llama-archs --tp-sampling > codex/tp-sampling-safe.log 2>&1
build-cpu/bin/test-sampling > codex/sampling-safe.log 2>&1
python3 codex/negative_reuse.py > codex/negative-safe.log 2>&1
python3 codex/run_reuse_cpu.py --seeds 16 --rounds 12 > codex/synthetic-safe.log 2>&1
python3 codex/run_reuse_cpu.py --real --seeds 4 --rounds 12 > codex/real-safe.log 2>&1
for k in 3 5; do
    for seed in 1729 2718; do
        for arm in up patch; do
            build=build-cpu
            [[ $arm != up ]] || build=build-base-cpu
            "$build/bin/reuse-server-cpu" "$MODEL" "codex/reuse-results/server-$k-$seed-$arm.bin" 0 "$k" "$seed" 48 > "codex/reuse-results/server-$k-$seed-$arm.log" 2>&1
        done
        cmp "codex/reuse-results/server-$k-$seed-up.bin" "codex/reuse-results/server-$k-$seed-patch.bin"
    done
done
python3 codex/summarize_reuse.py
