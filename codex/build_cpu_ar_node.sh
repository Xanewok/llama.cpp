#!/usr/bin/env bash
# CPU builds for the AllReduce-node spike: build-cpu (this tree) and a base build of ac7465885 whose binaries are copied to
# codex/base-cpu (the base build dir is deleted afterwards). Compiles go through buildgate; run `gpuq status --brief` first.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
gate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
BASE=ac7465885dc9f13de396ccd91b4be5eaffde3730
mkdir -p build-tmp codex/cpu-results
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF)
fixture_cmd() { # build_dir src_dir extra_rpath
    local build=$1 src=$2 rpath=$3
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -I"$src/tools/server" -I"$src/common" -I"$src/include" -I"$src/ggml/include" \
        -I"$src/vendor" -I"$src/tools/mtmd" -I"$src" "$PWD/codex/ar-node-server-cpu.cpp" \
        "$build/tools/server/libserver-context.a" "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$rpath" \
        -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/ar-node-server-cpu"
}

targets=(llama-server test-alloc test-sampling)
cmake -B build-cpu "${flags[@]}" > build-tmp/cmake-cpu.log 2>&1
"$gate" cmake --build build-cpu --target "${targets[@]}" -j4 > build-tmp/build-cpu.log 2>&1
fixture_cmd build-cpu "$PWD" "$PWD/build-cpu/bin"
cp build-cpu/bin/llama-server build-cpu/bin/cpu-serving

if [[ ! -x codex/base-cpu/ar-node-server-cpu ]]; then
    rm -rf codex/base-src; mkdir -p codex/base-src
    git archive "$BASE" | tar -x -C codex/base-src
    cmake -S codex/base-src -B build-base-cpu "${flags[@]}" > build-tmp/cmake-base-cpu.log 2>&1
    "$gate" cmake --build build-base-cpu --target llama-server -j4 > build-tmp/build-base-cpu.log 2>&1
    fixture_cmd build-base-cpu "$PWD/codex/base-src" '$ORIGIN'
    mkdir -p codex/base-cpu
    cp -a build-base-cpu/bin/*.so* build-base-cpu/bin/ar-node-server-cpu codex/base-cpu/
    rm -rf build-base-cpu
    rm -rf codex/base-src/.git
fi
echo "builds done"
