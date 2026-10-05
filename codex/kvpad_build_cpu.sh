#!/usr/bin/env bash
# CPU build + harness binaries for the n_kv_pad checks (nice 19, no GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/build-tmp"
mkdir -p build-tmp
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
nice -n 19 cmake -B build-cpu "${flags[@]}" > build-tmp/cmake-cpu.log 2>&1
nice -n 19 ~/repos/qwen38-eval-worktrees/bin/buildgate cmake --build build-cpu --target llama-server -j4 > build-tmp/build-cpu.log 2>&1
b=build-cpu
cp "$b/bin/llama-server" "$b/bin/cpu-serving"
nice -n 19 g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Ivendor -Itools/mtmd -I. codex/reuse-server-cpu.cpp "$b/tools/server/libserver-context.a" "$b/common/libllama-common-base.a" -L"$b/bin" -Wl,-rpath,"$PWD/$b/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$b/bin/reuse-server-cpu"
echo done
