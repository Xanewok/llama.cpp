#!/usr/bin/env bash
set -euo pipefail
if [[ ${HOST_DEPTH_NICED:-} != 1 ]]; then
    export HOST_DEPTH_NICED=1
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
mkdir -p build-tmp
export TMPDIR="$PWD/build-tmp"
export CUDA_VISIBLE_DEVICES=""
c++ -O3 -DNDEBUG -std=c++17 -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -Ibuild-cpu/common codex/host-depth.cpp -o build-cpu/host-depth \
  -Wl,--start-group "${LLAMA_LIB:-build-cpu/src/libllama.a}" build-cpu/common/libllama-common.a build-cpu/ggml/src/libggml.a build-cpu/ggml/src/libggml-cpu.a build-cpu/ggml/src/libggml-base.a build-cpu/common/libllama-common-base.a build-cpu/vendor/cpp-httplib/libcpp-httplib.a -Wl,--end-group -pthread -ldl -lm -lssl -lcrypto
