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
c++ -O3 -UNDEBUG -std=c++17 -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -Ibuild-mask-cpu/common codex/host-depth.cpp -o build-mask-cpu/host-depth \
  -Wl,--start-group "${LLAMA_LIB:-build-mask-cpu/src/libllama.a}" build-mask-cpu/common/libllama-common.a build-mask-cpu/ggml/src/libggml.a build-mask-cpu/ggml/src/libggml-cpu.a build-mask-cpu/ggml/src/libggml-base.a build-mask-cpu/common/libllama-common-base.a build-mask-cpu/vendor/cpp-httplib/libcpp-httplib.a -Wl,--end-group -pthread -ldl -lm -lssl -lcrypto
