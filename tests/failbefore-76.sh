#!/usr/bin/env bash
set -euo pipefail
out=$1
mkdir -p "$out"
build=$PWD/build-failbefore-76-cpu
read -r -a extra <<< "${BW_CPU_EXTRA:-}"
"${BW_BUILDGATE:?}" cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF "${extra[@]}" > "$out/configure.log" 2>&1
"$BW_BUILDGATE" cmake --build "$build" -j"${BW_JOBS:-4}" --target test-alloc > "$out/build.log" 2>&1
"$BW_GATE" "$build/bin/test-alloc" --q8-capacity > "$out/capacity.log" 2>&1
grep -Fq '76 PASS oversized cache request rejected; exact-size request reused' "$out/capacity.log"
