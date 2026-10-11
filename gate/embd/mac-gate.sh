#!/usr/bin/env bash
# Mac CPU gate (run on mini-builder from the repo root with the tree rsynced to $SRC): build, units, model identity.
set -u
export PATH=$HOME/qwen38-tools/bin:$PATH CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
SRC=${SRC:-$PWD}; B=${B:-$SRC/build}; MODEL=${MODEL:-$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
cd "$SRC"
step=${1:?build|units|model}
case $step in
build)
    cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF \
        -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON || exit 1
    nice -n 19 cmake --build "$B" --target test-tok-rows test-backend-ops llama -j4 || exit 1
    nice -n 19 clang++ -std=c++17 -O1 -Iinclude -Iggml/include gate/embd/driver.cpp -o "$B/bin/embd-driver" \
        -L"$B/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin" || exit 1
    ;;
units)
    nice -n 19 "$B/bin/test-tok-rows"; echo "test-tok-rows rc=$?"
    nice -n 19 "$B/bin/test-backend-ops" test -o GET_ROWS -b CPU 2>&1 | tail -4; echo "test-backend-ops GET_ROWS rc=${PIPESTATUS[0]}"
    ;;
model)
    run() { # name meta [mutant]
        local out=$B/model-$1; shift
        env ${2:+LLAMA_EMBD_ROWS_MUTANT=$2} nice -n 19 "$B/bin/embd-driver" "$MODEL" "$1" 24 > "$out.txt" 2> "$out.log"
        local rc=$?
        echo "$(basename "$out") rc=$rc engaged=$(grep -ac 'embd_rows: engaged' "$out.log")"
    }
    run meta-patch 1; run meta-off 1 off; run meta-row 1 row; run plain 0
    cmp "$B/model-meta-patch.txt" "$B/model-meta-off.txt" && echo "IDENT patch==off"
    cmp "$B/model-meta-patch.txt" "$B/model-plain.txt" && echo "IDENT meta==plain"
    cmp -s "$B/model-meta-patch.txt" "$B/model-meta-row.txt" || echo "MUTANT row differs"
    cat "$B/model-meta-patch.txt"
    ;;
esac
