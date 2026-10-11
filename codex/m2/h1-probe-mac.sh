#!/usr/bin/env bash
# bw-mac payload, H1/H2 probe for the chain-M2 post-DC host stretch: per-row draft sample time and cur_p size (H1ROW,
# H1STEP) and output_reorder cost (H1REORDER), np2 k5, with and without the ignore_eos EOG bias, per-stream and unified KV.
# Usage: h1-probe-mac.sh OUT. rc: 0 done, 75 RAM gate skip, 10 build.
set -uo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
OUT=${1:?out}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODEL=${BW_MODEL:-$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
GATE=$ROOT/codex/m2/model-gate.sh
# shellcheck disable=SC2086
cmake -S codex/stack7-test -B build-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_CCACHE=${BW_CCACHE:-ON} -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF \
    ${BW_CPU_EXTRA:--DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod} \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$OUT/cmake.log" 2>&1 || exit 10
nice -n 19 cmake --build build-cpu --target bw-np2-driver -j3 > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 10; }
echo "build ok"
run() { # tag chain [ENV=VAL ...]
    local tag=$1 chain=$2 rc=0
    shift 2
    (
        export LLAMA_MTP_CHAIN="$chain" NP2_N_MAX=5 NP2_THREADS=4 NP2_ALLOW_SHORT=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
            LLAMA_H1_PROBE=1 LLAMA_MTP_CHAIN_DUMP="$OUT/$tag.dump.txt" "$@"
        bash "$GATE" "$ROOT/build-cpu/bin/bw-np2-driver" "$MODEL" "$OUT/$tag.json" 0 2 1729 64
    ) > "$OUT/$tag.log" 2>&1 || rc=$?
    echo "$tag rc=$rc"
    [[ $rc == 75 ]] && exit 75
}
run on-eos   1 NP2_IGNORE_EOS=1
run on-noeos 1
run off-eos  0 NP2_IGNORE_EOS=1
run off-noeos 0
run on-eos-uni 1 NP2_IGNORE_EOS=1 NP2_KV_UNIFIED=1
echo "h1 probe done"
