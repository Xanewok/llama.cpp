#!/usr/bin/env bash
# Mac CPU gate of embd-dev (STEP fix 2): CPU build, test-tok-rows (gathered rows vs whole-table GET_ROWS, bitwise, with
# row / nibble / f16 mutants), then model-level identity on Meta(CPU,CPU), where the device path engages (meta device):
# patch vs LLAMA_EMBD_ROWS_MUTANT=off (the old CPU GET_ROWS) must agree on every token and counter at np2 (fold) and np1
# (chain); the row mutant must change tokens; plain CPU must not engage.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
export S7_MODEL=${BW_MODEL:?} S7_GATE=$ROOT/codex/mac-model-gate.sh
unset LLAMA_MTP_CHAIN LLAMA_MTP_FOLD_C LLAMA_EMBD_ROWS_MUTANT LLAMA_ROUND_TRACE LLAMA_MTP_CHAIN_DUMP LLAMA_MTP_DRAFT_DUMP
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-embd-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
[[ -z $(git status --porcelain --untracked-files=no) ]]
git rev-parse HEAD > "$OUT/source-commit.txt"
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-tok-rows bw-np2-driver -j4
}
units() {
    "$BW_GATE" build-cpu/bin/test-tok-rows iq4_xs > "$OUT/tok-rows.txt" 2>&1
    grep -c 'test-tok-rows: PASS' "$OUT/tok-rows.txt" > /dev/null
    grep -c 'tok_rows: CPU iq4_xs bitwise=1' "$OUT/tok-rows.txt" > /dev/null
    grep -c 'tok_rows: CPU iq4_xs row mutant caught' "$OUT/tok-rows.txt" > /dev/null
    grep -c 'tok_rows: iq4_xs reassociated differing=0/' "$OUT/tok-rows.txt" > /dev/null
    grep -cE 'tok_rows: iq4_xs f16-intermediate differing=[1-9]' "$OUT/tok-rows.txt" > /dev/null
    grep -cE 'tok_rows: iq4_xs nibbles-swapped differing=[1-9]' "$OUT/tok-rows.txt" > /dev/null
}
# ARM META NP CHAIN [MUTANT]
boot() {
    local arm=$1 meta=$2 np=$3 chain=$4 mutant=${5:-}
    local -a env_args=(LLAMA_MTP_CHAIN="$chain" LLAMA_MTP_FOLD_C=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1)
    [[ -z $mutant ]] || env_args+=(LLAMA_EMBD_ROWS_MUTANT="$mutant")
    env "${env_args[@]}" "$S7_GATE" build-cpu/bin/bw-np2-driver "$S7_MODEL" "$OUT/model/$arm.json" "$meta" "$np" 1729 64 \
        > "$OUT/model/$arm.log" 2>&1
}
model() {
    mkdir -p "$OUT/model"
    boot np2-patch 1 2 0
    boot np2-off   1 2 0 off
    boot np2-row   1 2 0 row
    boot np1-chain-patch 1 1 1
    boot np1-chain-off   1 1 1 off
    boot np2-plain 0 2 0
    python3 codex/embd/check_model.py "$OUT/model"
}
stage 20 cpu-build build
stage 21 units units
stage 22 model-identity model
echo 'bw-embd-mac: PASS'
