#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-s10-k11b-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-meta-input-stage test-alloc -j4
}
units() {
    timeout 600 "$BW_GATE" build-cpu/bin/test-meta-input-stage
    grep -c 'fake/staged: OK' "$OUT/units.log" > /dev/null
    timeout 600 "$BW_GATE" build-cpu/bin/test-alloc
}
# The meta glue mutant must break the staged contract (test aborts or fails).
glue_mutant() {
    grep -c 'GGML_META_INPUT_STAGE_MUTANT' ggml/src/ggml-backend-meta.cpp > /dev/null || { echo 'no glue mutant in this tree'; return 0; }
    if env GGML_META_INPUT_STAGE_MUTANT=1 timeout 600 "$BW_GATE" build-cpu/bin/test-meta-input-stage; then
        echo 'FAIL glue mutant escaped'; return 1
    fi
}
stage 20 cpu-build build
stage 21 units units
stage 22 glue-mutant glue_mutant
echo 'bw-s10-k11b-mac: PASS'
