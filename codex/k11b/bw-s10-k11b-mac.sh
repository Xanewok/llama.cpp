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
    "$BW_GATE" build-cpu/bin/test-meta-input-stage
    grep -c 'fake/staged: OK' "$OUT/units.log" > /dev/null
    "$BW_GATE" build-cpu/bin/test-alloc
}
# The meta glue mutant must break the staged contract (test aborts or fails).
glue_mutant() {
    grep -c 'GGML_META_INPUT_STAGE_MUTANT' ggml/src/ggml-backend-meta.cpp > /dev/null || { echo 'no glue mutant in this tree'; return 0; }
    if env GGML_META_INPUT_STAGE_MUTANT=1 "$BW_GATE" build-cpu/bin/test-meta-input-stage; then
        echo 'FAIL glue mutant escaped'; return 1
    fi
}
# The staging code of ggml-cuda.cu compiled against a host stand-in for the CUDA runtime (codex/cuda-sim/cuda_stub.h): streams are
# queues, so reusing a slot before every rank's upload event shows up as corrupted data. Mutants of the sliced code must fail.
slice_snippet() {  # OUTFILE [FROM TO]  (mutant: replace exactly one occurrence of FROM with TO)
    python3 - "$1" "${2:-}" "${3:-}" <<'PY_SLICE'
import sys
src = open('ggml/src/ggml-cuda/ggml-cuda.cu').read()
a = src.index('// The caller has selected ctx->device.\nstatic void ggml_cuda_input_stream_ensure')
b = src.index('#endif\n\nstatic void ggml_backend_cuda_get_tensor_async')
snippet = src[a:b]
if sys.argv[2]:
    assert snippet.count(sys.argv[2]) == 1, 'mutant anchor must match exactly once'
    snippet = snippet.replace(sys.argv[2], sys.argv[3])
open(sys.argv[1], 'w').write(snippet)
PY_SLICE
}
sim_build() {  # NAME [FROM TO]
    local name=$1
    shift
    mkdir -p build-sim
    slice_snippet "build-sim/$name.inc" "$@"
    nice -n 19 c++ -std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wno-unused-function -Iggml/include -Iggml/src -Icodex/cuda-sim \
        "-DSTAGE_SNIPPET=\"$ROOT/build-sim/$name.inc\"" codex/cuda-sim/stage_sim.cpp -o "build-sim/$name" \
        -Lbuild-cpu/bin -lggml-base -Wl,-rpath,"$ROOT/build-cpu/bin"
}
sim_run() { ASAN_OPTIONS=detect_leaks=0 "$BW_GATE" "build-sim/$1" "$2"; }
cuda_sim() {
    sim_build stage
    sim_run stage perrank
    sim_run stage declines
    sim_run stage ring-lazy
    sim_run stage ring-random
    sim_run stage order-slow1 2>&1 | grep -ac 'slowest host link first: CUDA1 ' > /dev/null
    sim_run stage order-slow0 2>&1 | grep -ac 'slowest host link first: CUDA0 ' > /dev/null
    sim_run stage order-sym 2>&1 | grep -ac 'slowest host link first: CUDA0 ' > /dev/null
    sim_run stage order-unmeasured 2>&1 | grep -ac 'slowest host link first: CUDA0 ' > /dev/null
}
# Each mutant must make its scenario fail.
cuda_sim_mutants() {
    sim_build nowait 'if (slot.pending) {
        for (size_t r = 0; r < n; ++r) {
            ggml_cuda_set_device(st->devices[r]);' 'if (false) {
        for (size_t r = 0; r < n; ++r) {
            ggml_cuda_set_device(st->devices[r]);'
    if sim_run nowait ring-lazy; then echo 'FAIL slot-wait mutant escaped'; return 1; fi
    sim_build rank0wait 'if (slot.pending) {
        for (size_t r = 0; r < n; ++r) {' 'if (slot.pending) {
        for (size_t r = 0; r < 1; ++r) {'
    if sim_run rank0wait ring-lazy; then echo 'FAIL rank-0-only wait mutant escaped'; return 1; fi
    sim_build ascending 'link_class(a) > link_class(b)' 'link_class(a) < link_class(b)'
    if sim_run ascending order-slow1 2>&1 | grep -ac 'slowest host link first: CUDA1 ' > /dev/null; then echo 'FAIL order mutant escaped'; return 1; fi
    sim_build shifted 'ctx->input_srcs.push_back((char *) slot.data + st->offsets[i]);' 'ctx->input_srcs.push_back((char *) slot.data + st->offsets[i] + (i == 1 ? 1 : 0));'
    if sim_run shifted ring-lazy; then echo 'FAIL shifted-source mutant escaped'; return 1; fi
}

stage 20 cpu-build build
stage 21 units units
stage 22 glue-mutant glue_mutant
stage 23 cuda-sim cuda_sim
stage 24 cuda-sim-mutants cuda_sim_mutants
echo 'bw-s10-k11b-mac: PASS'
