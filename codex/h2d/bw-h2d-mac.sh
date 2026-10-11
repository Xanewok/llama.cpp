#!/usr/bin/env bash
# Mac CPU gate of h2d-contig (STEP fix 3): CPU build + test-meta-input-stage (CUDA parts skip), then the input staging and
# scatter code of ggml-cuda.cu run against a host stand-in for the CUDA runtime (codex/h2d/cuda_stub.h), scenarios + mutants.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
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
    echo "bw-h2d-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-meta-input-stage test-alloc -j4
}
units() {
    "$BW_GATE" build-cpu/bin/test-meta-input-stage
    grep -c 'fake/staged: OK' "$OUT/units.log" > /dev/null
    grep -c 'cuda scatter per rank: SKIP' "$OUT/units.log" > /dev/null
    grep -c 'cuda scatter shared: SKIP' "$OUT/units.log" > /dev/null
    "$BW_GATE" build-cpu/bin/test-alloc
}
slice() {  # FILE OUTFILE START END [FROM TO]  (mutant: replace exactly one occurrence of FROM with TO)
    python3 - "$@" <<'PY_SLICE'
import sys
f, out, a, b = sys.argv[1:5]
src = open(f).read()
a = src.index(a.encode().decode('unicode_escape'))
b = src.index(b.encode().decode('unicode_escape'), a)
snippet = src[a:b]
if len(sys.argv) > 5:
    assert snippet.count(sys.argv[5]) == 1, 'mutant anchor must match exactly once'
    snippet = snippet.replace(sys.argv[5], sys.argv[6])
open(out, 'w').write(snippet)
PY_SLICE
}
CU=ggml/src/ggml-cuda/ggml-cuda.cu
CUH=ggml/src/ggml-cuda/input-scatter.cuh
CU_A='// The caller has selected ctx->device.\nstatic void ggml_cuda_input_stream_ensure'
CU_B='#endif\n\nstatic void ggml_backend_cuda_get_tensor_async'
CUH_A='// Entries per launch'
CUH_B='// max_size: the largest size'
sim_build() {  # NAME [cu|cuh FROM TO]
    local name=$1
    shift
    mkdir -p build-sim
    if [[ ${1:-} == cu ]]; then slice "$CU" "build-sim/$name.inc" "$CU_A" "$CU_B" "$2" "$3"; else slice "$CU" "build-sim/$name.inc" "$CU_A" "$CU_B"; fi
    if [[ ${1:-} == cuh ]]; then slice "$CUH" "build-sim/$name.scatter.inc" "$CUH_A" "$CUH_B" "$2" "$3"; else slice "$CUH" "build-sim/$name.scatter.inc" "$CUH_A" "$CUH_B"; fi
    nice -n 19 c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -Wall -Wno-unused-function \
        -Wno-unknown-pragmas -Iggml/include -Iggml/src -Icodex/h2d \
        "-DSTAGE_SNIPPET=\"$ROOT/build-sim/$name.inc\"" "-DSCATTER_SNIPPET=\"$ROOT/build-sim/$name.scatter.inc\"" \
        codex/h2d/stage_sim.cpp -o "build-sim/$name" -Lbuild-cpu/bin -lggml-base -Wl,-rpath,"$ROOT/build-cpu/bin"
}
sim_run() { ASAN_OPTIONS=detect_leaks=0 "$BW_GATE" "build-sim/$1" "$2"; }
SCATTER=(scatter-perrank-lazy scatter-perrank-random scatter-stage-lazy scatter-stage-random scatter-fallback)
cuda_sim() {
    sim_build stage
    local s
    for s in perrank declines ring-lazy ring-random "${SCATTER[@]}"; do
        sim_run stage "$s"
    done
    sim_run stage order-slow1 2>&1 | grep -ac 'slowest host link first: CUDA1 ' > /dev/null
    sim_run stage order-slow0 2>&1 | grep -ac 'slowest host link first: CUDA0 ' > /dev/null
    sim_run stage order-sym 2>&1 | grep -ac 'slowest host link first: CUDA0 ' > /dev/null
    sim_run stage order-unmeasured 2>&1 | grep -ac 'slowest host link first: CUDA0 ' > /dev/null
    GGML_CUDA_INPUT_SCATTER_MUTANT=1 sim_run stage scatter-perrank-lazy && { echo 'FAIL env mutant escaped'; return 1; }
    sim_run stage scatter-perrank-lazy 2>&1 | grep -ac 'input_scatter: engaged device=1 .* mutant=0' > /dev/null
}
escapes() {  # NAME SCENARIO...: the mutant build must fail every scenario given
    local name=$1 s
    shift
    for s in "$@"; do
        if sim_run "$name" "$s"; then echo "FAIL mutant $name escaped $s"; return 1; fi
    done
}
# Each mutant of the sliced code must make its scenarios fail.
cuda_sim_mutants() {
    sim_build nowait cu 'if (slot.pending) {
        for (size_t r = 0; r < n; ++r) {
            ggml_cuda_set_device(st->devices[r]);' 'if (false) {
        for (size_t r = 0; r < n; ++r) {
            ggml_cuda_set_device(st->devices[r]);'
    escapes nowait ring-lazy scatter-stage-lazy
    sim_build shifted cu 'ctx->input_srcs.push_back((char *) slot.data + st->offsets[i]);' 'ctx->input_srcs.push_back((char *) slot.data + st->offsets[i] + (i == 1 ? 1 : 0));'
    escapes shifted ring-lazy scatter-stage-lazy
    sim_build offs cu 'args.offs[args.n] = (size_t) (src - base);' 'args.offs[args.n] = (size_t) (src - base) + (i == 1 ? 16 : 0);'
    escapes offs scatter-perrank-lazy scatter-stage-random
    sim_build lastchunk cu 'if (++args.n == GGML_CUDA_INPUT_SCATTER_MAX || i + 1 == n) {' 'if (++args.n == GGML_CUDA_INPUT_SCATTER_MAX) {'
    escapes lastchunk scatter-perrank-lazy scatter-stage-lazy
    sim_build nosync cu 'CUDA_CHECK(cudaStreamSynchronize(ctx->input_stream));
        CUDA_CHECK(cudaFree(ctx->input_scatter_buf));' 'CUDA_CHECK(cudaFree(ctx->input_scatter_buf));'
    escapes nosync scatter-perrank-lazy
    sim_build halfcopy cu 'cudaMemcpyAsync(ctx->input_scatter_buf, base, span,' 'cudaMemcpyAsync(ctx->input_scatter_buf, base, span / 2,'
    escapes halfcopy scatter-perrank-random scatter-stage-lazy
    sim_build unordered cu 'if (!overlap && n > 0 && ggml_cuda_input_scatter_reserve' 'if (n > 0 && ggml_cuda_input_scatter_reserve'
    escapes unordered ring-lazy perrank
    sim_build wide cuh 'if (mis % 16 == 0) {' 'if (mis % 8 == 0) {'
    escapes wide scatter-perrank-lazy
    sim_build bytetail cuh 'for (size_t i = tid; i < size; i += nthreads) {' 'for (size_t i = tid; i + 1 < size; i += nthreads) {'
    escapes bytetail scatter-perrank-lazy scatter-stage-lazy
}

stage 20 cpu-build build
stage 21 units units
stage 23 cuda-sim cuda_sim
stage 24 cuda-sim-mutants cuda_sim_mutants
echo 'bw-h2d-mac: PASS'
