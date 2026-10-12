# K11b mirrored host-input staging. The staging code of ggml-cuda.cu is compiled against a host stand-in for the CUDA runtime
# (codex/k11b/cuda-sim/cuda_stub.h): streams are queues, so reusing a slot before every rank's upload event corrupts data.
# Codes 120-129.
SIM=$D/sim
meta_stage_unit() {
    "$BW_GATE" build-cpu/bin/test-meta-input-stage | tee "$D/meta-input-stage.out"
    grep -ac 'fake/staged: OK' "$D/meta-input-stage.out" > /dev/null
}
# The meta glue mutant must break the staged contract (test aborts or fails).
glue_mutant() {
    [[ $(grep -c 'GGML_META_INPUT_STAGE_MUTANT' ggml/src/ggml-backend-meta.cpp) -ge 1 ]] || { echo 'FAIL no glue mutant in this tree'; return 1; }
    if env GGML_META_INPUT_STAGE_MUTANT=1 "$BW_GATE" build-cpu/bin/test-meta-input-stage; then
        echo 'FAIL glue mutant escaped'; return 1
    fi
    echo 'glue mutant killed'
}
# slice_snippet OUTFILE [FROM TO]: the staging code of ggml-cuda.cu; with FROM/TO, replace exactly one occurrence (a moved anchor fails).
slice_snippet() {
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
    mkdir -p "$SIM"
    slice_snippet "$SIM/$name.inc" "$@"
    "$BW_BUILDGATE" c++ -std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wno-unused-function -Iggml/include -Iggml/src -Icodex/k11b/cuda-sim \
        "-DSTAGE_SNIPPET=\"$SIM/$name.inc\"" codex/k11b/cuda-sim/stage_sim.cpp -o "$SIM/$name" \
        -Lbuild-cpu/bin -lggml-base -Wl,-rpath,"$ROOT/build-cpu/bin"
}
sim_run() { ASAN_OPTIONS=detect_leaks=0 "$BW_GATE" "$SIM/$1" "$2"; }
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
    echo 'slot-wait mutant killed'
    sim_build rank0wait 'if (slot.pending) {
        for (size_t r = 0; r < n; ++r) {' 'if (slot.pending) {
        for (size_t r = 0; r < 1; ++r) {'
    if sim_run rank0wait ring-lazy; then echo 'FAIL rank-0-only wait mutant escaped'; return 1; fi
    echo 'rank-0-only wait mutant killed'
    sim_build ascending 'link_class(a) > link_class(b)' 'link_class(a) < link_class(b)'
    if sim_run ascending order-slow1 2>&1 | grep -ac 'slowest host link first: CUDA1 ' > /dev/null; then echo 'FAIL order mutant escaped'; return 1; fi
    echo 'order mutant killed'
    sim_build shifted 'ctx->input_srcs.push_back((char *) slot.data + st->offsets[i]);' 'ctx->input_srcs.push_back((char *) slot.data + st->offsets[i] + (i == 1 ? 1 : 0));'
    if sim_run shifted ring-lazy; then echo 'FAIL shifted-source mutant escaped'; return 1; fi
    echo 'shifted-source mutant killed'
}
k11b_cheap() {
    stage 120 meta-input-stage-unit meta_stage_unit
    stage 121 meta-glue-mutant glue_mutant
    stage 122 cuda-sim cuda_sim
    stage 123 cuda-sim-mutants cuda_sim_mutants
}
