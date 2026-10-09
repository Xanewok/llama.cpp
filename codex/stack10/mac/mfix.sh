# K4 np2 fix: Meta recreates every main subgraph after an arena reallocation (ggml-backend-meta.cpp). Codes 54-57.
# Pre-fix control: GGML_META_SUBGRAPH_MUTANT=1 (local switch) restores the old loop bound in the same binary.
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=0
unset GGML_META_SUBGRAPH_MUTANT
unit() {
    "$BW_GATE" build-cpu/bin/test-alloc > "$D/test-alloc.txt" 2>&1
    grep -aq 'test_meta_subgraph_capacity(compute) PASSED' "$D/test-alloc.txt"
    grep -aq 'test_meta_subgraph_capacity(plan) PASSED' "$D/test-alloc.txt"
}
asan_build() {
    local asan='-fsanitize=address -fno-omit-frame-pointer'
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-asan "${flags[@]}" "${extra[@]}" \
        -DCMAKE_C_FLAGS="$asan" -DCMAKE_CXX_FLAGS="$asan" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address \
        -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address
    "$BW_BUILDGATE" cmake --build build-asan --target test-alloc bw-np2-driver -j"${BW_JOBS:-4}"
}
# fixed must pass; the mutant must die in the server's crash site (memset in ggml_hash_set_reset under graph_compute_impl)
asan_unit() {
    "$BW_GATE" build-asan/bin/test-alloc > "$D/asan-fixed.txt" 2>&1
    grep -aq 'test_meta_subgraph_capacity(plan) PASSED' "$D/asan-fixed.txt"
    if "$BW_GATE" env GGML_META_SUBGRAPH_MUTANT=1 build-asan/bin/test-alloc > "$D/asan-mutant.txt" 2>&1; then
        echo "FAIL pre-fix mutant escaped test-alloc"; return 1
    fi
    grep -aq 'test_meta_subgraph_capacity(compute) ' "$D/asan-mutant.txt"
    grep -aq 'ERROR: AddressSanitizer' "$D/asan-mutant.txt"
    grep -aq 'in ggml_hash_set_reset' "$D/asan-mutant.txt"
    grep -aq 'in ggml_backend_meta_graph_compute_impl' "$D/asan-mutant.txt"
    echo "pre-fix mutant killed: $(grep -a -m1 'ERROR: AddressSanitizer' "$D/asan-mutant.txt")"
}
# np2 driver under Meta(CPU,CPU): fixed with K4 on must pass clean. The mutant arms (K4 on / K4 off) are recorded, not gated:
# they show whether this driver's graph order reaches a freed subgraph, and whether that depends on K4.
np2arm() {
    local name=$1 rc=0
    shift
    "$MGATE" env "$@" HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
        build-asan/bin/bw-np2-driver "$MODEL" "$D/np2/$name.json" 1 2 1729 64 > "$D/np2/$name.log" 2>&1 || rc=$?
    printf '%s\trc=%s\tasan=%s\tmeta_frame=%s\n' "$name" "$rc" "$(grep -ac 'ERROR: AddressSanitizer' "$D/np2/$name.log")" \
        "$(grep -ac 'in ggml_backend_meta_graph_compute_impl' "$D/np2/$name.log")" | tee -a "$D/np2/arms.tsv"
    return "$rc"
}
asan_np2() {
    mkdir -p "$D/np2"
    : > "$D/np2/arms.tsv"
    np2arm fixed-on GGML_SSM_CONV_RS_LOG=0
    np2arm mutant-on GGML_META_SUBGRAPH_MUTANT=1 || true
    np2arm mutant-off GGML_META_SUBGRAPH_MUTANT=1 GGML_SSM_CONV_RS_OFF=1 || true
}
mfix_cheap() {
    stage 54 meta-subgraphs-unit unit
    stage 55 meta-subgraphs-asan-build asan_build
    stage 56 meta-subgraphs-asan-unit asan_unit
}
mfix_model() {
    stage 57 meta-subgraphs-asan-np2 asan_np2
}
