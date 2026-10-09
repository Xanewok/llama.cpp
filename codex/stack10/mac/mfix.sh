# K4 np2 fix: Meta recreates every main subgraph after an arena reallocation (ggml-backend-meta.cpp). Codes 54-57.
# The pre-fix mutant is the old loop bound, patched in place and restored on exit; only build-asan is rebuilt with it.
MFILE=ggml/src/ggml-backend-meta.cpp
mutate() {
    [[ $(grep -c -F 'i < backend_ctx->max_subgraphs; i++' "$MFILE") == 1 ]]
    trap 'git checkout -- "$MFILE"' EXIT
    sed -i '' 's/i < backend_ctx->max_subgraphs; i++/i < n_subgraphs; i++/' "$MFILE"
    [[ $(grep -c -F 'i < backend_ctx->max_subgraphs; i++' "$MFILE") == 0 ]]
}
restore() {
    git checkout -- "$MFILE"
    trap - EXIT
    "$BW_BUILDGATE" cmake --build build-asan --target test-alloc bw-np2-driver -j"${BW_JOBS:-4}"
}
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=0
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
# fixed must pass; the pre-fix mutant must die with a use-after-free in both test variants
asan_unit() {
    "$BW_GATE" build-asan/bin/test-alloc > "$D/asan-fixed.txt" 2>&1
    grep -aq 'test_meta_subgraph_capacity(plan) PASSED' "$D/asan-fixed.txt"
    mutate
    "$BW_BUILDGATE" cmake --build build-asan --target test-alloc -j"${BW_JOBS:-4}"
    if "$BW_GATE" build-asan/bin/test-alloc > "$D/asan-mutant.txt" 2>&1; then
        echo "FAIL pre-fix mutant escaped test-alloc"; return 1
    fi
    grep -aq 'test_meta_subgraph_capacity(compute) ' "$D/asan-mutant.txt"
    grep -aq 'heap-use-after-free' "$D/asan-mutant.txt"
    echo "pre-fix mutant killed: $(grep -ac 'heap-use-after-free' "$D/asan-mutant.txt") use-after-free reports"
    restore
}
# np2 driver under Meta(CPU,CPU): fixed with K4 on must pass clean. The pre-fix arms (K4 on / K4 off) are recorded, not gated:
# they show whether this driver's graph order reaches a freed subgraph, and whether that depends on K4.
np2arm() {
    local name=$1
    shift
    local rc=0
    "$MGATE" env "$@" HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
        build-asan/bin/bw-np2-driver "$MODEL" "$D/np2/$name.json" 1 2 1729 64 > "$D/np2/$name.log" 2>&1 || rc=$?
    printf '%s\trc=%s\tuaf=%s\n' "$name" "$rc" "$(grep -ac 'heap-use-after-free' "$D/np2/$name.log")" | tee -a "$D/np2/arms.tsv"
    return "$rc"
}
asan_np2() {
    mkdir -p "$D/np2"
    : > "$D/np2/arms.tsv"
    np2arm fixed-on GGML_SSM_CONV_RS_LOG=0
    mutate
    "$BW_BUILDGATE" cmake --build build-asan --target bw-np2-driver -j"${BW_JOBS:-4}"
    np2arm prefix-on GGML_SSM_CONV_RS_LOG=0 || true
    np2arm prefix-off GGML_SSM_CONV_RS_OFF=1 || true
    restore
}
mfix_cheap() {
    stage 54 meta-subgraphs-unit unit
    stage 55 meta-subgraphs-asan-build asan_build
    stage 56 meta-subgraphs-asan-unit asan_unit
}
mfix_model() {
    stage 57 meta-subgraphs-asan-np2 asan_np2
}
