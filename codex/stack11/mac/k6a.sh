# K6a shared q8_1 activations (q8-cache.h) and the MUL_MAT_SHARED identity. Codes 60-64.
source "$LIBDIR/k6common.sh"
contracts() {
    host_build q8-cache-contract codex/k6/q8-cache-contract.cpp
    "$BIN/q8-cache-contract"
}
contract_mutants() {
    local m=$D/mutants
    mutate ggml/src/ggml-cuda/q8-cache.h "$m/stale" 'if (start <= src ? src - start < size : start - src < src_size) {' 'if (false) {'
    host_build q8-cache-stale codex/k6/q8-cache-contract.cpp "$m/stale"
    must_fail "$BIN/q8-cache-stale"
    mutate ggml/src/ggml-cuda/q8-cache.h "$m/bound" 'GGML_CUDA_Q8_REUSE_MAX_ROWS = 16' 'GGML_CUDA_Q8_REUSE_MAX_ROWS = 17'
    host_build q8-cache-bound codex/k6/q8-cache-contract.cpp "$m/bound"
    must_fail "$BIN/q8-cache-bound"
}
shared_identity() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_SHARED -j 1 > "$D/shared.out" 2>&1
    grep -aq 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS' "$D/shared.out"
}
k6a_cheap() {
    stage 60 q8-cache-contract contracts
    stage 61 q8-cache-contract-mutants contract_mutants
    stage 62 mul-mat-shared-identity shared_identity
}
