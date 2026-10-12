# K6b residual add + RMS norm fusion (add-rms-norm.h matcher) and the ADD_RMS_NORM identity. Codes 65-69.
source "$LIBDIR/k6common.sh"
contracts() {
    host_build add-rms-norm-matcher codex/k6/add-rms-norm-matcher.cpp
    "$BIN/add-rms-norm-matcher"
}
contract_mutants() {
    local m=$D/mutants
    mutate ggml/src/ggml-cuda/add-rms-norm.h "$m/cap" 'GGML_CUDA_ADD_RMS_NORM_MAX_ROWS = 16' 'GGML_CUDA_ADD_RMS_NORM_MAX_ROWS = 17'
    host_build matcher-cap codex/k6/add-rms-norm-matcher.cpp "$m/cap"
    must_fail "$BIN/matcher-cap"
    mutate ggml/src/ggml-cuda/add-rms-norm.h "$m/norm" 'valid = valid && !overlaps(norm, t);' '(void) t;'
    host_build matcher-norm codex/k6/add-rms-norm-matcher.cpp "$m/norm"
    must_fail "$BIN/matcher-norm"
    mutate ggml/src/ggml-cuda/add-rms-norm.h "$m/resid" ' && !overlaps(mul, add);' ';'
    host_build matcher-resid codex/k6/add-rms-norm-matcher.cpp "$m/resid"
    must_fail "$BIN/matcher-resid"
}
add_rms_identity() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o RMS_NORM_MUL_ADD,ADD_RMS_NORM -j 1 > "$D/add-rms.out" 2>&1
    grep -aq 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS' "$D/add-rms.out"
}
k6b_cheap() {
    stage 65 add-rms-norm-matcher contracts
    stage 66 add-rms-norm-matcher-mutants contract_mutants
    stage 67 add-rms-norm-identity add_rms_identity
}
