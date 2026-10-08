# K6 shared q8 activations + residual add/RMS norm fusion. Host contracts compile the real CUDA headers against the CPU toolchain.
# Codes 60-69.
CXXF=(-std=c++17 -O1 -Iggml/include -Iggml/src -Iggml/src/ggml-cuda)
LIBS=(-Lbuild-cpu/bin "-Wl,-rpath,$ROOT/build-cpu/bin" -lggml-base)
BIN=$D/bin
# host_build NAME SOURCE [INCLUDE_OVERRIDE]
host_build() {
    local include=()
    [[ -z ${3:-} ]] || include=("-I$3")
    mkdir -p "$BIN"
    "$BW_BUILDGATE" c++ ${include[@]+"${include[@]}"} "${CXXF[@]}" -Icodex/k6 "$2" "${LIBS[@]}" -o "$BIN/$1"
}
# The pattern must match exactly one line of the real header and change the copy, so a moved anchor cannot pass silently.
mutate() {
    local header=$1 dir=$2 pattern=$3 replacement=$4
    mkdir -p "$dir"
    [[ $(grep -cF -- "$pattern" "$header") == 1 ]] || { echo "FAIL anchor does not match exactly once in $header: $pattern"; return 1; }
    sed "s|$pattern|$replacement|" "$header" > "$dir/$(basename "$header")"
    if cmp -s "$header" "$dir/$(basename "$header")"; then
        echo "FAIL mutation pattern did not change $header: $pattern"; return 1
    fi
}
must_fail() {
    local rc=0
    "$@" || rc=$?
    [[ $rc == 1 ]] || { echo "FAIL negative control did not fail with rc 1 (rc=$rc): $*"; return 1; }
    echo "killed: $*"
}
contracts() {
    host_build q8-cache-contract codex/k6/q8-cache-contract.cpp
    "$BIN/q8-cache-contract"
    host_build add-rms-norm-matcher codex/k6/add-rms-norm-matcher.cpp
    "$BIN/add-rms-norm-matcher"
}
contract_mutants() {
    local m=$D/mutants
    mutate ggml/src/ggml-cuda/q8-cache.h "$m/stale" 'if (start <= src ? src - start < size : start - src < src_size) {' 'if (false) {'
    host_build q8-cache-stale codex/k6/q8-cache-contract.cpp "$m/stale"
    must_fail "$BIN/q8-cache-stale"
    mutate ggml/src/ggml-cuda/q8-cache.h "$m/bound" 'GGML_CUDA_Q8_REUSE_MAX_ROWS = 16' 'GGML_CUDA_Q8_REUSE_MAX_ROWS = 17'
    host_build q8-cache-bound codex/k6/q8-cache-contract.cpp "$m/bound"
    must_fail "$BIN/q8-cache-bound"
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
shared_identity() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_SHARED -j 1 > "$D/shared.out" 2>&1
    grep -aq 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS' "$D/shared.out"
}
add_rms_identity() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o RMS_NORM_MUL_ADD,ADD_RMS_NORM -j 1 > "$D/add-rms.out" 2>&1
    grep -aq 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS' "$D/add-rms.out"
}
k6_cheap() {
    stage 60 host-contracts contracts
    stage 61 host-contract-mutants contract_mutants
    stage 62 mul-mat-shared-identity shared_identity
    stage 63 add-rms-norm-identity add_rms_identity
}
