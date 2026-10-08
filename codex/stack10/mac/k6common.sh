# Shared by k6a and k6b: host contracts compile the real CUDA headers against the CPU toolchain.
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
