# K10 batched top-k: the CUDA kernels run on CPU threads (tests/test-topk-batched.cpp, emulation build), negative controls, and the
# CPU side of the new TOP_K cases. Codes 80-89.
emu_build() {
    "$BW_BUILDGATE" c++ -std=c++17 -O1 -pthread tests/test-topk-batched.cpp -o "$D/test-topk-batched-emu"
}
emu_run() {
    "$BW_GATE" "$D/test-topk-batched-emu"
}
emu_mutants() {
    local m
    for m in 1 2 3 4 5 6; do
        if env TOPK_MUTANT=$m TOPK_FAILFAST=1 "$BW_GATE" "$D/test-topk-batched-emu" > "$D/mutant-$m.out" 2>&1; then
            echo "FAIL mutant $m escaped"; return 1
        fi
        grep -aq '^FAIL ' "$D/mutant-$m.out"
        echo "mutant $m killed: $(grep -a '^FAIL ' "$D/mutant-$m.out" | sed -n 1p)"
    done
}
cmake_unit() {
    "$BW_GATE" build-cpu/bin/test-topk-batched
}
# canonical cases are CUDA-only: not supported on CPU, and the other TOP_K cases still pass
topk_cpu_ops() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o TOP_K -j 1 > "$D/top-k-cpu.out" 2>&1
    grep -aq 'not supported' "$D/top-k-cpu.out"
    grep -aq 'canonical=1' "$D/top-k-cpu.out"
}
# the CPU order (partial sort, first two swapped) is not canonical: the exact-order check must reject it
canonical_control() {
    if env TOPK_CANONICAL_ANY=1 "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o TOP_K -p 'canonical=1' -j 1 > "$D/canonical-control.out" 2>&1; then
        echo 'FAIL canonical check accepted the CPU order'; return 1
    fi
    grep -aq 'canonical=1' "$D/canonical-control.out"
}
k10_cheap() {
    stage 80 emu-build emu_build
    stage 81 emu-kernels emu_run
    stage 82 emu-mutants emu_mutants
    stage 83 topk-unit cmake_unit
    stage 84 topk-cpu-ops topk_cpu_ops
    stage 85 canonical-control canonical_control
}
