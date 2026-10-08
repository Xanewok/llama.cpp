# K9 MMQ PDL: CUDA code is not compiled here; this validates the mmq-chain harness on the CPU backend (stable hashes, alternating
# inputs differ, corruption is caught, perf mode runs). Codes 90-99.
T=build-cpu/bin/test-backend-ops
# run_chain TAG ARGS...: harness lines to $D/TAG.out (the gate may print WAIT lines on stdout), stderr to $D/TAG.err, status in RC
run_chain() {
    local tag=$1
    shift
    RC=0
    "$BW_GATE" $T mmq-chain CPU "$@" > "$D/$tag.raw" 2> "$D/$tag.err" || RC=$?
    grep -a '^\(chain\|perf\) ' "$D/$tag.raw" > "$D/$tag.out" || true
}
chain() {
    run_chain hash-1 hash 8 quick
    [[ $RC == 0 ]]
    run_chain hash-2 hash 8 quick
    [[ $RC == 0 ]]
    cmp "$D/hash-1.out" "$D/hash-2.out"
    [[ $(grep -c '^chain ' "$D/hash-1.out") == 2 ]]
    awk '$1 == "chain" && $5 == $6 {bad = 1} END {exit bad}' "$D/hash-1.out"
    [[ $(grep -c 'recompute mismatches 0 of 8' "$D/hash-1.err") == 2 ]]
    run_chain corrupt hash 8 quick corrupt
    [[ $RC == 1 ]]
    grep -aq 'mmq-chain: FAIL 2 recomputes differ' "$D/corrupt.err"
    echo 'corrupt control killed: mmq-chain: FAIL 2 recomputes differ'
    run_chain perf perf 3 quick
    [[ $RC == 0 ]]
    [[ $(grep -c '^perf ' "$D/perf.out") == 2 ]]
}
k9_cheap() {
    stage 90 mmq-chain-harness chain
}
