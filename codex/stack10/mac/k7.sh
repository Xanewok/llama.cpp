# K7 paired q/k RMS norm + scale. Codes 40-49.
T=build-cpu/bin/test-backend-ops
pair() {
    "$BW_GATE" $T test -b CPU -o RMS_NORM_SCALE_PAIR -j 1 > "$D/pair.out" 2>&1
    grep -ac 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$D/pair.out" > /dev/null
}
neighbours() {
    "$BW_GATE" $T test -b CPU -o RMS_NORM -j 1 > "$D/rms-norm.out" 2>&1
    "$BW_GATE" $T test -b CPU -o SCALE -j 1 > "$D/scale.out" 2>&1
}
# Each guard must be what rejects its unsafe layout: with it removed the test has to fail with its own message.
alias_mutants() {
    local mask msg
    for mask in 1 2 4 8; do
        case $mask in
            1) msg='first output over the second input' ;;
            2) msg='second output over the first input' ;;
            4) msg='outputs overlap' ;;
            8) msg='second input is a view of the first output' ;;
        esac
        if "$BW_GATE" env GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT=$mask $T test -b CPU -o RMS_NORM_SCALE_PAIR -j 1 > "$D/alias-mutant-$mask.out" 2>&1; then
            echo "FAIL alias-guard mutant $mask escaped"; return 1
        fi
        grep -ac "FAIL RMS_NORM_SCALE_PAIR matcher accepted: $msg" "$D/alias-mutant-$mask.out" > /dev/null
        echo "alias mutant $mask killed: $msg"
    done
}
# Real qwen35 graph: the q and k chains must be adjacent for the matcher (sched debug lists non-view nodes).
graph() {
    "$MGATE" env GGML_SCHED_DEBUG=2 build-cpu/bin/llama-completion -m "$MODEL" -c 256 -t 4 -ngl 0 -n 2 -p 'Hello' -no-cnv --no-warmup -lv 5 > "$D/graph-raw.log" 2>&1
    python3 codex/k7/analyze_graph.py "$D/graph-raw.log"
    rm -f "$D/graph-raw.log"
}
k7_cheap() {
    stage 40 alias-guard-mutants alias_mutants
    stage 41 pair-identity pair
    stage 42 neighbour-op-tests neighbours
}
k7_model() {
    stage 43 real-graph-adjacency graph
}
