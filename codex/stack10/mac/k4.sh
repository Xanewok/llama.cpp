# K4 SSM_CONV_RS: indexed conv-state read. The identity run must engage (cases > 0) and the off switch must hand every case back
# (cases = 0). Codes 50-59.
unset GGML_SSM_CONV_RS_OFF GGML_SSM_CONV_RS_MUTANT GGML_SSM_CONV_RS_LOG
ops() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV_RS -j 1 | tee "$D/ops-on.txt"
    grep -aq 'byte identity with the gathered path and changing-id replay PASS cases=[1-9]' "$D/ops-on.txt"
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV -j 1 > "$D/ops-ssm-conv.txt"
    "$BW_GATE" env GGML_SSM_CONV_RS_OFF=1 build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV_RS -j 1 | tee "$D/ops-off.txt"
    grep -aq 'PASS cases=0 ' "$D/ops-off.txt"
}
ops_mutants() {
    local m
    for m in 1 2; do
        if "$BW_GATE" env GGML_SSM_CONV_RS_MUTANT=$m build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV_RS -j 1 > "$D/ops-mutant$m.txt" 2>&1; then
            echo "FAIL ops mutant $m escaped"; return 1
        fi
        grep -aq 'FAIL SSM_CONV_RS byte identity' "$D/ops-mutant$m.txt"
        echo "ops mutant $m killed: FAIL SSM_CONV_RS byte identity"
    done
}
# arm NAME RS_SEQ EXPECT REF ENV...: writes $D/model/NAME.bin, or with REF compares in-process, byte for byte, against REF.bin.
arm() {
    local name=$1 rs_seq=$2 expect=$3 ref=$4
    shift 4
    local cmp=(--getter-logits "$D/model/$name.bin")
    [[ -z $ref ]] || cmp=(--getter-reference "$D/model/$ref.bin")
    "$MGATE" env "$@" build-cpu/bin/test-recurrent-state-rollback --getter-gate "${cmp[@]}" --conv-expect "$expect" \
        --getter-rs-seq "$rs_seq" -m "$MODEL" -c 4096 -t 4 -ngl 0 -lv 5 > "$D/model/$name.log" 2>&1
}
model() {
    mkdir -p "$D/model"
    arm gathered8 8 gathered '' GGML_SSM_CONV_RS_OFF=1
    arm rs8 8 rs gathered8 GGML_SSM_CONV_RS_LOG=0
    arm gathered0 0 gathered '' GGML_SSM_CONV_RS_OFF=1
    arm rs0 0 rs gathered0 GGML_SSM_CONV_RS_LOG=0
    local m
    for m in 1 2; do
        if arm mutant$m 8 rs gathered8 GGML_SSM_CONV_RS_MUTANT=$m; then
            echo "FAIL model mutant $m escaped"; return 1
        fi
        grep -aq 'logits differ' "$D/model/mutant$m.log"
        echo "model mutant $m killed: logits differ"
    done
}
np2() {
    mkdir -p "$D/np2"
    local meta
    for meta in 0 1; do
        "$MGATE" env GGML_SSM_CONV_RS_OFF=1 HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
            build-cpu/bin/bw-np2-driver "$MODEL" "$D/np2/$meta-off.json" "$meta" 2 1729 64 > "$D/np2/$meta-off.log" 2>&1
        "$MGATE" env HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
            build-cpu/bin/bw-np2-driver "$MODEL" "$D/np2/$meta-on.json" "$meta" 2 1729 64 > "$D/np2/$meta-on.log" 2>&1
        cmp "$D/np2/$meta-off.json" "$D/np2/$meta-on.json"
    done
}
k4_cheap() {
    stage 50 ssm-conv-rs-ops-identity ops
    stage 51 ssm-conv-rs-ops-mutants ops_mutants
}
k4_model() {
    stage 52 ssm-conv-rs-model-gate model
    stage 53 ssm-conv-rs-np2-driver np2
}
