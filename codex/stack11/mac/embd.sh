# stack11 port check: master #29622/#30257 moved the MTP hidden-state input from ubatch.embd to ubatch.embd_state, and stack11 moved
# the fold (c19629b30) and the device chain (21ba77aa8/d298d3dfd) onto it (5a778545e). On CPU each path must reproduce its
# switched-off reference exactly: tokens and draft/accept counters of bw-np2-driver JSON. Stage codes 160-169.
# Contract: arms differ only in LLAMA_MTP_FOLD_C / LLAMA_MTP_CHAIN; the engagement line must be present in the "on" log and absent
# in the "off" log, so a silently disengaged path cannot pass as identical.
unset LLAMA_MTP_CHAIN LLAMA_MTP_FOLD_C
drive() {
    local name=$1 meta=$2 np=$3
    shift 3
    "$MGATE" env "$@" HOST_KV_Q8=1 NP2_THREADS=2 NP2_N_MAX=5 build-cpu/bin/bw-np2-driver "$MODEL" "$D/$name.json" "$meta" "$np" 1729 48 \
        > "$D/$name.log" 2>&1
}
same() {
    python3 - "$D/$1.json" "$D/$2.json" <<'PY_SAME'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a == b, f'{sys.argv[1]} != {sys.argv[2]}: tokens or draft/accept counters differ'
print('identical', sys.argv[1], sys.argv[2])
PY_SAME
}
fold_identity() {
    local meta
    for meta in 0 1; do
        drive "fold-$meta-off" "$meta" 2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=0
        drive "fold-$meta-on" "$meta" 2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1
        [[ $(grep -ac 'mtp_fold_decode: engaged n_seq=2' "$D/fold-$meta-on.log") -ge 1 ]]
        [[ $(grep -ac 'mtp_fold_decode: engaged' "$D/fold-$meta-off.log") == 0 ]]
        same "fold-$meta-off" "fold-$meta-on"
    done
}
chain_identity() {
    local meta
    for meta in 0 1; do
        drive "chain-$meta-off" "$meta" 1 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1
        drive "chain-$meta-on" "$meta" 1 LLAMA_MTP_CHAIN=1 LLAMA_MTP_FOLD_C=1
        [[ $(grep -ac 'MTP chain: k=5' "$D/chain-$meta-on.log") -ge 1 ]]
        [[ $(grep -ac 'MTP chain: k=' "$D/chain-$meta-off.log") == 0 ]]
        same "chain-$meta-off" "chain-$meta-on"
    done
}
backend_sampler() {
    "$MGATE" build-cpu/bin/test-backend-sampler -m "$MODEL" > "$D/backend-sampler.out" 2>&1
}
embd_cheap() {
    stage 160 kv-mask-units "$BW_GATE" build-cpu/bin/test-kv-mask
}
embd_model() {
    stage 161 backend-sampler backend_sampler
    stage 162 fold-embd-state-identity fold_identity
    stage 163 chain-embd-state-identity chain_identity
}
