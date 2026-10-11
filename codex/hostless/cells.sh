#!/usr/bin/env bash
# Hostless spike CPU identity cells through bw-np2-driver (np2, k5, seed 1729, chain on). Usage: cells.sh BIN MODEL OUT PART
# PART a: plain CPU L0, L0 repeat (floor), L2, L3, L2 slotshift mutant (must differ).
# PART b: HOST_DEFER (CPU backends as streams) L0, L2, L3; Meta(CPU,CPU) tensor split L0..L3 (L1 engages only there).
# PART all: both. HL_GATE replaces the Mac gate (e.g. bin/cpugate). A gate skip (rc 75) stops with rc 75.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
BIN=${1:?bin} MODEL=${2:?model} OUT=${3:?out} PART=${4:?part}
GATE=${HL_GATE:-$(cd "$(dirname "$0")/.." && pwd)/m2/gate.sh}
gate() { if [[ $GATE == *.sh ]]; then bash "$GATE" "$@"; else "$GATE" "$@"; fi; }
mkdir -p "$OUT"
run() { # tag meta level [ENV=VAL ...]
    local tag=$1 meta=$2 level=$3
    shift 3
    [[ -s $OUT/$tag.json ]] && return 0
    : > "$OUT/$tag.trace.jsonl"
    local t0=$SECONDS rc=0
    (
        export LLAMA_MTP_CHAIN=1 NP2_N_MAX=5 NP2_THREADS=4 NP2_ALLOW_SHORT=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
            LLAMA_HOSTLESS="$level" "$@" LLAMA_ROUND_TRACE="$OUT/$tag.trace.jsonl" LLAMA_MTP_CHAIN_DUMP="$OUT/$tag.dump.txt"
        gate "$BIN" "$MODEL" "$OUT/$tag.json" "$meta" 2 1729 64
    ) > "$OUT/$tag.log" 2>&1 || rc=$?
    echo "$tag rc=$rc elapsed_s=$((SECONDS - t0))"
    [[ $rc == 75 ]] && { grep 'skipped: RAM' "$OUT/$tag.log"; exit 75; }
    echo "$rc" > "$OUT/$tag.rc"
}
unset LLAMA_HOSTLESS LLAMA_HOSTLESS_MUTANT HOST_DEFER LLAMA_MTP_FOLD_C
if [[ $PART == a || $PART == all ]]; then
    run 0-L0 0 0
    run 0-L0-repeat 0 0
    run 0-L2 0 2
    run 0-L3 0 3
    run 0-L2-mut-slotshift 0 2 LLAMA_HOSTLESS_MUTANT=slotshift
fi
if [[ $PART == b || $PART == all ]]; then
    run 0d-L0 0 0 HOST_DEFER=1
    run 0d-L2 0 2 HOST_DEFER=1
    run 0d-L3 0 3 HOST_DEFER=1
    run 1-L0 1 0
    run 1-L1 1 1
    run 1-L2 1 2
    run 1-L3 1 3
fi
echo "hostless cells done ($PART)"
