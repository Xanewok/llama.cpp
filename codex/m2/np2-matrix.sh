#!/usr/bin/env bash
# np2 CPU identity of the multi-sequence MTP chain (M2) through bw-np2-driver. Usage: np2-matrix.sh BIN MODEL OUT
# Cells: <meta>-k<k>-s<seed>-<off|on>, plus plain k3 s1729 variants: unified KV, greedy slot 1 (path switches mid-request)
# and the negative controls. M2_CELLS=quick keeps k5 s1729 (plain + Meta), greedy slot 1, the floor and two mutants.
# M2_GATE replaces the Mac gate (e.g. bin/cpugate). Judge: judge-np2.py OUT [quick]. A gate skip (rc 75) stops with rc 75.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
BIN=${1:?bin} MODEL=${2:?model} OUT=${3:?out}
GATE=${M2_GATE:-$(cd "$(dirname "$0")" && pwd)/gate.sh}
QUICK=$([[ ${M2_CELLS:-} == quick ]] && echo 1 || echo 0)
gate() { if [[ $GATE == *.sh ]]; then bash "$GATE" "$@"; else "$GATE" "$@"; fi; }
mkdir -p "$OUT"
run() { # tag meta k seed chain [ENV=VAL ...]
    local tag=$1 meta=$2 k=$3 seed=$4 chain=$5
    shift 5
    [[ -s $OUT/$tag.json ]] && return 0
    : > "$OUT/$tag.trace.jsonl"
    local t0=$SECONDS rc=0
    (
        export LLAMA_MTP_CHAIN="$chain" NP2_N_MAX="$k" NP2_THREADS=4 NP2_ALLOW_SHORT=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 "$@" \
            LLAMA_ROUND_TRACE="$OUT/$tag.trace.jsonl" LLAMA_MTP_CHAIN_DUMP="$OUT/$tag.dump.txt"
        gate "$BIN" "$MODEL" "$OUT/$tag.json" "$meta" 2 "$seed" 64
    ) > "$OUT/$tag.log" 2>&1 || rc=$?
    echo "$tag rc=$rc elapsed_s=$((SECONDS - t0))"
    [[ $rc == 75 ]] && { grep 'skipped: RAM' "$OUT/$tag.log"; exit 75; }
    echo "$rc" > "$OUT/$tag.rc"
}
for meta in 0 1; do
    for k in 3 5; do
        [[ $QUICK == 1 && $k == 3 ]] && continue
        for seed in 1729 2718; do
            [[ $QUICK == 1 && $seed == 2718 ]] && continue
            for chain in 0 1; do
                run "$meta-k$k-s$seed-$([[ $chain == 1 ]] && echo on || echo off)" "$meta" "$k" "$seed" "$chain"
            done
        done
    done
done
if [[ $QUICK == 1 ]]; then
    run 0-k5-s1729-off-repeat 0 5 1729 0
    run 0-k5-s1729-greedy1-off 0 5 1729 0 NP2_SLOT1_TEMP=0
    run 0-k5-s1729-greedy1-on  0 5 1729 1 NP2_SLOT1_TEMP=0
    run 0-k5-s1729-mut-swap 0 5 1729 1 LLAMA_MTP_CHAIN_DEBUG_SWAP_SEQ=1
    run 0-k5-s1729-greedy1-mut-norng 0 5 1729 1 NP2_SLOT1_TEMP=0 LLAMA_MTP_CHAIN_DEBUG_NO_RNG_SYNC=1
    echo "np2 matrix runs done (quick)"
    exit 0
fi
run 0-k3-s1729-off-repeat 0 3 1729 0
run 0-k3-s1729-uni-off 0 3 1729 0 NP2_KV_UNIFIED=1
run 0-k3-s1729-uni-on  0 3 1729 1 NP2_KV_UNIFIED=1
run 0-k3-s1729-greedy1-off 0 3 1729 0 NP2_SLOT1_TEMP=0
run 0-k3-s1729-greedy1-on  0 3 1729 1 NP2_SLOT1_TEMP=0
run 1-k3-s1729-greedy1-off 1 3 1729 0 NP2_SLOT1_TEMP=0
run 1-k3-s1729-greedy1-on  1 3 1729 1 NP2_SLOT1_TEMP=0
run 0-k3-s1729-mut-swap 0 3 1729 1 LLAMA_MTP_CHAIN_DEBUG_SWAP_SEQ=1
run 0-k3-s1729-mut-swap-uni 0 3 1729 1 LLAMA_MTP_CHAIN_DEBUG_SWAP_SEQ=1 NP2_KV_UNIFIED=1
run 0-k3-s1729-mut-revu 0 3 1729 1 LLAMA_MTP_CHAIN_DEBUG_REVERSE_U=1
run 0-k3-s1729-mut-shift 0 3 1729 1 LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD=1
run 0-k3-s1729-greedy1-mut-norng 0 3 1729 1 NP2_SLOT1_TEMP=0 LLAMA_MTP_CHAIN_DEBUG_NO_RNG_SYNC=1
echo "np2 matrix runs done"
