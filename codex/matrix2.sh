#!/bin/bash
# After the replay pairing fix (0628a408f): replay identity, no-replay no-op, np2 determinism control.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""
B=$PWD/../build-cpu/bin/cpu-serving
R=$PWD/matrix-results
arm() {
    local name=$1 fuse=$2; shift 2
    [ -s "$R/$name/records.json" ] && return 0
    mkdir -p "$R/$name"
    ~/repos/qwen38-eval-worktrees/bin/cpugate python3 run_arm.py "$R/$name" "$B" --env LLAMA_MTP_FUSE_CATCHUP=$fuse "$@" > "$R/$name/stdout.txt" 2>&1 || echo "FAILED $name"
}
cmpr() { echo "$1 vs $2: $(python3 cmp.py "$R/$1" "$R/$2")"; }
arm t3b-eager 0 --env LLAMA_DIAG_FORCE_REPLAY=1 --prompts 6 --seeds 1 --predict 32 --extra=-nr
arm t3b-fused 1 --env LLAMA_DIAG_FORCE_REPLAY=1 --prompts 6 --seeds 1 --predict 32 --extra=-nr
cmpr t3b-eager t3b-fused
cmpr t3-fused t3b-fused
arm t5b-eager 0 --np 2 --prompts 8 --seeds 1 --predict 32 --extra=-nr
cmpr t5-eager t5b-eager
mkdir -p $R/t5d-eager $R/t5d-fused
arm t5d-eager 0 --env LLAMA_MTP_DUMP=$R/t5d-eager/dump.txt --np 2 --prompts 2 --seeds 1 --predict 32 --extra=-nr
arm t5d-fused 1 --env LLAMA_MTP_DUMP=$R/t5d-fused/dump.txt --np 2 --prompts 2 --seeds 1 --predict 32 --extra=-nr
cmpr t5d-eager t5d-fused
arm t1b-eager 0 --prompts 20 --seeds 1 7 --predict 32 --extra=-nr
cmpr t1-eager t1b-eager
