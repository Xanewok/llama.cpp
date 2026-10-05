#!/bin/bash
# CPU identity matrix for fused MTP catch-up; one model process at a time.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""
B=$PWD/../build-cpu/bin/cpu-serving
R=$PWD/matrix-results
arm() { # name fuse extra-args...
    local name=$1 fuse=$2; shift 2
    [ -s "$R/$name/records.json" ] && return 0
    mkdir -p "$R/$name"
    ~/repos/qwen38-eval-worktrees/bin/cpugate python3 run_arm.py "$R/$name" "$B" --env LLAMA_MTP_FUSE_CATCHUP=$fuse "$@" > "$R/$name/stdout.txt" 2>&1 || echo "FAILED $name"
}
cmpr() { echo "$1 vs $2: $(python3 cmp.py "$R/$1" "$R/$2")"; }
# T1 np1 identity under batch-invariant kernels (-nr)
arm t1-eager 0 --prompts 20 --seeds 1 7 --predict 32 --extra=-nr
arm t1-fused 1 --prompts 20 --seeds 1 7 --predict 32 --extra=-nr
cmpr t1-eager t1-fused
# T2 negative controls (must differ)
arm t2-neg1 1 --env LLAMA_MTP_FUSE_NEGCTL=1 --prompts 20 --seeds 1 7 --predict 32 --extra=-nr
cmpr t1-eager t2-neg1
arm t2-neg2 1 --env LLAMA_MTP_FUSE_NEGCTL=2 --prompts 20 --seeds 1 7 --predict 32 --extra=-nr
cmpr t1-eager t2-neg2
# T3 forced target checkpoint replay
arm t3-eager 0 --env LLAMA_DIAG_FORCE_REPLAY=1 --prompts 6 --seeds 1 --predict 32 --extra=-nr
arm t3-fused 1 --env LLAMA_DIAG_FORCE_REPLAY=1 --prompts 6 --seeds 1 --predict 32 --extra=-nr
cmpr t3-eager t3-fused
# T4 two cached turns (flush on release)
arm t4-eager 0 --turns 2 --prompts 6 --seeds 1 --predict 32 --extra=-nr
arm t4-fused 1 --turns 2 --prompts 6 --seeds 1 --predict 32 --extra=-nr
cmpr t4-eager t4-fused
# T5 np2 concurrent
arm t5-eager 0 --np 2 --prompts 8 --seeds 1 --predict 32 --extra=-nr
arm t5-fused 1 --np 2 --prompts 8 --seeds 1 --predict 32 --extra=-nr
cmpr t5-eager t5-fused
# T6 default kernels (repack on): divergence rate, distribution only
arm t6-eager 0 --prompts 20 --seeds 1 7 --predict 32
arm t6-fused 1 --prompts 20 --seeds 1 7 --predict 32
cmpr t6-eager t6-fused
