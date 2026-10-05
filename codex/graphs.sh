#!/bin/bash
# Graph reuse per phase: eager/fused x LLAMA_GRAPH_SLOTS 2 (old two-slot behaviour) / 12.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""
B=$PWD/../build-cpu/bin/cpu-serving
R=$PWD/graph-results
for fuse in 0 1; do for slots in 2 12; do
    name=f$fuse-s$slots; [ -s "$R/$name/records.json" ] && continue
    mkdir -p "$R/$name"
    ~/repos/qwen38-eval-worktrees/bin/cpugate python3 run_arm.py "$R/$name" "$B" --env LLAMA_MTP_FUSE_CATCHUP=$fuse \
        --env LLAMA_GRAPH_SLOTS=$slots --env LLAMA_ROUND_TRACE=$R/$name/trace.jsonl \
        --prompts 4 --seeds 1 --predict 64 --extra=-nr > "$R/$name/stdout.txt" 2>&1 || echo "FAILED $name"
done; done
for a in f0-s2 f0-s12 f1-s2 f1-s12; do echo "$a: $(python3 cmp.py $R/f0-s2 $R/$a)"; done
