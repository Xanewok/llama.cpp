#!/usr/bin/env bash
# Meta(CPU,CPU) 33k-token prompt, 8 threads (bitwise is not available under Meta: its CPU sub-backends use 4 threads and the
# split-KV FA path chunks by n_kv). One process at a time, through cpugate.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" CPU_THREADS=8
for pad in "$@"; do
  PADS=$pad ~/repos/qwen38-eval-worktrees/bin/cpugate python3 codex/kvpad_cpu.py codex/kvpad-results/long/pad$pad long 1 > codex/kvpad-results/long-pad$pad.out 2>&1
done
echo "LONG_DONE"
