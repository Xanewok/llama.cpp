#!/usr/bin/env bash
# One CPU model process at a time, each through cpugate. Usage: kvpad_seq.sh SCENARIO META PAD...
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" CPU_THREADS=1
scenario=$1; meta=$2; shift 2
for pad in "$@"; do
  PADS=$pad N_DECODE=${N_DECODE:-512} ~/repos/qwen38-eval-worktrees/bin/cpugate python3 codex/kvpad_cpu.py codex/kvpad-results/bitwise/pad$pad "$scenario" "$meta" > codex/kvpad-results/bitwise-$scenario-m$meta-pad$pad.out 2>&1
done
echo "SEQ_DONE $scenario $meta"
