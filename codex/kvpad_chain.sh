#!/usr/bin/env bash
# 1-thread runs, one process per (scenario, meta, pad): bitwise identity needs a thread-count-independent CPU FA
# (the split-KV path chunks by n_kv). Usage: kvpad_chain.sh SCENARIO META PADS...  (<= 3 processes at once)
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" CPU_THREADS=1 OMP_NUM_THREADS=1
scenario=$1; meta=$2; shift 2
for pad in "$@"; do
  PADS=$pad N_DECODE=${N_DECODE:-512} python3 codex/kvpad_cpu.py codex/kvpad-results/bitwise/pad$pad "$scenario" "$meta" > codex/kvpad-results/bitwise-$scenario-m$meta-pad$pad.out 2>&1 &
done
wait
echo "CHAIN_DONE $scenario $meta"
