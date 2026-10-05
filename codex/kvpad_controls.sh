#!/usr/bin/env bash
# Negative controls and plumbing, each step through cpugate (one CPU model process at a time).
#  mutA: n_ctx padded to 256 only  -> --kv-pad 4096 on a 6144-cell context must hit GGML_ASSERT(kv_size % n_pad == 0)
#  mutB: padded cells left unmasked -> bitwise identity vs the unmutated 1-thread reference must break (pads 0 and 4096)
set -u
cd "$(dirname "$0")/.."
G=~/repos/qwen38-eval-worktrees/bin/cpugate
export CUDA_VISIBLE_DEVICES=""
M=/mnt/scratch/work/kv-pad
O=codex/kvpad-results/controls
mkdir -p $O
PADS=4096 $G python3 codex/kvpad_cpu.py $O/assert-mutA assert 0 $M/mutA
PADS=0    $G python3 codex/kvpad_cpu.py $O/assert-mutA-default assert 0 $M/mutA
PADS=4096 $G python3 codex/kvpad_cpu.py $O/assert-fixed assert 0
for pad in 0 4096; do
  CPU_THREADS=1 N_DECODE=64 PADS=$pad $G python3 codex/kvpad_cpu.py $O/maskB-pad$pad decode 0 $M/mutB
done
$G python3 codex/kvpad_cli_check.py
echo CONTROLS_DONE
