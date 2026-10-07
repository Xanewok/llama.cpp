#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:?output}
mkdir -p "$OUT"
CPUGATE=${S7_GATE:-$HOME/repos/qwen38-eval-worktrees/bin/cpugate}
MODEL=${S7_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
for meta in 0 1; do
    for chain in 0 1; do
        env LLAMA_MTP_CHAIN="$chain" LLAMA_MTP_FOLD_C=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
            "$CPUGATE" build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/$meta-$chain.json" "$meta" 2 1729 64 \
            > "$OUT/$meta-$chain.log" 2>&1
    done
done
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
for meta in (0,1):
    off=json.loads((p/f'{meta}-0.json').read_text())
    on=json.loads((p/f'{meta}-1.json').read_text())
    # Report-only: the np2 chain fallback is not exact (bw-stack7 a1, cause open); np2/duo items run chain off.
    if off != on:
        print(f'WARN meta={meta}: np2 chain request changed tokens or counters')
    for chain in (0,1):
        log=(p/f'{meta}-{chain}.log').read_text()
        assert 'LLAMA_MTP_FOLD_C ignored: needs one sequence' in log, (meta,chain,'fold gate not engaged')
print('np2 fold gate PASS (chain comparison report-only)')
PY
