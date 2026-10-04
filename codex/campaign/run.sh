#!/usr/bin/env bash
# Entry for the stack3-mask A/B items. MODE=run|canary (DRY_RUN=1 = CPU preflight). gpuq freezes this file into a hidden sibling: locate files by absolute path.
# The server inherits this process's nice, and the contract is nice 0: only the CPU preflight is niced.
set -uo pipefail
HERE=/home/xanewok/repos/stack3-mask/codex/campaign
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export UV_OFFLINE=1 HF_HUB_OFFLINE=1
: "${AB_ITEM:?AB_ITEM (iq4|j5q) not set}"
export AB_ITEM
MODE=${MODE:-run}
DRY_RUN=${DRY_RUN:-1}
OUT=${OUT:-$HERE/runs/manual}
mkdir -p "$OUT"
cd "$HERE" || exit 8
if [ "$DRY_RUN" = 1 ]; then
  exec nice -n 19 python3 "$HERE/ab_orchestrate.py" preflight --out "$OUT"
fi
case "$MODE" in
  run) exec python3 "$HERE/ab_orchestrate.py" run --out "$OUT" ;;
  canary) exec python3 "$HERE/ab_orchestrate.py" canary --out "$OUT" ;;
  *) echo "unknown MODE $MODE"; exit 8 ;;
esac
