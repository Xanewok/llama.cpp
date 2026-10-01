#!/usr/bin/env bash
# Parent-only; each fresh Python boot tests all quantization variants.
# Frozen roots must contain bin/python and SHA256SUMS covering the interpreter,
# vLLM Python files and CUDA libraries, torch libraries, and any tuned extension.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ ${1:-} != --execute ]]; then
    export CUDA_VISIBLE_DEVICES=''
    nice -n 19 python3 "$ROOT/codex/marlin_bench.py"
    echo 'Plan: up patch patch up up patch; devices 0 then 1; six fresh boots; budget 1200 s.'
    exit 0
fi
: "${PARENT_GPU:?Set PARENT_GPU=1 only in the parent GPU queue}"
[[ $PARENT_GPU == 1 ]]
: "${CUDA_VISIBLE_DEVICES:?Parent must select the two idle GPUs}"
: "${UP_ROOT:?Frozen baseline environment root}"
: "${PATCH_ROOT:?Frozen candidate environment root}"
OUT=${OUT:-$ROOT/codex/gpu-results-$(date +%Y%m%d-%H%M%S)}
[[ $OUT == "$ROOT/"* ]] || { echo 'OUT must be inside this worktree' >&2; exit 6; }
mkdir "$OUT"
deadline=$((SECONDS + ${BUDGET_S:-1200}))
xid() {
    journalctl -k -b --no-pager > "$OUT/kernel.log"
    if rg -qi 'Xid|fallen off' "$OUT/kernel.log"; then echo 'Xid/fallen-off detected' >&2; exit 3; fi
}
for frozen in "$UP_ROOT" "$PATCH_ROOT"; do
    [[ -x $frozen/bin/python && -f $frozen/SHA256SUMS ]]
    (cd "$frozen" && sha256sum -c --quiet SHA256SUMS)
done
sha256sum "$ROOT/codex/marlin_bench.py" "$ROOT/codex/evidence/shapes.json" > "$OUT/harness.sha256"
xid
if pgrep -x llama-server >/dev/null; then echo 'Another server is running' >&2; exit 4; fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits > "$OUT/gpu-before.csv"
while IFS=, read -r idx mem util; do
    (( mem <= 1500 && util == 0 )) || { echo "GPU $idx is busy" >&2; exit 4; }
done < "$OUT/gpu-before.csv"
pid=
cleanup() {
    if [[ -n $pid ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        pid=
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
boot=0
for arm in up patch patch up up patch; do
    boot=$((boot + 1))
    frozen=$UP_ROOT
    tune=${UP_MARLIN_TUNE:-0}
    [[ $arm == up ]] || frozen=$PATCH_ROOT
    [[ $arm == up ]] || tune=${PATCH_MARLIN_TUNE:-0}
    (cd "$frozen" && sha256sum -c --quiet SHA256SUMS)
    for device in 0 1; do
        left=$((deadline - SECONDS))
        (( left > 0 )) || exit 7
        cap=180
        (( left >= cap )) || cap=$left
        # timeout owns only this boot; cleanup signals its exact PID.
        env -u VLLM_MARLIN_INPUT_DTYPE PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$frozen/lib/python3.12/site-packages" \
            FROZEN_ROOT="$frozen" VLLM_MARLIN_USE_ATOMIC_ADD=0 \
            VLLM_MARLIN_TUNE="$tune" nice -n 19 timeout --signal=TERM --kill-after=5 "$cap" \
            "$frozen/bin/python" "$ROOT/codex/marlin_bench.py" --execute --device "$device" --budget-s "$cap" \
            > "$OUT/$boot-$arm-d$device.jsonl" 2> "$OUT/$boot-$arm-d$device.log" &
        pid=$!
        wait "$pid" || { status=$?; pid=; exit "$status"; }
        pid=
        xid
    done
done
sha256sum -c --quiet "$OUT/harness.sha256"
nice -n 19 python3 "$ROOT/codex/summarize.py" "$OUT" > "$OUT/summary.json"
cat "$OUT/summary.json"
