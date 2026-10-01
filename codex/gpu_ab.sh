#!/usr/bin/env bash
# Parent only. Default is CPU dry-run. Probe first-draft overlap, not GPU speed.
set -euo pipefail
renice -n 19 -p "$$" >/dev/null
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
PY=/home/xanewok/repos/HyperQwen/venv/bin/python
FROZEN=$HERE/frozen
OUT=${OUT:-$HERE/gpu-results-$(date +%Y%m%d-%H%M%S)}
OUT=$(realpath -m -- "$OUT")
WORKTREE=$(dirname -- "$HERE")
[[ $OUT == "$WORKTREE/"* ]] || { echo 'Output must be inside this worktree' >&2; exit 4; }
BUDGET_S=${BUDGET_S:-1800}
DRY_RUN=${DRY_RUN:-1}
GPU_UUIDS=${GPU_UUIDS:-GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f}
declare -A MODEL=(
 [up]=/mnt/scratch/work/models/hq/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8
 [patch]=/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh
)
[[ ${CONTROL_IDENTICAL:-0} == 0 ]] || MODEL[patch]=${MODEL[up]}
ARMS=(up patch patch up)
T0=$SECONDS
PID=
left() { echo "$((BUDGET_S - SECONDS + T0))"; }
stop_child() {
    [[ -n $PID ]] || return 0
    kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    PID=
}
trap stop_child EXIT
trap 'exit 130' INT TERM
# This dry path does not call GPU utilities or start a model process.
(cd "$HERE" && sha256sum --check --quiet frozen/SHA256SUMS)
[[ -d $OUT ]] && { echo "Refuse existing output directory: $OUT" >&2; exit 6; }
if [[ $DRY_RUN == 1 ]]; then
    for arm in "${ARMS[@]}"; do
        printf 'DRY_RUN: %s %s teacher_probe.py --model %s --tokens tokens.json --device cuda:0\n' "$arm" "$PY" "${MODEL[$arm]}"
    done
    echo "Parent run: PARENT_GPU_RUN=1 DRY_RUN=0 bash codex/gpu_ab.sh"
    exit 0
fi
[[ ${PARENT_GPU_RUN:-0} == 1 ]] || { echo 'Parent authorization flag required' >&2; exit 4; }
# Heavy input hashing is bounded too. Stored checkpoint manifests can be stale after prep.
for arm in up patch; do
    timeout --signal=TERM --kill-after=10 "$(left)" sha256sum --check --quiet "$FROZEN/$arm.sha256"
done
[[ $(left) -gt 0 ]] || exit 7
mkdir -p "$OUT"
cp "$FROZEN/SOURCE" "$OUT/SOURCE"
xid_snapshot() { journalctl -k -b --no-pager | sed -n '/Xid\|fallen off/p'; }
xid_snapshot > "$OUT/kernel-before.txt"
[[ ! -s $OUT/kernel-before.txt ]] || { echo 'Refuse pre-existing Xid' >&2; exit 3; }
# Query only the pair reserved by this item; leave other boards out of the guard.
used=$(nvidia-smi -i "$GPU_UUIDS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
[[ $used -le 1500 ]] || { echo "GPUs busy: $used MiB" >&2; exit 4; }
pgrep -x llama-server >/dev/null && { echo 'llama-server active' >&2; exit 4; }
export CUDA_VISIBLE_DEVICES=$GPU_UUIDS
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
for i in "${!ARMS[@]}"; do
    arm=${ARMS[$i]}
    [[ $(left) -gt 0 ]] || exit 7
    timeout --signal=TERM --kill-after=10 "$(left)" nice -n 19 "$PY" "$FROZEN/teacher_probe.py" \
        --model "${MODEL[$arm]}" --tokens "$FROZEN/tokens.json" --device cuda:0 \
        --out "$OUT/$i-$arm.json" > "$OUT/$i-$arm.log" 2>&1 &
    PID=$!
    rc=0; wait "$PID" || rc=$?; PID=
    [[ $rc == 0 ]] || { echo "Probe failed: $arm (rc=$rc)" >&2; exit "$rc"; }
    xid_snapshot > "$OUT/kernel-after.txt"
    cmp -s "$OUT/kernel-before.txt" "$OUT/kernel-after.txt" || exit 3
    (cd "$HERE" && sha256sum --check --quiet frozen/SHA256SUMS)
done
CONTROL_IDENTICAL=${CONTROL_IDENTICAL:-0} CUDA_VISIBLE_DEVICES='' nice -n 19 "$PY" "$FROZEN/compare_probe.py" "$OUT" | tee "$OUT/summary.txt"
[[ $(left) -gt 0 ]] || exit 7
