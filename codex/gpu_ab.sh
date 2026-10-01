#!/usr/bin/env bash
# Parent only. Plan mode never loads a CUDA library or queries a GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CUDA_VISIBLE_DEVICES=''
PLAN=${PLAN:-1}
FROZEN=${FROZEN:-/home/xanewok/repos/qwen38-weekend/frozen/stack2}
MODEL=${MODEL:-/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf}
SUBSET=${SUBSET:-/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json}
OUT=${OUT:-$ROOT/build-gpu-ab}
BUDGET_S=${BUDGET_S:-420}
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
ARMS=(up patch patch up)
if [[ $PLAN != 0 ]]; then
    printf '%s\n' 'PLAN: parent only, no earlier than 2026-10-03 06:00 Europe/Warsaw.' \
        'Frozen stack2 0dba27b90; same bytes in both arms.' \
        'up: GGML_CUDA_FA_MMA_Q8_0=0; patch: GGML_CUDA_FA_MMA_Q8_0=1.' \
        'Interleaved process boots: up patch patch up; k=3,4,5; full/subset; all rollback positions.' \
        'Teacher-forced Swift q8_0 KV, tensor split, backend candidates; compare full F32 logits on identical histories.' \
        'Read-only on campaign data. Exact child-PID cleanup, Xid and idle-device checks, bounded wall time.' \
        "Output: $OUT" \
        'To execute after reviewing: RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_ab.sh'
    exit 0
fi
[[ ${RUN_PARENT_GPU:-0} == 1 ]] || { echo 'Parent execution flag is required'; exit 2; }
T0=$(date +%s)
[[ $(date +%s) -ge $(date -d '2026-10-03T06:00:00+02:00' +%s) ]] || { echo 'Too early'; exit 2; }
case "$OUT" in "$ROOT"/build-*) ;; *) echo 'OUT must be a build-* directory inside this worktree'; exit 2;; esac
[[ ! -e $OUT ]] || { echo 'OUT exists; choose a fresh build-* directory'; exit 2; }
[[ $(head -n 1 "$FROZEN/SOURCE") == 0dba27b9002b8347c6f14a9f4a1651b3478065ef ]] || { echo 'Wrong source'; exit 6; }
(cd "$FROZEN/build/bin" && nice -n 19 sha256sum -c --quiet SHA256SUMS)
[[ $(stat -c %s "$MODEL") == 15475951296 ]] || { echo 'Wrong model size'; exit 6; }
mkdir -p "$OUT/frozen/bin" "$OUT/tmp" "$OUT/cuda-cache"
export TMPDIR="$OUT/tmp" CUDA_CACHE_PATH="$OUT/cuda-cache"
cp "$FROZEN/build/bin/"*.so* "$OUT/frozen/bin/"
cp "$FROZEN/SOURCE" "$OUT/frozen/SOURCE"
cp "$ROOT/codex/check.cpp" "$OUT/frozen/check.cpp"
nice -n 19 g++ -std=c++17 -O2 -DS4_PARENT_GPU -I"$ROOT/common" -I"$ROOT/include" -I"$ROOT/ggml/include" -I"$ROOT/vendor" \
    "$OUT/frozen/check.cpp" "$OUT/frozen/bin/libllama-common.so" "$OUT/frozen/bin/libllama.so" \
    "$OUT/frozen/bin/libggml.so" "$OUT/frozen/bin/libggml-cpu.so" "$OUT/frozen/bin/libggml-base.so" \
    "$ROOT/build-cpu/common/libllama-common-base.a" -Wl,-rpath,'$ORIGIN' -Wl,-rpath-link,"$OUT/frozen/bin" -o "$OUT/frozen/bin/s4-gpu-probe"
(cd "$OUT/frozen" && nice -n 19 sha256sum SOURCE check.cpp bin/* > SHA256SUMS)
nice -n 19 sha256sum "$MODEL" "$SUBSET" > "$OUT/INPUTS.sha256"
PID=
stop() {
    [[ -n $PID ]] || return 0
    if kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        for _ in {1..10}; do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null || true
    fi
    wait "$PID" 2>/dev/null || true
    PID=
}
trap stop EXIT
trap 'exit 130' INT TERM
xid() {
    local log
    log=$(journalctl -k -b --no-pager) || { echo 'Cannot read kernel log'; return 3; }
    if [[ $log == *Xid* || $log == *'fallen off'* ]]; then echo 'Xid/fallen-off detected'; return 3; fi
}
export CUDA_VISIBLE_DEVICES=$UUIDS OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$OUT/frozen/bin"
xid
[[ -z $(pgrep -x llama-server || true) && -z $(pgrep -x weekend-srv || true) ]] || { echo 'Serving process is running'; exit 4; }
usage=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits) || exit 4
found=0
while IFS=, read -r uuid used; do
    [[ $UUIDS == *"$uuid"* ]] || continue
    found=$((found + 1))
    [[ ${used// /} -le 1500 ]] || { echo "Device busy: $uuid $used MiB"; exit 4; }
done <<< "$usage"
[[ $found == 2 ]] || { echo 'Expected devices not found'; exit 4; }
for i in "${!ARMS[@]}"; do
    [[ $(( $(date +%s) - T0 )) -lt $BUDGET_S ]] || { echo 'Wall budget exceeded'; exit 7; }
    arm=${ARMS[$i]}; run="$OUT/$i-$arm"
    mkdir -p "$run/logits"
    (cd "$OUT/frozen" && nice -n 19 sha256sum -c --quiet SHA256SUMS)
    nice -n 19 sha256sum -c --quiet "$OUT/INPUTS.sha256"
    q8=1; [[ $arm == up ]] && q8=0
    env GGML_CUDA_FA_MMA_Q8_0=$q8 S4_SUBSET="$SUBSET" S4_LOGITS_DIR="$run/logits" S4_DIAGNOSTIC=1 \
        nice -n 19 "$OUT/frozen/bin/s4-gpu-probe" "$MODEL" gpu > "$run/probe.log" 2>&1 &
    PID=$!
    while kill -0 "$PID" 2>/dev/null; do
        [[ $(( $(date +%s) - T0 )) -lt $BUDGET_S ]] || { echo 'Wall budget exceeded'; exit 7; }
        [[ $(ps -o stat= -p "$PID" 2>/dev/null || true) != Z* ]] || break
        sleep 1
    done
    status=0; wait "$PID" || status=$?; PID=
    echo "$status" > "$run/status"
    [[ $(( $(date +%s) - T0 )) -lt $BUDGET_S ]] || { echo 'Wall budget exceeded'; exit 7; }
    xid
done
nice -n 19 python3 "$ROOT/codex/gpu_compare.py" "$OUT" | tee "$OUT/summary.txt"
