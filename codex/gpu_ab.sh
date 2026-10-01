#!/usr/bin/env bash
# Parent only: identical frozen bytes, interleaved prompt-split controls.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CUDA_VISIBLE_DEVICES=''
PLAN=${PLAN:-1}
CASE=${CASE:-ubatch}
PREFIX=${PREFIX:-257}
K=${K:-5}
ROLLBACK=${ROLLBACK:-5}
FOLLOW=${FOLLOW:-200}
BUDGET_S=${BUDGET_S:-1200}
FROZEN=${FROZEN:-/home/xanewok/repos/qwen38-weekend/frozen/stack2}
MODEL=${MODEL:-/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf}
OUT=${OUT:-$ROOT/build-gpu-state-$(date +%Y%m%d-%H%M%S)}
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
ARMS=(up patch patch up)
UP=()
PATCH=()
case "$CASE" in
    ubatch) UP=(E_UBATCH=128); PATCH=(E_UBATCH=64) ;;
    fusion) PATCH=(GGML_CUDA_DISABLE_FUSION=1) ;;
    bf16) PATCH=(GGML_CUDA_AR_BF16_THRESHOLD=0) ;;
    graphs) PATCH=(GGML_CUDA_DISABLE_GRAPHS=1) ;;
    *) echo 'Unknown CASE'; exit 2 ;;
esac
if [[ $PLAN != 0 ]]; then
    printf '%s\n' "PLAN: frozen stack2; boots up patch patch up; CASE=$CASE." \
        "up: ${UP[*]:-(default)}; patch: ${PATCH[*]:-(default)}." \
        "prefix=$PREFIX k=$K rollback=$ROLLBACK follow=$FOLLOW; wall budget=$BUDGET_S seconds." \
        'Within each boot: rolled state vs independent retained-prefix decode and a distance-one batch control.' \
        'Record conv/GDN state errors before correction and full logits through the continuation.' \
        'Default q8_0 KV, FA on, tensor split 50,50, full head; no sampler or generated-token divergence.' \
        'GPU differences require comparison with the pre-rollback shape noise; no speed claim.' \
        'Execution requires RUN_PARENT_GPU=1 PLAN=0; plan mode never loads CUDA or queries devices.'
    exit 0
fi
[[ ${RUN_PARENT_GPU:-0} == 1 ]] || { echo 'Parent flag required'; exit 2; }
for v in "$PREFIX" "$K" "$ROLLBACK" "$FOLLOW" "$BUDGET_S"; do [[ $v =~ ^[0-9]+$ ]] || exit 2; done
[[ $PREFIX -gt 0 && $PREFIX -le 2048 && $K -ge 1 && $K -le 6 && $ROLLBACK -ge 1 && $ROLLBACK -le $K && $FOLLOW -le 200 && $BUDGET_S -ge 30 ]] || exit 2
case "$OUT" in "$ROOT"/build-*) ;; *) echo 'OUT must be build-* inside this worktree'; exit 2 ;; esac
[[ ! -e $OUT ]] || { echo 'Use a fresh OUT'; exit 2; }
T0=$(date +%s)
left() { echo $((T0 + BUDGET_S - $(date +%s))); }
check_budget() { [[ $(left) -gt 0 ]] || { echo 'Wall budget exceeded'; exit 7; }; }
mkdir -p "$OUT/frozen/bin" "$OUT/tmp" "$OUT/cuda-cache"
export TMPDIR="$OUT/tmp" CUDA_CACHE_PATH="$OUT/cuda-cache" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
(cd "$FROZEN/build/bin" && timeout "$(left)" nice -n 19 sha256sum -c --quiet SHA256SUMS)
timeout --kill-after=2 "$(left)" cp -L "$FROZEN/build/bin/"*.so* "$OUT/frozen/bin/"
cp "$FROZEN/SOURCE" "$OUT/frozen/SOURCE"
source_commit=$(head -1 "$FROZEN/SOURCE")
# The probe uses internal C++ memory layouts; these headers and implementations must match.
git -C "$ROOT" diff --exit-code "$source_commit" HEAD -- include src common ggml/include ggml/src/ggml-cpu >/dev/null
git -C "$ROOT" diff --exit-code -- include src common ggml/include ggml/src/ggml-cpu >/dev/null
cp "$ROOT/codex/state_split_probe.cpp" "$ROOT/codex/state_summary.py" "$OUT/frozen/"
cp "$ROOT/build-cpu/common/libllama-common-base.a" "$OUT/frozen/"
git -C "$ROOT" rev-parse HEAD > "$OUT/frozen/AUDIT_SOURCE"
timeout "$(left)" nice -n 19 g++ -std=c++17 -O2 -DE_PARENT_GPU \
    -I"$ROOT/common" -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/ggml/include" -I"$ROOT/vendor" \
    "$OUT/frozen/state_split_probe.cpp" "$OUT/frozen/bin/libllama-common.so" \
    "$OUT/frozen/bin/libllama.so" "$OUT/frozen/bin/libggml.so" \
    "$OUT/frozen/bin/libggml-cpu.so" "$OUT/frozen/bin/libggml-base.so" \
    "$OUT/frozen/libllama-common-base.a" -Wl,-rpath,'$ORIGIN' \
    -Wl,-rpath-link,"$OUT/frozen/bin" -o "$OUT/frozen/bin/state-split-probe"
(cd "$OUT/frozen" && timeout "$(left)" nice -n 19 sha256sum SOURCE AUDIT_SOURCE *.cpp *.py *.a bin/* > SHA256SUMS)
timeout "$(left)" nice -n 19 sha256sum "$MODEL" > "$OUT/INPUTS.sha256"
printf '%s\n' "CASE=$CASE PREFIX=$PREFIX K=$K ROLLBACK=$ROLLBACK FOLLOW=$FOLLOW BUDGET_S=$BUDGET_S" \
    "up=${UP[*]} patch=${PATCH[*]}" > "$OUT/config.txt"
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
    log=$(timeout --kill-after=2 "$(left)" journalctl -k -b --no-pager) || { echo 'Kernel log unreadable'; return 3; }
    [[ $log != *Xid* && $log != *'fallen off'* ]] || { echo 'Xid/fallen-off detected'; return 3; }
}
idle() {
    [[ -z $(pgrep -x llama-server || true) && -z $(pgrep -x weekend-srv || true) ]] || { echo 'Serving process running'; return 4; }
    local usage found=0 uuid used utilization
    usage=$(timeout --kill-after=2 "$(left)" nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits) || return 4
    while IFS=, read -r uuid used utilization; do
        [[ ,$UUIDS, == *",$uuid,"* ]] || continue
        found=$((found + 1))
        [[ ${used// /} -le 1500 && ${utilization// /} == 0 ]] || { echo "Device busy: $uuid"; return 4; }
    done <<< "$usage"
    [[ $found == 2 ]] || { echo 'Expected devices not found'; return 4; }
}
unset E_EXACT E_BROKEN_SLOT E_F16 E_FA E_THREADS E_UBATCH E_LOGITS_DIR \
    GGML_CUDA_DISABLE_FUSION GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_AR_BF16_THRESHOLD \
    GGML_CUDA_AR_COPY_THRESHOLD GGML_CUDA_AR_COPY_CHUNK_BYTES GGML_CUDA_FA_MMA_Q8_0_NCOLS1 \
    GGML_CUDA_INPUT_BATCH GGML_CUDA_PDL LLAMA_GRAPH_REUSE_DISABLE
export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_FA_MMA_Q8_0=1 GGML_CUDA_GRAPH_OPT=0
export CUDA_VISIBLE_DEVICES=$UUIDS LD_LIBRARY_PATH="$OUT/frozen/bin"
for i in "${!ARMS[@]}"; do
    check_budget; xid; idle
    arm=${ARMS[$i]}; run="$OUT/$i-$arm"
    mkdir -p "$run/logits"
    (cd "$OUT/frozen" && timeout "$(left)" nice -n 19 sha256sum -c --quiet SHA256SUMS)
    timeout "$(left)" nice -n 19 sha256sum -c --quiet "$OUT/INPUTS.sha256"
    knobs=("${UP[@]}"); [[ $arm == patch ]] && knobs=("${PATCH[@]}")
    env "${knobs[@]}" E_SERIALIZED=1 E_PREFIX="$PREFIX" E_K="$K" E_R="$ROLLBACK" E_FOLLOW="$FOLLOW" \
        E_LOGITS_DIR="$run/logits" nice -n 19 "$OUT/frozen/bin/state-split-probe" "$MODEL" gpu \
        > "$run/probe.log" 2>&1 &
    PID=$!
    while kill -0 "$PID" 2>/dev/null; do
        check_budget
        [[ $(ps -o stat= -p "$PID" 2>/dev/null || true) != Z* ]] || break
        sleep 1
    done
    status=0; wait "$PID" || status=$?; PID=
    xid
    [[ $status == 0 ]] || { echo "$status" > "$run/status"; exit 2; }
    echo 0 > "$run/status"
done
check_budget
timeout "$(left)" nice -n 19 python3 "$OUT/frozen/state_summary.py" --gpu "$OUT" | tee "$OUT/summary.json"
