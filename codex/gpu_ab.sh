#!/usr/bin/env bash
# Parent only. Plan mode cannot load CUDA libraries or inspect devices.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CUDA_VISIBLE_DEVICES=''
PLAN=${PLAN:-1}
CASE=${CASE:-tile}
PROBE=${PROBE:-state}
PREFIX_N=${PREFIX_N:-0}
BUDGET_S=${BUDGET_S:-1200}
BUILD=${BUILD:-$ROOT/build-cuda}
MODEL=${MODEL:-/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf}
SUBSET=${SUBSET:-/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json}
OUT=${OUT:-$ROOT/build-gpu-ab-$CASE-$PROBE-$(date +%Y%m%d-%H%M%S)}
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
ARMS=(up patch patch up)
UP=()
PATCH=()
case "$CASE" in
    tile) PATCH=(GGML_CUDA_FA_MMA_Q8_0_NCOLS1=4) ;;
    upload) PATCH=(GGML_CUDA_INPUT_BATCH=0) ;;
    graphs) PATCH=(GGML_CUDA_DISABLE_GRAPHS=1) ;;
    reuse) PATCH=(LLAMA_GRAPH_REUSE_DISABLE=1) ;;
    fusion) PATCH=(GGML_CUDA_DISABLE_FUSION=1) ;;
    pdl) PATCH=(GGML_CUDA_PDL=0) ;;
    bf16) PATCH=(GGML_CUDA_AR_BF16_THRESHOLD=0) ;;
    allreduce) PATCH=(GGML_CUDA_ALLREDUCE=none) ;;
    dequant) PATCH=(GGML_CUDA_FA_MMA_Q8_0=0) ;;
    target) PATCH=(D_AUDIT_HOST_TARGET=1) ;;
    ubatch) UP=(D_AUDIT_UBATCH=128); PATCH=(D_AUDIT_UBATCH=64) ;;
    *) echo 'Unknown CASE'; exit 2 ;;
esac
[[ $PROBE == state || $PROBE == draft ]] || { echo 'Unknown PROBE'; exit 2; }
if [[ $PLAN != 0 ]]; then
    printf '%s\n' "PLAN: $CASE / $PROBE; identical frozen binary bytes; boots: up patch patch up." \
        "up: ${UP[*]:-(default)}; patch: ${PATCH[*]}." \
        "Teacher prefix: $PREFIX_N tokens (synthetic unless TOKENS is supplied); budget: $BUDGET_S seconds." \
        'State probe: k=3,4,5; all verify rows and rollback depths; target host/candidate seed identity.' \
        'Draft probe: full/compact head; host/backend candidates; deferred scatter; verify rows 4,5,6.' \
        'No speed claim. BF16/AllReduce/tile changes may change floating point logits; differences need diagnosis.' \
        'Guarded GPU execution requires RUN_PARENT_GPU=1 PLAN=0; default is plan only.' \
        "Output: $OUT"
    exit 0
fi
[[ ${RUN_PARENT_GPU:-0} == 1 ]] || { echo 'Parent execution flag required'; exit 2; }
[[ $PREFIX_N =~ ^[0-9]+$ && $PREFIX_N -le 100000 && $BUDGET_S =~ ^[0-9]+$ && $BUDGET_S -ge 30 ]] || exit 2
case "$OUT" in "$ROOT"/build-*) ;; *) echo 'OUT must be build-* inside this worktree'; exit 2 ;; esac
[[ ! -e $OUT ]] || { echo 'Use a fresh OUT'; exit 2; }
T0=$(date +%s)
left() { echo $((T0 + BUDGET_S - $(date +%s))); }
check_budget() { [[ $(left) -gt 0 ]] || { echo 'Wall budget exceeded'; exit 7; }; }
mkdir -p "$OUT/frozen/bin" "$OUT/tmp" "$OUT/cuda-cache"
export TMPDIR="$OUT/tmp" CUDA_CACHE_PATH="$OUT/cuda-cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cp -L "$BUILD/bin/"*.so* "$OUT/frozen/bin/"
cp "$BUILD/SOURCE" "$OUT/frozen/SOURCE"
git -C "$ROOT" rev-parse HEAD > "$OUT/frozen/AUDIT_SOURCE"
build_source=$(head -n 1 "$BUILD/SOURCE")
git -C "$ROOT" diff --exit-code "$build_source" HEAD -- ggml/src ggml/include src common include >/dev/null
git -C "$ROOT" diff --exit-code -- ggml/src ggml/include src common include >/dev/null
cp "$ROOT/codex/state_probe.cpp" "$ROOT/codex/draft_probe.cpp" "$ROOT/codex/gpu_compare.py" "$OUT/frozen/"
cp "$ROOT/build-cpu/common/libllama-common-base.a" "$OUT/frozen/"
for probe in state draft; do
    timeout "$(left)" nice -n 19 g++ -std=c++17 -O2 -DS4_PARENT_GPU \
        -I"$ROOT/common" -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/ggml/include" -I"$ROOT/vendor" \
        "$OUT/frozen/${probe}_probe.cpp" "$OUT/frozen/bin/libllama-common.so" \
        "$OUT/frozen/bin/libllama.so" "$OUT/frozen/bin/libggml.so" \
        "$OUT/frozen/bin/libggml-cpu.so" "$OUT/frozen/bin/libggml-base.so" \
        "$OUT/frozen/libllama-common-base.a" -Wl,-rpath,'$ORIGIN' \
        -Wl,-rpath-link,"$OUT/frozen/bin" -o "$OUT/frozen/bin/audit-$probe"
done
(cd "$OUT/frozen" && nice -n 19 sha256sum SOURCE AUDIT_SOURCE *.cpp *.py *.a bin/* > SHA256SUMS)
inputs=("$MODEL" "$SUBSET")
unset D_AUDIT_TOKENS
if [[ -n ${TOKENS:-} ]]; then
    cp "$TOKENS" "$OUT/frozen/tokens.json"
    inputs+=("$OUT/frozen/tokens.json")
    export D_AUDIT_TOKENS="$OUT/frozen/tokens.json"
fi
timeout "$(left)" nice -n 19 sha256sum "${inputs[@]}" > "$OUT/INPUTS.sha256"
printf '%s\n' "CASE=$CASE PROBE=$PROBE PREFIX_N=$PREFIX_N BUDGET_S=$BUDGET_S" \
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
    log=$(journalctl -k -b --no-pager) || { echo 'Kernel log unreadable'; return 3; }
    [[ $log != *Xid* && $log != *'fallen off'* ]] || { echo 'Xid/fallen-off detected'; return 3; }
}
idle() {
    [[ -z $(pgrep -x llama-server || true) && -z $(pgrep -x weekend-srv || true) ]] || { echo 'Serving process running'; return 4; }
    local usage found=0 uuid used utilization
    usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits) || return 4
    while IFS=, read -r uuid used utilization; do
        [[ ,$UUIDS, == *",$uuid,"* ]] || continue
        found=$((found + 1))
        [[ ${used// /} -le 1500 ]] || { echo "Device busy: $uuid $used MiB"; return 4; }
        [[ ${utilization// /} == 0 ]] || { echo "Device active: $uuid $utilization percent"; return 4; }
    done <<< "$usage"
    [[ $found == 2 ]] || { echo 'Expected devices not found'; return 4; }
}
# Every tested knob starts from its default; inherited shell knobs must not contaminate an arm.
unset GGML_CUDA_FA_MMA_Q8_0_NCOLS1 GGML_CUDA_INPUT_BATCH GGML_CUDA_DISABLE_GRAPHS \
    LLAMA_GRAPH_REUSE_DISABLE GGML_CUDA_DISABLE_FUSION GGML_CUDA_PDL \
    GGML_CUDA_AR_BF16_THRESHOLD GGML_CUDA_AR_COPY_THRESHOLD GGML_CUDA_AR_COPY_CHUNK_BYTES \
    S4_CORRUPT_Q D_AUDIT_HOST_TARGET D_AUDIT_UBATCH D_AUDIT_FA_OFF D_AUDIT_K D_AUDIT_FULL_ONLY D_AUDIT_KV_F16
export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_FA_MMA_Q8_0=1 GGML_CUDA_GRAPH_OPT=0
export CUDA_VISIBLE_DEVICES=$UUIDS LD_LIBRARY_PATH="$OUT/frozen/bin"
for i in "${!ARMS[@]}"; do
    check_budget
    xid
    idle
    arm=${ARMS[$i]}
    run="$OUT/$i-$arm"
    mkdir -p "$run/logits"
    timeout "$(left)" bash -c 'cd "$1" && nice -n 19 sha256sum -c --quiet SHA256SUMS' _ "$OUT/frozen"
    timeout "$(left)" nice -n 19 sha256sum -c --quiet "$OUT/INPUTS.sha256"
    knobs=("${UP[@]}"); [[ $arm == patch ]] && knobs=("${PATCH[@]}")
    if [[ $PROBE == state ]]; then rows=(0); else rows=(4 5 6); fi
    status=0
    for row in "${rows[@]}"; do
        dir="$run/logits"; [[ $PROBE == draft ]] && dir="$run/rows-$row"
        mkdir -p "$dir"
        env "${knobs[@]}" D_AUDIT_PREFIX_N="$PREFIX_N" S4_ROWS="$row" \
            S4_SUBSET="$SUBSET" S4_LOGITS_DIR="$dir" S4_DIAGNOSTIC=1 \
            nice -n 19 "$OUT/frozen/bin/audit-$PROBE" "$MODEL" gpu 3 deferred \
            > "$run/probe-$row.log" 2>&1 &
        PID=$!
        while kill -0 "$PID" 2>/dev/null; do
            check_budget
            [[ $(ps -o stat= -p "$PID" 2>/dev/null || true) != Z* ]] || break
            sleep 1
        done
        wait "$PID" || status=$?; PID=
        xid
        [[ $status == 0 ]] || { echo "$status" > "$run/status"; exit 2; }
        if [[ $PROBE == draft ]]; then
            for f in "$dir/"*.f32; do cp "$f" "$run/logits/rows-$row-$(basename "$f")"; done
        fi
    done
    echo "$status" > "$run/status"
done
check_budget
timeout "$(left)" nice -n 19 python3 "$OUT/frozen/gpu_compare.py" "$OUT" | tee "$OUT/summary.txt"
