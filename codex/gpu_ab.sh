#!/usr/bin/env bash
# Parent only. No GPU discovery or binary execution occurs in plan mode.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PLAN=${PLAN:-1}
BUDGET_S=${BUDGET_S:-7200}
SANITIZER=${SANITIZER:-/usr/local/cuda-13.2/bin/compute-sanitizer}
UUIDS=${UUIDS:-GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f}
if [[ $PLAN != 0 ]]; then
    printf '%s\n' 'PLAN: freeze build-cuda/bin with SHA256SUMS; require two idle GPUs and readable clean Xid log.' \
        'Per GPU: up(FA_MMA_Q8_0=0), patch(1), patch(1), up(0); every boot runs the full Swift suite against CPU.' \
        'Per GPU and arm: memcheck, racecheck, initcheck, synccheck on query counts 5/6 at short/long KV.' \
        'Long comparison cases check batch MMA, single-row q8 VEC (aligned KV), and dequantized f16 KV on identical inputs.' \
        'Require every selected case OK, no unsupported/empty run, finite output, max head/query NMSE <= 5e-4.' \
        'Large constants are range stress, not a claim about model activation bounds. Any failure remains a failed suite.' \
        "Wall budget: $BUDGET_S s; exact owned PID cleanup; no server process starts." \
        'Execute only when parent queues it: RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_ab.sh'
    exit 0
fi
[[ ${RUN_PARENT_GPU:-0} == 1 ]] || { echo 'REFUSE: RUN_PARENT_GPU=1 required'; exit 4; }
OUT=${OUT:-$ROOT/build-gpu-ab-$(date +%Y%m%d-%H%M%S)}
case "$OUT" in "$ROOT"/build-*) ;; *) echo 'REFUSE: OUT must be a build-* directory in this worktree'; exit 4;; esac
[[ ! -e $OUT ]] || { echo 'REFUSE: OUT exists'; exit 4; }
export CUDA_VISIBLE_DEVICES="$UUIDS" OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=8 LLAMA_TEST_SWIFT_FA=1
unset LLAMA_TEST_SWIFT_FA_CORRUPT
(cd "$ROOT" && sha256sum -c --quiet codex/cuda-build.sha256) || { echo 'REFUSE: CUDA build/source manifest changed'; exit 6; }
[[ -x $SANITIZER ]] || { echo 'REFUSE: compute-sanitizer missing'; exit 6; }
[[ $(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$UUIDS" | wc -l) == 2 ]] || exit 4
pgrep -x llama-server >/dev/null && { echo 'REFUSE: existing llama-server'; exit 4; }
xid_check() {
    local j
    j=$(journalctl -k -b --no-pager) || { echo 'REFUSE: unreadable kernel log'; return 3; }
    if grep -qiE 'NVRM:.*Xid|fallen off' <<< "$j"; then echo 'REFUSE: Xid/fallen off'; return 3; fi
}
idle_check() {
    local values mem util
    values=$(nvidia-smi -i "$UUIDS" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    while IFS=, read -r mem util; do
        [[ $mem =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && $util =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || return 4
        (( mem <= 750 && util == 0 )) || { echo 'REFUSE: GPUs busy'; return 4; }
    done <<< "$values"
    [[ -z $(nvidia-smi -i "$UUIDS" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || return 4
}
xid_check
idle_check
mkdir -p "$OUT/frozen"
cp "$ROOT/build-cuda/bin/test-backend-ops" "$OUT/frozen/swift-fa-gpu"
cp -L "$ROOT"/build-cuda/bin/lib*.so* "$OUT/frozen/"
cp "$ROOT/codex/gpu_ab.sh" "$OUT/frozen/queue-script.sh"
git -C "$ROOT" rev-parse HEAD > "$OUT/frozen/SOURCE"
git -C "$ROOT" diff --binary > "$OUT/frozen/LOCAL.diff"
(cd "$OUT/frozen" && sha256sum swift-fa-gpu lib*.so* queue-script.sh SOURCE LOCAL.diff > SHA256SUMS)
export LD_LIBRARY_PATH="$OUT/frozen${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
START=$SECONDS
PID=
stop_owned() {
    if [[ -n $PID ]]; then
        kill -TERM "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=
    fi
}
trap stop_owned EXIT
trap 'exit 130' INT TERM
failures=0
run() {
    local tag=$1 expected=$2; shift 2
    local left=$(( BUDGET_S - (SECONDS - START) )) rc=0
    (( left > 0 )) || { echo 'FAIL: wall budget'; exit 7; }
    (cd "$OUT/frozen" && sha256sum -c --quiet SHA256SUMS) || exit 6
    idle_check
    timeout --signal=TERM --kill-after=5 "$left" nice -n 19 "$@" > "$OUT/$tag.log" 2>&1 &
    PID=$!
    wait "$PID" || rc=$?
    PID=
    xid_check
    if (( rc != 0 )) || [[ $(grep -c 'swift_verify=1.*OK' "$OUT/$tag.log") != $expected ]] || grep -qiE 'not supported|FAIL|NaN|inf mismatch|[1-9][0-9]* hazards|[1-9][0-9]* errors' "$OUT/$tag.log"; then
        echo "FAIL: $tag (rc=$rc)" | tee -a "$OUT/verdict.txt"
        failures=$((failures + 1))
    else
        echo "PASS: $tag" | tee -a "$OUT/verdict.txt"
    fi
    (( rc != 124 && rc != 137 )) || exit 7
}
BIN=$OUT/frozen/swift-fa-gpu
for gpu in CUDA0 CUDA1; do
    boot=0
    for arm in 0 1 1 0; do
        boot=$((boot + 1))
        run "$gpu-boot$boot-arm$arm" 362 env GGML_CUDA_FA_MMA_Q8_0="$arm" "$BIN" test -b "$gpu" -o FLASH_ATTN_EXT -j 1
    done
    for arm in 0 1; do
        for tool in memcheck racecheck initcheck synccheck; do
            run "$gpu-$tool-arm$arm" 40 env GGML_CUDA_FA_MMA_Q8_0="$arm" "$SANITIZER" --tool "$tool" --error-exitcode 9 \
                "$BIN" test -b "$gpu" -o FLASH_ATTN_EXT -j 1 \
                -p 'nr23=\[6,1\],kv=(4096|102400|102399),nb=(5|6),.*pattern=(causal|late|peak),compare_rows=0'
        done
    done
    idle_check
    left=$(( BUDGET_S - (SECONDS - START) ))
    (( left > 0 )) || exit 7
    timeout --kill-after=5 "$left" nice -n 19 env LLAMA_TEST_SWIFT_FA_CORRUPT=1 "$BIN" test -b "$gpu" -o FLASH_ATTN_EXT -j 1 \
        -p 'nr23=\[6,1\],kv=4096,nb=5,.*type_K=q8_0.*pattern=causal,compare_rows=0,used_kv=4096$' > "$OUT/$gpu-negative.log" 2>&1 &
    PID=$!
    rc=0; wait "$PID" || rc=$?; PID=
    xid_check
    [[ $rc == 1 ]] && grep -q 'FAIL' "$OUT/$gpu-negative.log" || { echo 'FAIL: negative control'; failures=$((failures + 1)); }
done
(cd "$OUT/frozen" && sha256sum -c --quiet SHA256SUMS)
(( failures == 0 )) || exit 1
echo 'PASS: all numerical cases, sanitizer tools, and negative controls.' | tee -a "$OUT/verdict.txt"
