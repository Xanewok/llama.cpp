#!/usr/bin/env bash
# Parent only. PLAN=1 does not discover GPUs or run binaries.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PLAN=${PLAN:-1}
BUDGET_S=${BUDGET_S:-7200}
OLD=${OLD:-$ROOT/build-gpu-ab-20261001-110357/frozen}
NEW=$ROOT/build-cuda/bin
UUIDS=${UUIDS:-GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f}
if [[ $PLAN != 0 ]]; then
    printf '%s\n' 'PLAN: frozen old/new binaries and libraries; verify manifests before each process.' \
        'Both GPUs: old,new,new,old boots; old expects 332/362 with exactly 30 large inf failures, new 386/386 including large nb 1/2/7.' \
        'Per boot: nsys kernel timings for nb 4/8, KV 8192/32768/98304, f16/q8_0, direct q8 staging enabled.' \
        'Record median/mean MMA us, fixup us, registers, dynamic shared memory and launch geometry.' \
        'Negative control: corrupt new output and require failure. Require idle GPUs and clean, readable Xid log before/after runs.' \
        "Budget: $BUDGET_S s. Exact owned PID cleanup. No server starts." \
        'Parent invocation: RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_accum_ab.sh'
    exit 0
fi
[[ ${RUN_PARENT_GPU:-0} == 1 ]] || { echo 'REFUSE: RUN_PARENT_GPU=1 required'; exit 4; }
OUT=${OUT:-$ROOT/build-gpu-accum-ab-$(date +%Y%m%d-%H%M%S)}
case "$OUT" in "$ROOT"/build-*) ;; *) echo 'REFUSE: OUT outside worktree build directory'; exit 4;; esac
[[ ! -e $OUT ]] || { echo 'REFUSE: OUT exists'; exit 4; }
export CUDA_VISIBLE_DEVICES="$UUIDS" OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=8 GGML_CUDA_FA_MMA_Q8_0=1
unset LLAMA_TEST_SWIFT_FA LLAMA_TEST_SWIFT_FA_CORRUPT LLAMA_TEST_SWIFT_FA_ACCUM
(cd "$OLD" && sha256sum -c --quiet SHA256SUMS)
(cd "$ROOT" && sha256sum -c --quiet codex/cuda-accum.sha256)
command -v nsys >/dev/null || { echo 'REFUSE: nsys missing'; exit 6; }
pgrep -x llama-server >/dev/null && { echo 'REFUSE: existing llama-server'; exit 4; }
xid_check() {
    local log
    log=$(journalctl -k -b --no-pager) || { echo 'REFUSE: unreadable kernel log'; return 3; }
    if grep -qiE 'NVRM:.*Xid|fallen off' <<< "$log"; then echo 'REFUSE: Xid/fallen off'; return 3; fi
}
idle_check() {
    local values mem util count=0
    values=$(nvidia-smi -i "$UUIDS" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    while IFS=, read -r mem util; do
        [[ $mem =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && $util =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || return 4
        (( mem <= 750 && util == 0 )) || { echo 'REFUSE: GPUs busy'; return 4; }
        count=$((count + 1))
    done <<< "$values"
    [[ $count == 2 && -z $(nvidia-smi -i "$UUIDS" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || return 4
}
wait_idle() {
    for _ in {1..15}; do
        idle_check >/dev/null && return 0
        sleep 1
    done
    echo 'REFUSE: GPUs did not become idle after owned process exit'
    return 4
}
xid_check
idle_check
mkdir -p "$OUT/frozen/old" "$OUT/frozen/new"
cp "$OLD/swift-fa-gpu" "$OUT/frozen/old/swift-fa-gpu"
cp -L "$OLD"/lib*.so* "$OUT/frozen/old/"
cp "$NEW/test-backend-ops" "$OUT/frozen/new/swift-fa-gpu"
cp -L "$NEW"/lib*.so* "$OUT/frozen/new/"
cp "$ROOT/codex/gpu_accum_ab.sh" "$ROOT/codex/accum_gpu_results.py" "$OUT/frozen/"
cp "$OLD/SOURCE" "$OUT/frozen/OLD-SOURCE"
git -C "$ROOT" rev-parse HEAD > "$OUT/frozen/NEW-SOURCE"
git -C "$ROOT" diff --binary > "$OUT/frozen/LOCAL.diff"
(cd "$OUT/frozen" && sha256sum old/* new/* gpu_accum_ab.sh accum_gpu_results.py OLD-SOURCE NEW-SOURCE LOCAL.diff > SHA256SUMS)
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
run() {
    local tag=$1; shift
    local left=$(( BUDGET_S - (SECONDS - START) )) rc=0
    (( left > 0 )) || { echo 'FAIL: wall budget'; exit 7; }
    (cd "$OUT/frozen" && sha256sum -c --quiet SHA256SUMS)
    xid_check
    idle_check
    timeout --signal=TERM --kill-after=5 "$left" nice -n 19 "$@" > "$OUT/$tag.log" 2>&1 &
    PID=$!
    wait "$PID" || rc=$?
    PID=
    xid_check
    wait_idle
    (( rc != 124 && rc != 137 )) || { echo 'FAIL: wall budget'; exit 7; }
    RUN_RC=$rc
}
for gpu in CUDA0 CUDA1; do
    boot=0
    for arm in old new new old; do
        boot=$((boot + 1))
        tag=$gpu-boot$boot-$arm
        dir=$OUT/frozen/$arm
        bin=$dir/swift-fa-gpu
        extra=()
        [[ $arm == new ]] && extra=(LLAMA_TEST_SWIFT_FA_ACCUM=1)
        run "$tag-suite" env LD_LIBRARY_PATH="$dir" LLAMA_TEST_SWIFT_FA=1 "${extra[@]}" "$bin" test -b "$gpu" -o FLASH_ATTN_EXT -j 1
        if [[ $arm == old ]]; then [[ $RUN_RC == 1 ]]; else [[ $RUN_RC == 0 ]]; fi
        python3 "$OUT/frozen/accum_gpu_results.py" suite "$OUT/$tag-suite.log" --arm "$arm" >> "$OUT/results.jsonl"
        for kv in 8192 32768 98304; do
            for nb in 4 8; do
                for type in f16 q8_0; do
                    perf=$tag-kv$kv-nb$nb-$type
                    filter="^hsk=256,hsv=256,nh=2,nr23=\[6,1\],kv=$kv,nb=$nb,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=$type,type_V=$type,permute=\[0,2,1,3\],kv_view=0,"
                    run "$perf" nsys profile --trace=cuda --sample=none --cpuctxsw=none --output="$OUT/$perf" \
                        env LD_LIBRARY_PATH="$dir" "$bin" perf -b "$gpu" -o FLASH_ATTN_EXT -p "$filter" -j 1
                    [[ $RUN_RC == 0 ]] || exit 1
                    run "$perf-export" nsys export --type=sqlite --output="$OUT/$perf.sqlite" "$OUT/$perf.nsys-rep"
                    [[ $RUN_RC == 0 ]] || exit 1
                    python3 "$OUT/frozen/accum_gpu_results.py" kernels "$OUT/$perf.sqlite" > "$OUT/$perf-kernels.jsonl"
                done
            done
        done
    done
    dir=$OUT/frozen/new
    run "$gpu-negative" env LD_LIBRARY_PATH="$dir" LLAMA_TEST_SWIFT_FA=1 LLAMA_TEST_SWIFT_FA_CORRUPT=1 \
        "$dir/swift-fa-gpu" test -b "$gpu" -o FLASH_ATTN_EXT -j 1 \
        -p 'kv=4096,nb=4,.*type_K=q8_0.*pattern=large'
    [[ $RUN_RC == 1 ]] && grep -q 'FAIL' "$OUT/$gpu-negative.log" || { echo 'FAIL: negative control'; exit 1; }
done
(cd "$OUT/frozen" && sha256sum -c --quiet SHA256SUMS)
echo 'PASS: expected old failures, clean new suite, kernel traces and corruption controls.'
