#!/usr/bin/env bash
# Parent-only; DRY_RUN=1 never accesses CUDA. RUN_GPU_AB=1 DRY_RUN=0 opts into the queued experiment.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T0=$SECONDS
SP=/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad
STACK_COMMIT=b8668a92e5188c8294e4e133f4426a3506fac50c
UP_DIR=${UP_DIR:-$ROOT/codex/frozen/up/bin}
STACK_DIR=${STACK_DIR:-$ROOT/codex/frozen/stack/bin}
MODEL=${MODEL:-$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf}
VOCAB=${VOCAB:-$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json}
TEMPLATE=${TEMPLATE:-$HOME/repos/qwen38-eval/results/session2/pair-5060ti-2060s/campaign-swift-fullsuite-20260915/swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja}
SMOKE=$SP/prefilter2/codex_smoke.py
OUT=${OUT:-$ROOT/codex/gpu-ab-results}
PORT=${PORT:-8096}
BUDGET_S=${BUDGET_S:-2400}
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
ARMS=(up full sub3 sub4 sub5 sub5 sub4 sub3 full up)
CMD=(-m "$MODEL" -np 1 -fa on -fit off --spec-type draft-mtp --spec-draft-n-min 0 --spec-draft-p-min 0
     -b 2048 -ub 128 --jinja --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --host 127.0.0.1 --port "$PORT"
     -c 32768 -ctk q8_0 -ctv q8_0 -ngl 99 --spec-draft-type-k f16 --spec-draft-type-v f16 -sm tensor --tensor-split 50,50
     --chat-template-file "$TEMPLATE" --metrics --spec-draft-sampling probabilistic --spec-draft-backend-sampling -t 8 -tb 8)
(( BUDGET_S > 120 && BUDGET_S <= 2400 )) || exit 7
bounded() {
    local remaining=$((BUDGET_S - SECONDS + T0 - 35))
    (( remaining > 0 )) || return 7
    timeout "$remaining" nice -n 19 "$@"
}
for dir in "$UP_DIR" "$STACK_DIR"; do
    [[ -x $dir/vocab-ab && -f $dir/SHA256SUMS && -f $dir/../SOURCE ]] || { echo "Missing frozen build: $dir"; exit 6; }
    rg -q ' [ *]?vocab-ab$' "$dir/SHA256SUMS" || exit 6
    (cd "$dir" && bounded sha256sum -c --quiet SHA256SUMS)
    [[ $(head -1 "$dir/../SOURCE") =~ ^[0-9a-f]{40}$ ]] || exit 6
    head -1 "$dir/../SOURCE"
done
[[ $(head -1 "$UP_DIR/../SOURCE") == 68d99c67d958c28b6bff5fdc2db096f81c324ef9 ]] || exit 6
[[ $(head -1 "$STACK_DIR/../SOURCE") == "$STACK_COMMIT" ]] || exit 6
[[ $(stat -c %s "$MODEL") == 15475951296 && -f $VOCAB && -f $TEMPLATE && -f $SMOKE ]] || exit 6
if [[ ${DRY_RUN:-1} == 1 ]]; then
    for arm in "${ARMS[@]}"; do
        k=3; [[ $arm != sub4 && $arm != sub5 ]] || k=${arm#sub}
        dir=$STACK_DIR; [[ $arm != up ]] || dir=$UP_DIR
        extra=(); [[ $arm == up ]] || extra+=(--backend-sampling)
        [[ $arm != sub* ]] || extra+=(--spec-draft-vocab "$VOCAB")
        printf 'DRY_RUN: %s: %s/vocab-ab %s --spec-draft-n-max %s %s\n' "$arm" "$dir" "${CMD[*]}" "$k" "${extra[*]}"
    done
    echo 'DRY_RUN: each boot: warmup, tool seed 1, think seed 1, approximately 8k-token prompt seed 1; two boots per arm.'
    exit 0
fi
[[ ${RUN_GPU_AB:-0} == 1 ]] || { echo 'Parent must set RUN_GPU_AB=1 DRY_RUN=0'; exit 4; }
renice -n 19 -p "$$" >/dev/null
export CUDA_VISIBLE_DEVICES=$UUIDS
mkdir -p "$OUT"
[[ -z $(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || { echo "Refuse nonempty OUT: $OUT"; exit 6; }
bounded sha256sum "$MODEL" "$VOCAB" "$TEMPLATE" "$SMOKE" "$ROOT/codex/gpu_request.py" > "$OUT/input.sha256"
PID=
stop_server() {
    [[ -n $PID ]] || return 0
    if kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        for ((i=0; i<30; ++i)); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$PID" 2>/dev/null; then kill -KILL "$PID" 2>/dev/null || true; fi
    fi
    wait "$PID" 2>/dev/null || true
    PID=
}
trap stop_server EXIT
trap 'exit 130' INT TERM
xid_check() {
    bounded journalctl -k -b --no-pager > "$OUT/kernel.log" || return 3
    if rg -qi 'Xid|fallen off' "$OUT/kernel.log"; then return 3; fi
}
busy_check() {
    if pgrep -x llama-server >/dev/null || pgrep -x vocab-ab >/dev/null; then return 4; fi
    if [[ -n $(ss -Hltn "sport = :$PORT") ]]; then return 4; fi
    local used
    used=$(timeout 10 nice -n 19 nvidia-smi --id="$UUIDS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')
    [[ $used =~ ^[0-9]+$ && $used -le 1500 ]]
}
request() {
    local tag=$1 kind=$2 remaining
    remaining=$((BUDGET_S - SECONDS + T0 - 35))
    (( remaining > 0 )) || return 7
    timeout "$remaining" nice -n 19 python3 "$ROOT/codex/gpu_request.py" "$SMOKE" "$PORT" "$kind" "$OUT" "$remaining" > "$OUT/$tag-$kind.json"
}
declare -A NBOOT=([up]=0 [full]=0 [sub3]=0 [sub4]=0 [sub5]=0)
for arm in "${ARMS[@]}"; do
    xid_check
    busy_check
    (( SECONDS - T0 < BUDGET_S - 200 )) || exit 7
    NBOOT[$arm]=$((NBOOT[$arm] + 1))
    tag=$arm-${NBOOT[$arm]}
    dir=$STACK_DIR; [[ $arm != up ]] || dir=$UP_DIR
    k=3; [[ $arm != sub4 && $arm != sub5 ]] || k=${arm#sub}
    extra=(); [[ $arm == up ]] || extra+=(--backend-sampling)
    [[ $arm != sub* ]] || extra+=(--spec-draft-vocab "$VOCAB")
    (cd "$dir" && bounded sha256sum -c --quiet SHA256SUMS)
    LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" nice -n 19 "$dir/vocab-ab" "${CMD[@]}" --spec-draft-n-max "$k" "${extra[@]}" > "$OUT/$tag.server.log" 2>&1 &
    PID=$!
    healthy=0
    for ((poll=0; poll<300; ++poll)); do
        (( SECONDS - T0 < BUDGET_S - 35 )) || exit 7
        kill -0 "$PID" 2>/dev/null || exit 2
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" | rg -q '"ok"'; then healthy=1; break; fi
        sleep 1
    done
    (( healthy )) || exit 2
    for kind in warmup tool think long; do request "$tag" "$kind"; done
    stop_server
    if rg -qi 'backend offload failed|backend sampling.*using CPU|backend sampling.*needs' "$OUT/$tag.server.log"; then exit 2; fi
    if [[ $arm == sub* ]]; then rg -q 'reduced draft head: 25879 tokens' "$OUT/$tag.server.log" || exit 2; fi
    xid_check
done
bounded sha256sum -c --quiet "$OUT/input.sha256"
bounded python3 "$ROOT/codex/gpu_summary.py" "$OUT" | tee "$OUT/summary.txt"
(( SECONDS - T0 <= BUDGET_S )) || exit 7
