#!/usr/bin/env bash
# Parent only. Plan mode does not load CUDA libraries or query devices.
set -euo pipefail
ROOT=$(cd "$(dirname "$(realpath "$0")")/.." && pwd)
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VERIFY_SWAP_ROWS VERIFY_INDEPENDENT
PLAN=${PLAN:-1}
FROZEN=${FROZEN:-/home/xanewok/repos/qwen38-weekend/frozen/stack2}
MODEL=${MODEL:-/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf}
SUBSET=${SUBSET:-/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json}
CPU_BUILD=${CPU_BUILD:-/home/xanewok/repos/s4-runaway/stack2/build-cpu}
OUT=${OUT:-$ROOT/build-gpu-verify}
BUDGET_S=${BUDGET_S:-1800}
RS=${RS:-campaign}
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
if [[ $PLAN != 0 ]]; then
    printf '%s\n' 'PLAN: parent-only long-history logits probe, frozen stack2 0dba27b90.' \
        'Same binary; up/patch/patch/up = MMA knob 0/1/1/0, independent process boots.' \
        'Matrix: tensor/layer x q8_0/f16 target KV x subset/full draft head; draft KV f16.' \
        'Each boot: all six reconstructed histories, onset and available depths 20000/50000/90000.' \
        'Each point: n=1 reference and replay control, verify k=3/4/5, teacher-forced MTP chain.' \
        'Target RS uses the archived campaign value per history; draft RS=0, n_ctx=102400.' \
        'RS=3/4/5 can override the target snapshot layout for a separate bounded queue item.' \
        'Tool schemas and original token IDs are unavailable; histories are approximate.' \
        'Missing rows/boots and unresolved noise remain explicit on budget exhaustion.' \
        'Checks: frozen SHA256SUMS, source and model identity, idle devices, Xid, exact child PID cleanup.' \
        "Wall budget including preparation, freeze and analysis: $BUDGET_S s (maximum 1800)." \
        "Output: $OUT" \
        'Execute only in parent: RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_ab.sh'
    exit 0
fi
[[ ${RUN_PARENT_GPU:-0} == 1 ]] || { echo 'Parent execution flag required'; exit 2; }
[[ $BUDGET_S =~ ^[0-9]+$ && $BUDGET_S -gt 0 && $BUDGET_S -le 1800 ]] || { echo 'Budget must be within 1800 seconds'; exit 2; }
[[ $RS == campaign || $RS == 3 || $RS == 4 || $RS == 5 ]] || { echo 'RS must be campaign, 3, 4 or 5'; exit 2; }
[[ $OUT == "$ROOT"/build-* && ! -e $OUT ]] || { echo 'Use a fresh build-* OUT inside this clone'; exit 2; }
T0=$(date +%s)
# Reserve time for exact-PID shutdown and partial-result analysis.
DEADLINE=$((T0 + BUDGET_S - 120))
[[ $BUDGET_S -ge 150 ]] || { echo 'Budget is too small'; exit 2; }
PID=
stop() {
    [[ -n $PID ]] || return 0
    if kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        for _ in {1..5}; do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null || true
    fi
    wait "$PID" 2>/dev/null || true
    PID=
}
trap stop EXIT
trap 'exit 130' INT TERM
bounded() {
    local status=0
    "$@" &
    PID=$!
    while kill -0 "$PID" 2>/dev/null; do
        [[ $(date +%s) -lt $DEADLINE ]] || { stop; return 7; }
        [[ $(ps -o stat= -p "$PID" 2>/dev/null || true) != Z* ]] || break
        sleep 1
    done
    wait "$PID" || status=$?
    PID=
    return "$status"
}
xid() {
    local log
    log=$(timeout 5 nice -n 19 journalctl -k -b --no-pager) || { echo 'Cannot read kernel log'; return 3; }
    [[ $log != *Xid* && $log != *'fallen off'* ]] || { echo 'Xid/fallen-off detected'; return 3; }
}
idle() {
    [[ -z $(pgrep -x llama-server || true) && -z $(pgrep -x weekend-srv || true) ]] || { echo 'Serving process exists'; return 4; }
    local usage processes found=0 uuid used
    usage=$(timeout 5 nice -n 19 nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits) || return 4
    processes=$(timeout 5 nice -n 19 nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits) || return 4
    while IFS=, read -r uuid used; do
        [[ $UUIDS == *"$uuid"* ]] || continue
        found=$((found + 1))
        [[ ${used// /} -le 1500 ]] || { echo "Busy device: $uuid $used MiB"; return 4; }
        [[ $processes != *"$uuid"* ]] || { echo "Compute process on $uuid"; return 4; }
    done <<< "$usage"
    [[ $found == 2 ]] || { echo 'Expected devices missing'; return 4; }
}
[[ $(head -n1 "$FROZEN/SOURCE") == 0dba27b9002b8347c6f14a9f4a1651b3478065ef ]] || { echo 'Wrong source'; exit 6; }
[[ $(git -C "$ROOT" rev-parse 0dba27b90) == $(head -n1 "$FROZEN/SOURCE") ]] || exit 6
git -C "$ROOT" diff --quiet 0dba27b90 -- common include src ggml || { echo 'Headers/source differ from frozen stack2'; exit 6; }
rg -q '^GGML_CUDA:BOOL=OFF$' "$CPU_BUILD/CMakeCache.txt"
rg -q '^GGML_CPU_REPACK:BOOL=OFF$' "$CPU_BUILD/CMakeCache.txt"
mkdir -p "$OUT/frozen/bin" "$OUT/inputs" "$OUT/tmp" "$OUT/cuda-cache" "$ROOT/build-cpu/common"
export TMPDIR="$OUT/tmp" CUDA_CACHE_PATH="$OUT/cuda-cache"
bounded nice -n 19 bash -c 'cd "$1/build/bin"; sha256sum -c --quiet SHA256SUMS' _ "$FROZEN"
cp "$FROZEN/build/bin/"*.so* "$OUT/frozen/bin/"
cp "$FROZEN/SOURCE" "$OUT/frozen/SOURCE"
cp "$CPU_BUILD/common/libllama-common-base.a" "$ROOT/build-cpu/common/"
cp "$ROOT/codex/probe.cpp" "$OUT/frozen/probe.cpp"
cp "$SUBSET" "$OUT/inputs/draft-vocab.json"
# Pin the model against the archived campaign launch, not only its current size.
bounded nice -n 19 sha256sum "$MODEL" "$SUBSET" > "$OUT/INPUTS.sha256"
[[ $(head -n1 "$OUT/INPUTS.sha256" | cut -d' ' -f1) == e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df ]] || { echo 'Wrong Swift model'; exit 6; }
bounded nice -n 19 bash "$ROOT/codex/build_probe.sh" "$CPU_BUILD" "$OUT/verify-prepare" 0
bounded nice -n 19 python3 "$ROOT/codex/reconstruct.py" --out "$OUT/inputs" > "$OUT/extraction.json"
for file in "$OUT/inputs/"*.history.json; do
    bounded nice -n 19 "$OUT/verify-prepare" prepare "$MODEL" "$file" "${file%.history.json}.tokens.json" >> "$OUT/preparation.log" 2>&1
done
bounded nice -n 19 python3 - "$OUT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
histories = [json.loads(f.read_text()) for f in sorted((p / 'inputs').glob('*.tokens.json'))]
# Begin each boot with the observed onsets; later depths use the same context and real continuation.
histories.sort(key=lambda d: (not d['name'].startswith('S4'), d['name']))
(p / 'inputs/histories.json').write_text(json.dumps(histories))
configs = [(split, kv, head) for split, kv in [('tensor','q8_0'), ('layer','q8_0'), ('tensor','f16'), ('layer','f16')]
           for head in ['subset', 'full']]
schedule = []
for split, kv, head in configs:
    for fa in [0, 1, 1, 0]:
        config = f'{split}-{kv}-{fa}-{head}'
        schedule.append(dict(name=f'{len(schedule):02d}-{config}', config=config, split=split, kv=kv, fa=fa, head=head))
(p / 'schedule.json').write_text(json.dumps(schedule, indent=2))
(p / 'expected.json').write_text(json.dumps([s['name'] for s in schedule]))
PY
# Compile against the copied libraries; the source and binary are then frozen together.
bounded env PROBE_SOURCE="$OUT/frozen/probe.cpp" nice -n 19 bash "$ROOT/codex/build_probe.sh" "$OUT/frozen" "$OUT/frozen/bin/verify-gpu-probe" 1
cp "$ROOT/codex/compare.py" "$OUT/frozen/compare.py"
bounded nice -n 19 bash -c 'cd "$1"; sha256sum SOURCE probe.cpp compare.py bin/* > SHA256SUMS' _ "$OUT/frozen"
bounded nice -n 19 bash -c 'cd "$1"; sha256sum inputs/* > HISTORIES.sha256' _ "$OUT"
export CUDA_VISIBLE_DEVICES=$UUIDS LD_LIBRARY_PATH="$OUT/frozen/bin"
xid
idle
status=0
while IFS=$'\t' read -r name config split kv fa head; do
    [[ $(date +%s) -lt $DEADLINE ]] || { status=7; break; }
    bounded nice -n 19 bash -c 'cd "$1"; sha256sum -c --quiet SHA256SUMS' _ "$OUT/frozen"
    bounded nice -n 19 bash -c 'cd "$1"; sha256sum -c --quiet HISTORIES.sha256' _ "$OUT"
    bounded nice -n 19 sha256sum -c --quiet "$OUT/INPUTS.sha256"
    xid
    idle
    run="$OUT/boots/$name"
    mkdir -p "$run"
    nice -n 19 python3 - "$OUT/schedule.json" "$name" "$run/boot.json" <<'PY'
import json, sys
s = next(s for s in json.load(open(sys.argv[1])) if s['name'] == sys.argv[2])
json.dump(s, open(sys.argv[3], 'w'), indent=2)
PY
    ids=-; [[ $head == full ]] || ids="$OUT/inputs/draft-vocab.json"
    result=0
    bounded env GGML_CUDA_FA_MMA_Q8_0="$fa" nice -n 19 "$OUT/frozen/bin/verify-gpu-probe" run \
        "$MODEL" "$OUT/inputs/histories.json" "$run" "$split" "$kv" "$ids" "$RS" > "$run/probe.log" 2>&1 || result=$?
    printf '%s\n' "$result" > "$run/status"
    xid
    if [[ $result == 0 ]]; then touch "$run/COMPLETE"; else status=$result; break; fi
done < <(nice -n 19 python3 - "$OUT/schedule.json" <<'PY'
import json, sys
for s in json.load(open(sys.argv[1])):
    print('\t'.join(str(s[k]) for k in ['name','config','split','kv','fa','head']))
PY
)
# CPU analysis still runs with devices hidden. Timeout preserves raw artifacts and an explicit failure.
export CUDA_VISIBLE_DEVICES=''
DEADLINE=$((T0 + BUDGET_S - 7))
analysis=0
bounded nice -n 19 python3 "$OUT/frozen/compare.py" "$OUT" > "$OUT/summary.json" 2> "$OUT/analysis.error" || analysis=$?
[[ $analysis == 0 ]] || { echo 'Analysis incomplete; raw rows retained'; exit "$analysis"; }
cat "$OUT/summary.json"
exit "$status"
