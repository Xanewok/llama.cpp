#!/usr/bin/env bash
# Parent queue only. This script executes GPU code; Codex must not run it.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
UP_DIR=${UP_DIR:-$ROOT/codex/frozen/up/bin}
PATCH_DIR=${PATCH_DIR:-$ROOT/codex/frozen/patch/bin}
OUT=${OUT:-$ROOT/codex/gpu-ab-$(date +%Y%m%d-%H%M%S)}
DRIVE=$ROOT/codex/gpu_request.py
UP_COMMIT=68d99c67d958c28b6bff5fdc2db096f81c324ef9
PATCH_COMMIT=f46f8bbdb8a6b73eb6ddeabad49f24534e70e0cd
MODEL=$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
TEMPLATE=$HOME/repos/qwen38-eval/results/session2/pair-5060ti-2060s/campaign-swift-fullsuite-20260915/swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja
PORT=${PORT:-8096}
BUDGET_S=${BUDGET_S:-1200}
ARMS=(up patch patch up up patch)
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
PID=
T0=$SECONDS
left() { echo $((BUDGET_S - SECONDS + T0)); }
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) && [[ $s != Z* ]]; }
stop_server() {
    if [[ -n $PID ]]; then
        kill -TERM "$PID" 2>/dev/null || true
        for ((i=0; i<30; i++)); do alive "$PID" || break; sleep 1; done
        if alive "$PID"; then kill -KILL "$PID" 2>/dev/null || true; fi
        wait "$PID" 2>/dev/null || true
        PID=
    fi
}
xid() {
    local journal
    journal=$(journalctl -k -b --no-pager) || return 3
    [[ -n $journal && $journal != *'-- No entries --'* ]] || return 3
    if [[ $journal =~ Xid|fallen.off ]]; then return 3; fi
}
trap stop_server EXIT
trap 'exit 130' INT TERM
mkdir -p "$OUT"
[[ ! -e $OUT/summary.txt ]] || exit 6
for dir in "$UP_DIR" "$PATCH_DIR"; do
    [[ -f $dir/input-upload-srv && -f $dir/SHA256SUMS && -f $dir/../SOURCE ]] || exit 6
    (cd "$dir" && sha256sum -c --quiet SHA256SUMS) || exit 6
    cat "$dir/../SOURCE"
done
[[ $(head -1 "$UP_DIR/../SOURCE") = "$UP_COMMIT" ]] || exit 6
[[ $(head -1 "$PATCH_DIR/../SOURCE") = "$PATCH_COMMIT" ]] || exit 6
[[ $(stat -c %s "$MODEL") = 15475951296 ]] || exit 6
[[ -f $DRIVE && -f $TEMPLATE ]] || exit 6
CMD=(-m "$MODEL" -np 1 -fa on --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0
     -b 2048 -ub 128 --jinja --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --host 127.0.0.1 --port "$PORT"
     -c 32768 -ctk q8_0 -ctv q8_0 -ngl 99 --spec-draft-type-k f16 --spec-draft-type-v f16 -sm tensor --tensor-split 50,50
     -t 8 --chat-template-file "$TEMPLATE" --metrics --spec-draft-backend-sampling --spec-draft-sampling probabilistic)
if [[ ${DRY_RUN:-0} = 1 ]]; then
    printf 'Boots: %s\n' "${ARMS[*]}"
    printf 'Command: %q ' "${CMD[@]}"; printf '\n'
    exit 0
fi
export CUDA_VISIBLE_DEVICES=$UUIDS
xid
if pgrep -x llama-server >/dev/null || ss -Hltn "sport = :$PORT" | rg -q .; then exit 4; fi
used=$(nvidia-smi --id="$UUIDS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')
[[ $used -le 1500 ]] || exit 4
declare -A boots=()
for arm in "${ARMS[@]}"; do
    [[ $(left) -gt 320 ]] || exit 7
    boots[$arm]=$(( ${boots[$arm]:-0} + 1 ))
    tag=$arm-${boots[$arm]}
    dir=$UP_DIR; [[ $arm = up ]] || dir=$PATCH_DIR
    LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" nice -n 19 "$dir/input-upload-srv" "${CMD[@]}" > "$OUT/$tag.server.log" 2>&1 &
    PID=$!
    healthy=0
    for ((i=0; i<180; i++)); do
        [[ $(left) -gt 0 ]] || exit 7
        if curl -sf -m 2 "http://127.0.0.1:$PORT/health" | rg -q '"ok"'; then healthy=1; break; fi
        alive "$PID" || break
        sleep 1
    done
    [[ $healthy = 1 ]] || exit 2
    timeout "$(left)" nice -n 19 python3 "$DRIVE" "$PORT" fresh 1 128 1 > "$OUT/$tag-warmup.run.json"
    timeout "$(left)" nice -n 19 python3 "$DRIVE" "$PORT" follow 1 384 0 > "$OUT/$tag-tool.run.json"
    timeout "$(left)" nice -n 19 python3 "$DRIVE" "$PORT" follow 1 1024 1 > "$OUT/$tag-think.run.json"
    stop_server
    xid
done
nice -n 19 python3 - "$OUT" <<'PY' | tee "$OUT/summary.txt"
import glob, json, os, sys
values = {}
identities = {}
for path in sorted(glob.glob(sys.argv[1] + '/*-*.run.json')):
    if 'warmup' in path:
        continue
    row = json.loads(open(path).read().strip().splitlines()[-1])
    t = row['timings']
    arm, boot, request = os.path.basename(path).split('.')[0].split('-')
    rounds = t['draft_n'] / 3
    if rounds <= 0:
        raise SystemExit('No draft rounds: ' + path)
    ms = t['predicted_ms'] / rounds
    values.setdefault((arm, request), []).append(ms)
    identity = (row['completion_sha256'], t['predicted_n'], t['draft_n'], t.get('draft_n_accepted'))
    identities.setdefault(request, set()).add(identity)
    print(f'M {arm} {boot} {request}: {ms:.4f} ms/round; accepted {t.get("draft_n_accepted")}/{t["draft_n"]}')
for request in ('tool', 'think'):
    if len(identities[request]) != 1:
        raise SystemExit('Completion or acceptance changed: ' + request)
    up, patch = values['up', request], values['patch', request]
    if len(up) != 3 or len(patch) != 3:
        raise SystemExit('Incomplete boots')
    delta = sum(patch)/len(patch) - sum(up)/len(up)
    print(f'D {request}: patch-up {delta:.4f} ms/round (draft_n/3 estimates round count)')
PY
[[ $(left) -gt 0 ]] || exit 7
