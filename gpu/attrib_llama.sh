#!/usr/bin/env bash
# Attribution pack, llama.cpp arm: nsys captures of steady k=5 decode at 8k and ~90k plus one cold 32k prefill on the
# frozen stack2 binary with the S5 np1 flags, then attrib_pack.py -> budget.{json,md}. One boot, np1, 2x RTX 5060 Ti.
# Windows (gpu/llama_drive.py): prefill-32k, decode-8k, decode-90k, one nsys start/stop each. CANARY=1: decode-8k only.
# Deviation from S5: --cache-ram 0 (a host prompt-cache save must not land inside a window); no `nice`
# (host gaps are an axis being measured).
# Contract: only our own PIDs are stopped (nsys session by name, then the exact PID); the pack must accept every window
# (>= 40 steady rounds per decode, a cold prefill run) or the item fails with exit 8 after keeping all raw files.
# Output: $OUT/{<window>.nsys-rep,sqlite,runs.json,manifest.json,pack/budget.{json,md},server.log,console.log}.
# Exit: 0 ok, 2 server/nsys/driver failure, 3 Xid, 4 GPUs/port busy, 5 prefill was not cold, 6 frozen input changed,
#       7 wall budget exceeded, 8 pack rejected a trace.
# Env: DRY_RUN (1 = print the plan and run CPU preflight only), CANARY, OUT, BUDGET_S, PORT (gpuq {port}), XID_SINCE (journal --since; default
#      this boot), CPU_REHEARSAL=1 (4B model on the CPU, no GPU checks, port 8394, small windows; the pack is then EXPECTED to
#      reject: exit 8 proves the negative control).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACK=$(dirname "$HERE")
DRY_RUN=${DRY_RUN:-1}
CANARY=${CANARY:-0}
STACK=/home/xanewok/repos/qwen38-weekend/frozen/stack2/build/bin
MODEL=$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
MODEL_BYTES=15475951296
DRAFT_VOCAB=$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json
DRAFT_VOCAB_SHA=e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
OUT=${OUT:-$PACK/runs/attrib-llama}
PORT=${PORT:-${GPUQ_PORT:-8081}}
K=5
BUDGET_S=${BUDGET_S:-1650}
MEM_FREE_MIB=1500
WINDOWS=prefill-32k,decode-8k,decode-90k
CTX=102400
EXTRA=(-ngl 99 -sm tensor --tensor-split 50,50
  --spec-type draft-mtp --spec-draft-n-max $K --spec-draft-n-min 0 --spec-draft-p-min 0
  --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-sampling probabilistic
  --spec-draft-backend-sampling --backend-sampling -t 8 -tb 8 --spec-draft-vocab "$DRAFT_VOCAB")
DRV_EXTRA=()
BIN=$STACK/weekend-srv
if [ "$CANARY" = 1 ]; then WINDOWS=decode-8k; fi
if [ -n "${CPU_REHEARSAL:-}" ]; then
  PORT=8394; MODEL=$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf; MODEL_BYTES=; CTX=8192; K=3
  EXTRA=(-ngl 0 -t 4); DRV_EXTRA=(--tokens prefill-32k=1500,decode-8k=700,decode-90k=2000 --n-predict 400 --twin-n 100)
  BIN=${REHEARSAL_BIN:?set REHEARSAL_BIN to a renamed copy of the server binary (never named llama-server)}
  export CUDA_VISIBLE_DEVICES=
  BUDGET_S=${BUDGET_S:-900}
else
  export CUDA_VISIBLE_DEVICES=$UUIDS
fi
XID_SINCE=${XID_SINCE:-}
S=attribllama$$
T0=$(date +%s)
mkdir -p "$OUT"
if compgen -G "$OUT/*" > /dev/null; then
  prev=$OUT/prev-$(date +%Y%m%d-%H%M%S); mkdir -p "$prev"; find "$OUT" -maxdepth 1 -type f -exec mv {} "$prev"/ \;
fi
export TMPDIR=$OUT/tmp; mkdir -p "$TMPDIR"
CON=$OUT/console.log
log() { echo "[$(date +%T) +$(( $(date +%s) - T0 ))s] $*" | tee -a "$CON"; }
left() { echo $(( T0 + BUDGET_S - $(date +%s) )); }
xid() {
  [ -n "${CPU_REHEARSAL:-}" ] && { echo 0; return; }
  if [ -n "$XID_SINCE" ]; then journalctl -k --since "$XID_SINCE" --no-pager; else journalctl -k -b --no-pager; fi | grep -cE '\bXid\b|fallen off'
}
used() { [ -n "${CPU_REHEARSAL:-}" ] && { echo 0; return; }; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc; }
port_busy() { ss -Hltn "sport = :$PORT" | grep -q .; }
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1; [[ $s != Z* ]]; }

NPID=; CAPTURING=
stop_pid() {
  local p=$1
  [ -n "$p" ] || return 0
  if alive "$p"; then
    kill -TERM "$p" 2>/dev/null
    for _ in $(seq 40); do alive "$p" || break; sleep 1; done
    alive "$p" && kill -KILL "$p" 2>/dev/null
  fi
}
cleanup() {
  [ -n "$CAPTURING" ] && timeout 60 nsys stop --session=$S >/dev/null 2>&1
  timeout 60 nsys shutdown --session=$S --kill sigterm >/dev/null 2>&1
  stop_pid "$NPID"; wait 2>/dev/null
  port_busy && log "WARNING: port $PORT still busy after cleanup"
  log "exit"
}

cpu_checks() {
  (cd "$HERE/inputs" && sha256sum -c --quiet ../inputs.sha256) || { log "frozen corpus/template/client changed"; return 6; }
  command -v nsys > /dev/null || { log "nsys missing"; return 6; }
  if [ -z "${CPU_REHEARSAL:-}" ]; then
    (cd "$STACK" && sha256sum -c --quiet SHA256SUMS) || { log "frozen stack2 binary changed"; return 6; }
    [ "$(stat -c %s "$MODEL" 2>/dev/null)" = "$MODEL_BYTES" ] || { log "model missing or wrong size: $MODEL"; return 6; }
    [ "$(sha256sum "$DRAFT_VOCAB" | cut -d' ' -f1)" = "$DRAFT_VOCAB_SHA" ] || { log "draft vocab changed"; return 6; }
  fi
  python3 -m py_compile "$PACK/attrib_pack.py" "$HERE/llama_drive.py" "$HERE/make_manifest.py" || { log "python compile failed"; return 6; }
  python3 "$PACK/attrib_pack.py" pack --help > /dev/null || { log "attrib_pack.py does not start"; return 6; }
}
gates() {
  [ "$(xid)" = 0 ] || { log "REFUSE: Xid/fallen-off in the kernel log"; return 3; }
  local u; u=$(used)
  [ "$u" -le "$MEM_FREE_MIB" ] || { log "REFUSE: GPUs hold $u MiB (> $MEM_FREE_MIB)"; return 4; }
  port_busy && { log "REFUSE: port $PORT busy"; return 4; }
  log "gates ok: GPUs ${u} MiB, port $PORT free; windows $WINDOWS"
}

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN plan: boot $BIN (nsys launch session) np1 k=$K ctx $CTX; windows $WINDOWS; export sqlite; pack -> $OUT/pack"
  cpu_checks; rc=$?; log "DRY_RUN cpu checks rc=$rc"; exit $rc
fi
trap cleanup EXIT
trap 'log "signal; stopping"; exit 130' INT TERM
cpu_checks || exit $?
gates || exit $?

LD_LIBRARY_PATH=$(dirname "$BIN") nsys launch --session=$S --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  "$BIN" -m "$MODEL" --alias local-benchmark -c "$CTX" -np 1 -ctk q8_0 -ctv q8_0 -fa on "${EXTRA[@]}" \
  -b 2048 -ub 128 --cache-ram 0 --jinja --chat-template-file "$HERE/inputs/chat-template.jinja" \
  --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --host 127.0.0.1 --port "$PORT" > "$OUT/server.log" 2>&1 &
NPID=$!
ok=
for i in $(seq 420); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
  alive "$NPID" || break
  [ "$i" -lt 30 ] || nsys sessions list 2>/dev/null | grep -q "$S" || { log "nsys session gone"; break; }
  [ "$(left)" -gt 0 ] || { log "over budget while booting"; exit 7; }
  sleep 1
done
[ -n "$ok" ] || { log "server did not become healthy"; tail -5 "$OUT/server.log" | tee -a "$CON"; exit 2; }
log "healthy; nsys pid $NPID"

timeout -k 10 $(( $(left) - 240 )) python3 "$HERE/llama_drive.py" --port "$PORT" --session "$S" --out "$OUT" \
  --windows "$WINDOWS" --k "$K" "${DRV_EXTRA[@]}" >> "$CON" 2>&1
drc=$?
[ "$(xid)" = 0 ] || { log "Xid after the captures"; exit 3; }
log "driver rc=$drc"
timeout 60 nsys shutdown --session=$S --kill sigterm >/dev/null 2>&1
stop_pid "$NPID"; NPID=
[ "$drc" = 0 ] || exit $(( drc == 5 ? 5 : 2 ))

# Export is CPU-only; the GPUs are free from here.
for w in ${WINDOWS//,/ }; do
  [ -s "$OUT/$w.nsys-rep" ] || { log "$w: no report"; exit 2; }
  [ "$(left)" -gt 60 ] || { log "over budget before exporting $w"; exit 7; }
  timeout $(( $(left) - 30 )) nsys export --type sqlite --force-overwrite=true -o "$OUT/$w.sqlite" "$OUT/$w.nsys-rep" \
    > "$OUT/export-$w.log" 2>&1 || { log "$w: sqlite export failed"; exit 2; }
done
python3 "$HERE/make_manifest.py" --engine llama --model llama-swift15-iq4xs.json --out-dir "$OUT" --k "$K" \
  --title "llama.cpp stack2 Swift 1.5 IQ4_XS np1 k=$K, S5 flags, 2x RTX 5060 Ti (measured)" --manifest "$OUT/manifest.json" || exit 2
python3 "$PACK/attrib_pack.py" pack "$OUT/manifest.json" --out "$OUT/pack" >> "$CON" 2>&1
prc=$?
log "pack rc=$prc"
[ "$prc" = 0 ] || exit 8
[ "$(left)" -gt 0 ] || { log "over budget"; exit 7; }
log "done: $OUT/pack/budget.md"
