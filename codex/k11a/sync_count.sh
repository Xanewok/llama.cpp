#!/usr/bin/env bash
# K11a window stage: cudaStreamSynchronize per np2 round with the getter sync elision off / on / debug-checked.
# Derived from work/x2-hostprof-s8/x2_hostprof_s8.sh (proof boot, nsys launch, np2 driver), CUDA-trace pass only.
# One binary, three arms booted in turn, so the control is the same build: off = LLAMA_SYNC_ELIDE=0, on = default,
# debug = LLAMA_SYNC_ELIDE_DEBUG=1 (at every skipped sync the backends are queried; busy = a missed enqueue; then a real sync).
# First a proof boot under mutant 3 + debug (flag cleared after every enqueue): the query must report misses, so a debug arm
# without misses is not a blind detector.
# Mechanism capture, not a timing A/B. Exit: 0 ok, 2 server/nsys/driver failure, 3 Xid, 4 GPUs/port busy, 5 prefill was not cold,
# 6 input or proof failure, 7 wall budget exceeded, 8 gate failed.
# Env: ROOT (tree with build-cuda), OUT, PORT, BUDGET_S, XID_SINCE, DRY_RUN (1 = CPU checks only).
set -uo pipefail

PACK=/home/xanewok/repos/attrib-pack
HERE=$PACK/gpu
ROOT=${ROOT:?}
WD=$ROOT/codex/k11a
DRY_RUN=${DRY_RUN:-1}
NP=2
BIN_DIR=$ROOT/build-cuda/bin
BIN=$BIN_DIR/round-trace
export GGML_SCHED_INPUT_STAGING=1
export LLAMA_ARG_KV_PAD=4096
export LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=0
MODEL=$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
MODEL_BYTES=15475951296
DRAFT_VOCAB=$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json
DRAFT_VOCAB_SHA=e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
OUT=${OUT:?}
PORT=${PORT:-${GPUQ_PORT:-8081}}
K=5
MEM_FREE_MIB=1500
CTX=204800; UB=512; BUDGET_S=${BUDGET_S:-3000}
ARMS=(off on debug); WINDOWS=decode-8k
EXTRA=(-ngl 99 -sm tensor --tensor-split 50,50
  --spec-type draft-mtp --spec-draft-n-max $K --spec-draft-n-min 0 --spec-draft-p-min 0
  --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-sampling probabilistic
  --spec-draft-backend-sampling --backend-sampling -t 8 -tb 8 --spec-draft-vocab "$DRAFT_VOCAB")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$UUIDS}
XID_SINCE=${XID_SINCE:-}
S=k11a$$
T0=$(date +%s)
mkdir -p "$OUT"
export TMPDIR=$OUT/tmp; mkdir -p "$TMPDIR"
CON=$OUT/console.log
log() { echo "[$(date +%T) +$(( $(date +%s) - T0 ))s] $*" | tee -a "$CON"; }
left() { echo $(( T0 + BUDGET_S - $(date +%s) )); }
xid() {
  if [ -n "$XID_SINCE" ]; then journalctl -k --since "$XID_SINCE" --no-pager; else journalctl -k -b --no-pager; fi | grep -cE '\bXid\b|fallen off'
}
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc; }
port_busy() { ss -Hltn "sport = :$PORT" | grep -c . > /dev/null; }
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1; [[ $s != Z* ]]; }

NPID=
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
  timeout 60 nsys shutdown --session=$S --kill sigterm >/dev/null 2>&1
  stop_pid "$NPID"; wait 2>/dev/null
  port_busy && log "WARNING: port $PORT still busy after cleanup"
  log "exit"
}

cpu_checks() {
  (cd "$HERE/inputs" && sha256sum -c --quiet ../inputs.sha256) || { log "frozen corpus/template/client changed"; return 6; }
  command -v nsys > /dev/null || { log "nsys missing"; return 6; }
  [ "$(stat -c %s "$MODEL" 2>/dev/null)" = "$MODEL_BYTES" ] || { log "model missing or wrong size: $MODEL"; return 6; }
  [ "$(sha256sum "$DRAFT_VOCAB" | cut -d' ' -f1)" = "$DRAFT_VOCAB_SHA" ] || { log "draft vocab changed"; return 6; }
  python3 -m py_compile "$PACK/attrib_pack.py" "$HERE/llama_drive.py" "$WD/drive_np2.py" "$WD/count.py" || { log "python compile failed"; return 6; }
  python3 "$PACK/attrib_pack.py" pack --help > /dev/null || { log "attrib_pack.py does not start"; return 6; }
  if [ "$DRY_RUN" != 1 ]; then
    [ -x "$BIN" ] || { log "missing $BIN"; return 6; }
    [ "$(strings "$BIN_DIR/libllama.so" | grep -c 'backend busy at a skipped sync')" -ge 1 ] || { log "the build lacks the local query-based sync-elision debug check"; return 6; }
  fi
}
gates() {
  [ "$(xid)" = 0 ] || { log "REFUSE: Xid/fallen-off in the kernel log"; return 3; }
  local u; u=$(used)
  [ "$u" -le "$MEM_FREE_MIB" ] || { log "REFUSE: GPUs hold $u MiB (> $MEM_FREE_MIB)"; return 4; }
  port_busy && { log "REFUSE: port $PORT busy"; return 4; }
  log "gates ok: GPUs ${u} MiB, port $PORT free; NP=$NP arms ${ARMS[*]} window $WINDOWS; chain=$LLAMA_MTP_CHAIN fold=$LLAMA_MTP_FOLD_C"
}

# The inherited env is scrubbed first so a leftover switch cannot reach another arm.
arm_env() {  # ARM
  unset LLAMA_SYNC_ELIDE LLAMA_SYNC_ELIDE_DEBUG LLAMA_SYNC_ELIDE_MUTANT
  case $1 in
    off) export LLAMA_SYNC_ELIDE=0 ;;
    debug) export LLAMA_SYNC_ELIDE_DEBUG=1 ;;
    mut3) export LLAMA_SYNC_ELIDE_DEBUG=1 LLAMA_SYNC_ELIDE_MUTANT=3 ;;
  esac
}
args() { ARGS=(-m "$MODEL" --alias local-benchmark -c "$CTX" -np "$NP" -ctk q8_0 -ctv q8_0 -fa on "${EXTRA[@]}"
  -b 2048 -ub "$1" --cache-ram 0 --jinja --chat-template-file "$HERE/inputs/chat-template.jinja"
  --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --host 127.0.0.1 --port "$PORT"); }
wait_health() {  # pid
  local i
  for i in $(seq 420); do
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -c '"ok"' > /dev/null && return 0
    alive "$1" || return 2
    [ "$i" -lt 30 ] || [ -z "$S" ] || nsys sessions list 2>/dev/null | grep -c "$S" > /dev/null || { log "nsys session gone"; return 2; }
    [ "$(left)" -gt 0 ] || { log "over budget while booting"; return 7; }
    sleep 1
  done
  return 2
}
# Short boot of the same binary, argv and env at verbosity 5 plus the same ignore_eos request twice on slot 0 (the first saves a
# context checkpoint, the second restores it: state save and restore run in every arm); proves the arm's env reached the server.
proof_boot() {  # ARM UB
  local arm=$1 plog=$OUT/$1/proof-ub$2.log rc
  args "$2"; S=
  LD_LIBRARY_PATH=$BIN_DIR LLAMA_ARG_LOG_VERBOSITY=5 "$BIN" "${ARGS[@]}" > "$plog" 2>&1 &
  NPID=$!
  wait_health "$NPID"; rc=$?
  if [ "$rc" = 0 ]; then
    for r in resp resp2; do
      curl -s -m 120 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -o "$plog.$r" -d \
        '{"prompt":"Write a Rust function that parses a CSV line.","n_predict":96,"ignore_eos":true,"cache_prompt":true,"id_slot":0,"temperature":1,"top_k":20,"top_p":0.95,"min_p":0.05,"seed":1729}' || rc=2
    done
    tr '\0' '\n' < "/proc/$NPID/environ" | grep -E '^(GGML|LLAMA)_' | sort > "$OUT/$arm/server-env.txt"
  fi
  stop_pid "$NPID"; NPID=
  [ "$rc" = 0 ] || { log "proof boot/request failed (arm $arm, rc $rc); tail:"; tail -5 "$plog" | tee -a "$CON"; return $(( rc == 7 ? 7 : 2 )); }
  for _ in $(seq 30); do port_busy || break; sleep 1; done
  case $arm in
    off) [ "$(grep -cxF 'LLAMA_SYNC_ELIDE=0' "$OUT/$arm/server-env.txt")" = 1 ] || { log "elide env proof failed (arm off)"; return 6; } ;;
    on) [ "$(grep -c '^LLAMA_SYNC_ELIDE' "$OUT/$arm/server-env.txt")" = 0 ] || { log "elide env proof failed: a switch is set in arm on"; return 6; } ;;
    debug|mut3) [ "$(grep -cxF 'LLAMA_SYNC_ELIDE_DEBUG=1' "$OUT/$arm/server-env.txt")" = 1 ] || { log "elide env proof failed (arm $arm)"; return 6; }
      [ "$arm" = debug ] || [ "$(grep -cxF 'LLAMA_SYNC_ELIDE_MUTANT=3' "$OUT/$arm/server-env.txt")" = 1 ] || { log "elide env proof failed (arm mut3)"; return 6; }
      grep -aq 'sync_elide: debug check engaged' "$plog" || { log "debug check did not engage in the proof boot ($plog)"; return 6; }
      [ "$(grep -ac 'sync_elide: debug check engaged (backend query exact)' "$plog")" = "$(grep -ac 'sync_elide: debug check engaged' "$plog")" ] \
        || { log "debug check fell back to timing: a backend has no query ($plog)"; return 6; } ;;
  esac
  [ "$(grep -acE "n_kv_pad +[=] $LLAMA_ARG_KV_PAD\b" "$plog")" -ge 2 ] || { log "kv_pad proof failed ($plog)"; return 6; }
  [ "$(grep -acE "n_ubatch +[=] $2\b" "$plog")" -ge 2 ] || { log "ub proof failed: n_ubatch = $2 not logged for both contexts ($plog)"; return 6; }
  [ "$(grep -acE 'set_sampler:.*using CPU' "$plog")" = 0 ] || { log "CPU sampler warning in the proof boot ($plog)"; return 6; }
  [ "$(grep -acE 'MTP chain: k=|mtp_fold_decode: engaged' "$plog")" = 0 ] || { log "chain/fold engaged at np2 (-duo has both off) ($plog)"; return 6; }
  [ "$(grep -ac 'created context checkpoint' "$plog")" -ge 1 ] && [ "$(grep -ac 'restored context checkpoint' "$plog")" -ge 1 ] \
    || { log "proof boot did not save and restore a context checkpoint (arm $arm, $plog)"; return 6; }
  if [ "$arm" = mut3 ]; then
    [ "$(grep -ac 'sync_elide: missed: backend busy' "$plog")" -ge 1 ] && [ "$(grep -acE 'sync_elide: elided=[0-9]+ missed=[1-9]' "$plog")" -ge 1 ] \
      || { log "mutant 3 control: the debug check found no busy backend at a skipped sync ($plog)"; return 8; }
    log "mutant control ok: $(grep -ac 'sync_elide: missed: backend busy' "$plog") busy-backend misses under mutant 3"
    return 0
  fi
  [ "$(grep -ac 'sync_elide: missed:' "$plog")" = 0 ] || { log "debug check fired in the proof boot (arm $arm): $(grep -a -m1 'sync_elide: missed:' "$plog")"; return 8; }
  if [ "$arm" = debug ]; then
    # one destructor line per context, all missed=0
    [ "$(grep -acE 'sync_elide: elided=[0-9]+ missed=0 idle_slow=[0-9]+$' "$plog")" -ge 2 ] && [ "$(grep -acE 'sync_elide: elided=[0-9]+ missed=[1-9]' "$plog")" = 0 ] \
      || { log "debug arm: a context reported missed > 0 or no destructor count ($plog)"; return 8; }
  fi
  log "proof ok (arm $arm, ub $2, np $NP): elision switch as intended, n_kv_pad $LLAMA_ARG_KV_PAD x2, no CPU sampler, chain/fold off"
}
boot() {  # ARM UB
  proof_boot "$1" "$2" || return $?
  S=k11a$$$1
  args "$2"
  LD_LIBRARY_PATH=$BIN_DIR nsys launch --session=$S --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
    "$BIN" "${ARGS[@]}" > "$OUT/$1/server.log" 2>&1 &
  NPID=$!
  wait_health "$NPID"; local rc=$?
  [ "$rc" = 0 ] || { log "server did not become healthy (arm $1, rc $rc)"; tail -5 "$OUT/$1/server.log" | tee -a "$CON"; return $(( rc == 7 ? 7 : 2 )); }
  log "healthy (arm $1, ub $2 np $NP); nsys pid $NPID"
}
shutdown_server() {
  timeout 60 nsys shutdown --session=$S --kill sigterm >/dev/null 2>&1
  stop_pid "$NPID"; NPID=
}
drive() {  # OUTDIR WINDOWS
  local cap=$(( $(left) - ${RESERVE_S:-600} ))
  [ "$cap" -gt 120 ] || { log "over budget before driving $2"; return 7; }
  timeout -k 10 "$cap" python3 "$WD/drive_np2.py" run --port "$PORT" --session "$S" --out "$1" --windows "$2" --k "$K" --twin-n 64 >> "$CON" 2>&1
}

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN plan: NP=$NP ctx $CTX ub $UB boot $BIN (nsys launch) arms ${ARMS[*]} window $WINDOWS -> $OUT/<arm>"
  cpu_checks; rc=$?; log "DRY_RUN cpu checks rc=$rc"; exit $rc
fi
trap cleanup EXIT
trap 'log "signal; stopping"; exit 130' INT TERM
cpu_checks || exit $?
gates || exit $?

mkdir -p "$OUT/mut3"
arm_env mut3
log "mutant control: LLAMA_SYNC_ELIDE_MUTANT=$LLAMA_SYNC_ELIDE_MUTANT LLAMA_SYNC_ELIDE_DEBUG=$LLAMA_SYNC_ELIDE_DEBUG (proof boot only)"
proof_boot mut3 "$UB" || exit $?

for arm in "${ARMS[@]}"; do
  mkdir -p "$OUT/$arm"
  arm_env "$arm"
  log "arm $arm: LLAMA_SYNC_ELIDE=${LLAMA_SYNC_ELIDE-<unset>} LLAMA_SYNC_ELIDE_DEBUG=${LLAMA_SYNC_ELIDE_DEBUG-<unset>}"
  boot "$arm" "$UB" || exit $?
  drive "$OUT/$arm" "$WINDOWS"; drc=$?
  [ "$(xid)" = 0 ] || { log "Xid after the capture (arm $arm)"; exit 3; }
  log "arm $arm driver rc=$drc"
  shutdown_server
  [ "$drc" = 0 ] || exit $(( drc == 5 ? 5 : drc == 7 ? 7 : 2 ))
  nsys export --type sqlite --force-overwrite=true -o "$OUT/$arm/decode-8k.sqlite" "$OUT/$arm/decode-8k.nsys-rep" > "$OUT/$arm/export.log" 2>&1 \
    || { log "$arm: sqlite export failed"; exit 2; }
done

python3 "$WD/count.py" "$OUT" ${ARMS[@]} > "$OUT/sync-count.txt" 2>&1; crc=$?
tee -a "$CON" < "$OUT/sync-count.txt"
[ "$crc" = 0 ] || { log "sync-count gate failed rc=$crc"; exit 8; }
log "done: $OUT/sync-count.json"
