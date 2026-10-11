#!/usr/bin/env bash
# nsys trace of the hostless spike levels at np2 (copy of work/m2s10-trace/item/trace_m2s10.sh): arms L0..L3 are one frozen build
# (FROZEN, FROZEN_COMMIT; frozen/hostless-spike = chain-m2-s10 + LLAMA_HOSTLESS) booted with LLAMA_HOSTLESS=0..3, the served
# swift15-iq4-duo argv/env of the chain-m2-s10-np2-real-iq4 A/B, one boot per arm, both slots decoding at 8k and ~90k, nsys
# CUDA trace (CPU sampling off by default: gapcause.py needs none). Mechanism capture: gapcause.py / m2s10_buckets.py per arm;
# ms/round from a profiled boot is not a timing A/B (the unprofiled twin per window in runs.json is the only timing product).
# VRAM (M): nvidia-smi per GPU after each boot is healthy and after the drive, in <arm>/vram.csv; the boot log has the buffers.
# Differences from the A/B harness: texts are the attrib-pack corpus (drive_np2.py), not tokens64k.bin; the prefill and
# 2-slot decode driver is the one that produced the attrib-s10 / hostprof-s10 traces this analysis is validated on.
# Exit: 0 ok, 2 server/nsys/driver failure, 3 Xid, 4 GPUs/port busy, 5 prefill was not cold, 6 frozen input changed or a proof failed,
#       7 wall budget exceeded.
# Env: DRY_RUN (1 = plan + CPU preflight only; --preflight as $1 forces it), CANARY (decode-8k only, every arm), ARMS
#      (default "L0 L1 L2 L3"; "L0b" = a second L0 boot as boot-to-boot null), CPU_SAMPLE (0), OUT, BUDGET_S, ARM_CAP_S, PORT, XID_SINCE,
#      STALL_S, FROZEN, FROZEN_COMMIT (required).
set -uo pipefail
[ "${1:-}" = --preflight ] && DRY_RUN=1

PACK=/home/xanewok/repos/attrib-pack
HERE=$PACK/gpu
WT=/home/xanewok/repos/qwen38-eval-worktrees
WD=$WT/work/hq-gap-s8   # drive_np2.py (unchanged, pinned by the hqgap8 items)
DRY_RUN=${DRY_RUN:-1}
CANARY=${CANARY:-0}
ARMS=${ARMS:-"L0 L1 L2 L3"}
CPU_SAMPLE=${CPU_SAMPLE:-0}
FROZEN=${FROZEN:?FROZEN}
FROZEN_COMMIT=${FROZEN_COMMIT:?FROZEN_COMMIT}
NP=2
ITEM=${ITEM:-hostless-trace-iq4}
MODEL=$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
MODEL_BYTES=15475951296
DRAFT_VOCAB=$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json
DRAFT_VOCAB_SHA=e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6
TEMPLATE=/home/xanewok/repos/qwen38-eval/corpora/templates/swift-leading-instructions-v1.jinja
TEMPLATE_SHA=4af3dcf0e98bf87adf684a4824683ab15fffd64e93ed1170061ef86a15827ba8
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
OUT=${OUT:-/mnt/scratch/results/2026-10-11-$ITEM/full}
PORT=${PORT:-${GPUQ_PORT:-8081}}
K=5
MEM_FREE_MIB=1500
UB=512
CTX=204800
WINDOWS=decode-8k,decode-90k
MODEL_JSON=llama-swift15-iq4xs.json
BUDGET_S=${BUDGET_S:-4400}
ARM_CAP_S=${ARM_CAP_S:-700}
RESERVE_S=${RESERVE_S:-300}
STALL_S=${STALL_S:-150}

# The A/B harness scrubs these knobs from the environment of every arm; the arm env below is then exactly its arms.*.env.
unset GGML_CUDA_Q8_DECODE LLAMA_MTP_GRAPH_REUSE GGML_META_PLAN_REUSE LLAMA_ARG_LOG_VERBOSITY GGML_CUDA_DEBUG_SSM_CONV \
  GGML_CUDA_AR_COPY_STREAMS LLAMA_GRAPH_REUSE_DISABLE LLAMA_ROUND_TRACE GGML_SCHED_DEBUG GGML_CUDA_FA_MMA_Q8_0 \
  LLAMA_MTP_FOLD_DEBUG_MUTANT LLAMA_STACK7_PROOF LLAMA_HOST_TRIMS_SEQRM LLAMA_HOST_TRIMS_BULKH LLAMA_HOST_TRIMS_VERIFY \
  LLAMA_HOST_TRIMS_MUTANT LLAMA_HOST_TRIMS_PROOF GGML_CUDA_DEBUG_GDN_STATE LLAMA_MTP_DRAFT_DUMP LLAMA_MTP_CHAIN_DUMP \
  LLAMA_MTP_CHAIN_DEBUG_REVERSE_U LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD GGML_ROUND_TRACE_SYNC LLAMA_MTP_FOLD_DEBUG_LAYOUT \
  LLAMA_MTP_FOLD_DEBUG_SWAP_SEQ GGML_CUDA_GDN_JOINT_MUTANT GGML_CUDA_ADD_RMS_NORM GGML_CUDA_ADD_RMS_NORM_MUTANT \
  GGML_CUDA_ALPHA_ALIAS_MUTANT GGML_CUDA_ALPHA_GATE GGML_CUDA_ALPHA_GATE_MUTANT GGML_CUDA_FATTN_KV_MAX_MUTANT \
  GGML_CUDA_FATTN_KV_MAX_SHARE GGML_CUDA_INPUT_STAGE_MUTANT GGML_CUDA_INPUT_STAGE_ORDER GGML_CUDA_MMQ_PDL GGML_CUDA_MMQ_PDL_LOG \
  GGML_CUDA_MMQ_PDL_MUTANT GGML_CUDA_MMQ_SK_MUTANT GGML_CUDA_MMQ_TILES GGML_CUDA_MMQ_TILES_LOG GGML_CUDA_PDL GGML_CUDA_Q8_REUSE \
  GGML_CUDA_Q8_REUSE_STALE_MUTANT GGML_CUDA_RMS_NORM_SCALE_PAIR GGML_CUDA_RMS_NORM_SCALE_PAIR_MUTANT GGML_CUDA_TOPK_BATCHED \
  GGML_CUDA_TOPK_LOG GGML_CUDA_TOPK_MUTANT GGML_META_INPUT_RANK_ORDER GGML_META_INPUT_STAGE GGML_META_INPUT_STAGE_MUTANT \
  LLAMA_HOSTLESS LLAMA_HOSTLESS_MUTANT LLAMA_HOSTLESS_VERIFY GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT GGML_SSM_CONV_RS_LOG GGML_SSM_CONV_RS_MUTANT GGML_SSM_CONV_RS_OFF LLAMA_KV_ROT_ALLOC_FAIL \
  LLAMA_KV_ROT_MUTANT LLAMA_KV_ROT_PERSIST LLAMA_SYNC_ELIDE LLAMA_SYNC_ELIDE_DEBUG LLAMA_SYNC_ELIDE_MUTANT
export GGML_SCHED_INPUT_STAGING=1 LLAMA_ARG_KV_PAD=4096 LLAMA_MTP_CHAIN=1 LLAMA_MTP_FOLD_C=1

EXTRA=(-ngl 99 -sm tensor --tensor-split 50,50
  --spec-type draft-mtp --spec-draft-n-max $K --spec-draft-n-min 0 --spec-draft-p-min 0
  --spec-draft-type-k f16 --spec-draft-type-v f16 --metrics --spec-draft-sampling probabilistic
  --spec-draft-backend-sampling --backend-sampling -t 8 -tb 8 --spec-draft-vocab "$DRAFT_VOCAB")
BACKTRACE=dwarf; SAMPLING_PERIOD=2336904; SAMPLES_PER_BT=1; WINDOW_S=6
SAMPLE_ARGS="--sample=process-tree --cpuctxsw=process-tree --backtrace=$BACKTRACE --samples-per-backtrace=$SAMPLES_PER_BT --sampling-period=$SAMPLING_PERIOD"
[ "$CPU_SAMPLE" = 1 ] || SAMPLE_ARGS=""
HOST_EXTRA=(--window "$WINDOW_S" --delay 1.5 --n-predict 2000 --twin-n 600 --nsys-sample-args "$SAMPLE_ARGS")
[ "$CANARY" != 1 ] || { WINDOWS=decode-8k; BUDGET_S=${BUDGET_S_CANARY:-850}; RESERVE_S=${RESERVE_S_CANARY:-250}; ARM_CAP_S=${ARM_CAP_S_CANARY:-200}; }
export CUDA_VISIBLE_DEVICES=$UUIDS
XID_SINCE=${XID_SINCE:-}
T0=$(date +%s)
mkdir -p "$OUT"
if compgen -G "$OUT/*" > /dev/null; then
  prev=$OUT/prev-$(date +%Y%m%d-%H%M%S); mkdir -p "$prev"; find "$OUT" -maxdepth 1 -mindepth 1 -not -name 'prev-*' -exec mv {} "$prev"/ \;
fi
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

# Per-arm state, set by set_arm: ROOT (frozen tree), COMMIT (SOURCE line 1), STACK (bin dir), BIN, AOUT (arm output dir), S (nsys session),
# LEVEL (LLAMA_HOSTLESS of the arm).
ROOT=; COMMIT=; STACK=; BIN=; AOUT=; S=; KIND=; LEVEL=
set_arm() {  # ARM
  case $1 in
    L0|L0b) LEVEL=0 ;;
    L1) LEVEL=1 ;;
    L2) LEVEL=2 ;;
    L3) LEVEL=3 ;;
    *) log "unknown arm $1"; return 6 ;;
  esac
  ROOT=$FROZEN; COMMIT=$FROZEN_COMMIT; KIND=patch
  STACK=$ROOT/build/bin; BIN=$STACK/round-trace; AOUT=$OUT/$1; S=
  mkdir -p "$AOUT"
}
vram() {  # TAG: per-GPU used/total MiB (M)
  nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader,nounits | sed "s/^/$1, /" >> "$AOUT/vram.csv"
}

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
# Contract: return with $PORT free. NPID is the `nsys launch` wrapper; it exits on SIGTERM while a hung server (a stalled nsys
# stop leaves it in CUDA teardown) is still the listener, so wait, then TERM/KILL the listener (round-trace only).
free_port() {
  local i p
  for i in $(seq 30); do port_busy || return 0; sleep 1; done
  for p in $(ss -Htlnp "sport = :$PORT" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
    [ "$(ps -o comm= -p "$p" 2>/dev/null)" = round-trace ] || continue
    log "listener $p still holds port $PORT after the server wrapper exited; TERM"
    kill -TERM "$p" 2>/dev/null
    for i in $(seq 30); do alive "$p" || break; sleep 1; done
    alive "$p" && { log "listener $p ignored TERM; KILL"; kill -KILL "$p" 2>/dev/null; }
  done
  for i in $(seq 20); do port_busy || return 0; sleep 1; done
  return 1
}
cleanup() {
  [ -n "$S" ] && [ -n "$CAPTURING" ] && timeout 60 nsys stop --session=$S >/dev/null 2>&1
  [ -n "$S" ] && timeout 60 nsys shutdown --session=$S --kill sigterm >/dev/null 2>&1
  stop_pid "$NPID"; wait 2>/dev/null
  free_port || log "WARNING: port $PORT still busy after cleanup"
  log "exit"
}

cpu_checks() {
  local arm
  (cd "$HERE/inputs" && sha256sum -c --quiet ../inputs.sha256) || { log "frozen corpus/template/client changed"; return 6; }
  [ "$(sha256sum "$TEMPLATE" | cut -d' ' -f1)" = "$TEMPLATE_SHA" ] || { log "frozen corpus/template/client changed (A/B template)"; return 6; }
  command -v nsys > /dev/null || { log "nsys missing"; return 6; }
  [ "$(stat -c %s "$MODEL" 2>/dev/null)" = "$MODEL_BYTES" ] || { log "model missing or wrong size: $MODEL"; return 6; }
  [ "$(sha256sum "$DRAFT_VOCAB" | cut -d' ' -f1)" = "$DRAFT_VOCAB_SHA" ] || { log "draft vocab changed"; return 6; }
  python3 -m py_compile "$PACK/attrib_pack.py" "$HERE/llama_drive.py" "$HERE/make_manifest.py" "$WD/drive_np2.py" || { log "python compile failed"; return 6; }
  python3 "$PACK/attrib_pack.py" pack --help > /dev/null || { log "attrib_pack.py does not start"; return 6; }
  for arm in $ARMS; do
    set_arm "$arm" || return $?
    (cd "$STACK" && sha256sum -c --quiet SHA256SUMS) || { log "frozen $arm binary changed ($STACK)"; return 6; }
    [ "$(head -c 40 "$ROOT/SOURCE")" = "$COMMIT" ] || { log "frozen $arm SOURCE commit changed ($ROOT)"; return 6; }
    [ -x "$BIN" ] || { log "missing $BIN"; return 6; }
  done
  [ "$CPU_SAMPLE" != 1 ] || sampling_checks
}
# nsys CPU sampling needs perf_event access for user space, an nsys that knows the flags, and symbols + unwind tables in the frozen libs.
sampling_checks() {
  local p l arm; p=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 9)
  [ "$p" -le 2 ] || { log "perf_event_paranoid=$p > 2: nsys CPU sampling is unavailable"; return 2; }
  [ "$(nsys start --help 2>&1 | grep -c -e '--samples-per-backtrace')" -ge 1 ] || { log "this nsys has no --samples-per-backtrace"; return 2; }
  for arm in $ARMS; do
    set_arm "$arm" || return $?
    for l in libllama.so libllama-common.so libllama-server-impl.so libggml-base.so libggml-cuda.so libggml-cpu.so; do
      [ -f "$STACK/$l" ] || { log "missing $STACK/$l"; return 6; }
      [ "$(readelf -S "$STACK/$l" 2>/dev/null | grep -c -F '.symtab')" -ge 1 ] || { log "$l has no .symtab: frames would be unnamed"; return 6; }
      [ "$(readelf -S "$STACK/$l" 2>/dev/null | grep -c -F '.eh_frame')" -ge 1 ] || { log "$l has no .eh_frame: dwarf unwinding would fail"; return 6; }
    done
  done
}
gates() {
  [ "$(xid)" = 0 ] || { log "REFUSE: Xid/fallen-off in the kernel log"; return 3; }
  local u; u=$(used)
  [ "$u" -le "$MEM_FREE_MIB" ] || { log "REFUSE: GPUs hold $u MiB (> $MEM_FREE_MIB)"; return 4; }
  port_busy && { log "REFUSE: port $PORT busy"; return 4; }
  log "gates ok: GPUs ${u} MiB, port $PORT free; NP=$NP arms '$ARMS' windows $WINDOWS; chain=$LLAMA_MTP_CHAIN fold=$LLAMA_MTP_FOLD_C sample=$CPU_SAMPLE frozen=$FROZEN"
}

args() { ARGS=(-m "$MODEL" -c "$CTX" -np "$NP" -ctk q8_0 -ctv q8_0 -fa on "${EXTRA[@]}"
  -b 2048 -ub "$UB" --cache-ram 0 --jinja --chat-template-file "$TEMPLATE"
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
# Short boot of the same binary, argv and env at verbosity 5 (the INFO/DEBUG proof lines would sit inside the measured server
# otherwise) plus one ignore_eos request per slot, so the sampler/chain/fold lines are logged.
proof_boot() {
  local plog=$AOUT/proof-ub$UB.log rc
  args; S=
  LD_LIBRARY_PATH=$STACK LLAMA_STACK7_PROOF=1 LLAMA_ARG_LOG_VERBOSITY=5 LLAMA_HOSTLESS=$LEVEL LLAMA_HOSTLESS_VERIFY=1 \
    "$BIN" "${ARGS[@]}" > "$plog" 2>&1 &
  NPID=$!
  wait_health "$NPID"; rc=$?
  [ "$rc" = 0 ] && vram proof-healthy
  if [ "$rc" = 0 ]; then
    local sl pids=()
    for sl in $(seq 0 $(( NP - 1 ))); do
      curl -s -m 120 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -o "$plog.resp$sl" -d \
        '{"prompt":"Write a Rust function that parses a CSV line.","n_predict":96,"ignore_eos":true,"cache_prompt":false,"id_slot":'"$sl"',"temperature":1,"top_k":20,"top_p":0.95,"min_p":0.05,"seed":'"$(( 1729 + sl ))"'}' &
      pids+=($!)
    done
    for sl in "${pids[@]}"; do wait "$sl" || rc=2; done
  fi
  stop_pid "$NPID"; NPID=
  [ "$rc" = 0 ] || { log "proof boot/request failed (rc $rc); tail:"; tail -5 "$plog" | tee -a "$CON"; return $(( rc == 7 ? 7 : 2 )); }
  for _ in $(seq 30); do port_busy || break; sleep 1; done
  [ "$(grep -acE "n_kv_pad +[=] $LLAMA_ARG_KV_PAD\b" "$plog")" -ge 2 ] || { log "kv_pad proof failed ($plog)"; return 6; }
  grep -aq 'copy-engine H2D streams per GPU: 2' "$plog" || { log "AR dual-stream proof failed ($plog)"; return 6; }
  [ "$(grep -acE "n_ubatch +[=] $UB\b" "$plog")" -ge 2 ] || { log "ub proof failed: n_ubatch = $UB not logged for both contexts ($plog)"; return 6; }
  [ "$(grep -acE 'using backend candidates: n_cand = [1-9][0-9]*, n_ban = [1-9][0-9]*, top_k = 20\b' "$plog")" -ge 1 ] \
    || { log "cand-neginf proof failed: no 'using backend candidates ... n_ban' line ($plog)"; return 6; }
  [ "$(grep -acE 'set_sampler:.*using CPU' "$plog")" = 0 ] || { log "CPU sampler warning in the proof boot ($plog)"; return 6; }
  grep -aq 'mtp_fold_decode: engaged n_seq=2\b' "$plog" || { log "fold C proof failed: never engaged at n_seq=2 ($plog)"; return 6; }
  grep -aq 'MTP chain: k=5 seqs=2\b' "$plog" || { log "chain proof failed: no 'MTP chain: k=5 seqs=2' ($plog)"; return 6; }
  [ "$(grep -acE 'drafting step by step|the batch does not fit the MTP chain|MTP chain disabled' "$plog")" = 0 ] \
    || { log "chain fallback or refusal in the proof boot ($plog)"; return 6; }
  hostless_proof "$plog" proof || return 6
  log "proof ok (L$LEVEL, ub $UB, np $NP, $COMMIT): n_kv_pad $LLAMA_ARG_KV_PAD x2, AR H2D streams 2, backend candidates with ban, no CPU sampler, fold n_seq=2, chain k=5 seqs=2, hostless strings of L$LEVEL"
}
# hostless_proof LOG KIND: the strings of LEVEL and none above it. KIND proof = the verbosity-5 proof boot (verify on: device ids equal
# the host ids, no mismatch); KIND run = the measured boot (default verbosity: the level and placement lines only).
hostless_proof() {
  local f=$1 kind=$2 p
  if [ "$LEVEL" = 0 ]; then
    [ "$(grep -ac 'hostless:' "$f")" = 0 ] || { log "hostless proof failed: L0 logged hostless lines ($f)"; return 6; }
    return 0
  fi
  local want=("hostless: level=$LEVEL mutant=none" 'hostless: tok_embd buffer=Meta')
  [ "$LEVEL" -lt 2 ] || want+=('hostless: draft id slots=64 buffer=Meta')
  if [ "$kind" = proof ]; then
    [ "$LEVEL" -lt 2 ] || want+=('hostless: chain drafts deferred k=5 seqs=' 'hostless: target batch reads device draft ids'
                                 'hostless: device order engaged backends=' 'hostless verify: batches=')
    [ "$LEVEL" -lt 3 ] || want+=('hostless: output lag engaged')
  fi
  for p in "${want[@]}"; do
    grep -aqF "$p" "$f" || { log "hostless proof failed: no '$p' ($f)"; return 6; }
  done
  [ "$(grep -ac 'hostless verify: MISMATCH' "$f")" = 0 ] || { log "hostless proof failed: device ids differ from the host ids ($f)"; return 6; }
  [ "$LEVEL" -ge 2 ] || [ "$(grep -ac 'hostless: draft id slots' "$f")" = 0 ] || { log "hostless proof failed: L1 allocated id slots ($f)"; return 6; }
}
# boot: proof boot, then the measured server under nsys launch session S; sets NPID.
boot() {
  proof_boot || return $?
  S=hostless$$
  args
  LD_LIBRARY_PATH=$STACK LLAMA_HOSTLESS=$LEVEL nsys launch --session=$S --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
    "$BIN" "${ARGS[@]}" > "$AOUT/server.log" 2>&1 &
  NPID=$!
  wait_health "$NPID"; local rc=$?
  [ "$rc" = 0 ] || { log "server did not become healthy (rc $rc)"; tail -5 "$AOUT/server.log" | tee -a "$CON"; return $(( rc == 7 ? 7 : 2 )); }
  vram run-healthy
  log "healthy ($KIND ub $UB np $NP); nsys pid $NPID"
}
shutdown_server() {
  timeout 60 nsys shutdown --session=$S --kill sigterm >/dev/null 2>&1
  stop_pid "$NPID"; NPID=
  free_port || log "WARNING: port $PORT still busy after the server shutdown"
}
# Contract: a driver whose server logs nothing for STALL_S s is dead (live runs print every 3 s; np2-step-hostprof-s10 full-a1 froze at
# the first nsys stop and the driver then waited 23 min on SSE). Kill it and return 125 instead of burning the budget.
run_driver() {  # CMD...
  "$@" >> "$CON" 2>&1 &
  local dp=$! sz=-1 cur now last
  last=$(date +%s)
  while alive "$dp"; do
    cur=$(stat -c %s "$AOUT/server.log" 2>/dev/null || echo 0); now=$(date +%s)
    if [ "$cur" != "$sz" ]; then sz=$cur; last=$now
    elif [ $(( now - last )) -ge "$STALL_S" ]; then
      log "server stalled: no server.log output for $(( now - last )) s while the driver waits; killing the driver"
      kill -TERM "$dp" 2>/dev/null; sleep 3; alive "$dp" && kill -KILL "$dp" 2>/dev/null
      wait "$dp" 2>/dev/null; return 125
    fi
    sleep 2
  done
  wait "$dp"
}
drive() {  # OUTDIR WINDOWS
  local cap=$(( $(left) - RESERVE_S ))
  [ "$cap" -le "$ARM_CAP_S" ] || cap=$ARM_CAP_S
  [ "$cap" -gt 120 ] || { log "over budget before driving $2"; return 7; }
  run_driver timeout -k 10 "$cap" python3 "$WD/drive_np2.py" run --port "$PORT" --session "$S" --out "$1" --windows "$2" --k "$K" "${HOST_EXTRA[@]}"
}

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN plan: NP=$NP ctx $CTX ub $UB arms '$ARMS' (L<n> = $FROZEN with LLAMA_HOSTLESS=<n>; nsys launch) env KV_PAD=$LLAMA_ARG_KV_PAD STAGING=$GGML_SCHED_INPUT_STAGING CHAIN=$LLAMA_MTP_CHAIN FOLD_C=$LLAMA_MTP_FOLD_C; per arm proof boot (verbosity 5 + one ignore_eos request per slot) then cuda pass $WINDOWS (sampling: ${SAMPLE_ARGS:-off}) -> $OUT/<arm>"
  cpu_checks; rc=$?; log "DRY_RUN cpu checks rc=$rc"; exit $rc
fi
trap cleanup EXIT
trap 'log "signal; stopping"; exit 130' INT TERM
cpu_checks || exit $?
gates || exit $?

for arm in $ARMS; do
  set_arm "$arm" || exit $?
  [ "$(left)" -gt $(( RESERVE_S + 200 )) ] || { log "over budget before arm $arm"; exit 7; }
  log "== arm $arm ($COMMIT)"
  boot || exit $?
  CAPTURING=
  drive "$AOUT" "$WINDOWS"
  drc=$?
  vram run-after-drive
  hostless_proof "$AOUT/server.log" run || exit 6
  [ "$(xid)" = 0 ] || { log "Xid after the cuda pass of $arm"; exit 3; }
  log "$arm cuda pass driver rc=$drc"
  shutdown_server
  [ "$drc" = 0 ] || exit $(( drc == 5 ? 5 : drc == 7 ? 7 : 2 ))
  [ -s "$AOUT/runs.json" ] || { log "$arm: no runs.json"; exit 2; }
done

# ---- CPU only from here
for arm in $ARMS; do
  for w in ${WINDOWS//,/ }; do
    [ -s "$OUT/$arm/$w.nsys-rep" ] || { log "$arm/$w: no report"; exit 2; }
    [ "$(left)" -gt 90 ] || { log "over budget before exporting $arm/$w"; exit 7; }
    timeout $(( $(left) - 60 )) nsys export --type sqlite --force-overwrite=true -o "$OUT/$arm/$w.sqlite" "$OUT/$arm/$w.nsys-rep" > "$OUT/$arm/export-$w.log" 2>&1 \
      || { log "$arm/$w: sqlite export failed"; exit 2; }
  done
done
log "done: $OUT/<arm>/{${WINDOWS}}.sqlite + runs.json (twin ms/round) + vram.csv; analyse with work/sync-map/opus/gapcause.py per arm"
exit 0
