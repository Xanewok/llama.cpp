#!/usr/bin/env bash
# K11b mechanism capture: np2 8k serving shape of x2-hostprof-s8 (Swift 1.5 IQ4_XS, -np 2, -c 204800, ub 512, chain and fold C off,
# 2x RTX 5060 Ti, nsys CUDA trace, decode-8k window with both slots decoding), on this tree's CUDA build instead of frozen/stack8.
# Arms boot one after the other, one boot each (a mechanism capture, not a timing A/B):
#   base       GGML_META_INPUT_STAGE=0                                    per-rank copies, rank 0 first (stock path)
#   shared     (no K11b env)                                              one staging copy, slowest measured link first
#   sharedfwd  GGML_CUDA_INPUT_STAGE_ORDER=fwd                            one staging copy, rank 0 first (isolates the copy from the order)
#   s1rev      GGML_META_INPUT_STAGE=0 GGML_META_INPUT_RANK_ORDER=rev     per-rank copies, last rank first (S1)
#   base2      as base, last                                              same-arm repeat: the boot/time spread of every contrast above
# Per arm: proof boot (verbosity 5, one seeded request, env and staging log lines checked) -> measured boot under nsys launch ->
# cuda pass (decode-8k) -> sqlite export; then k11b_analyze.py. Exit: 0 ok, 2 server/nsys/driver failure, 3 Xid, 4 GPUs/port busy,
# 5 prefill was not cold, 6 input or proof changed, 7 wall budget exceeded.
# Env: DRY_RUN (1 = CPU preflight only), CANARY (1 = shared arm only), OUT, ITEM, BUDGET_S, PORT (gpuq {port}), XID_SINCE, TREE.
set -uo pipefail

PACK=/home/xanewok/repos/attrib-pack
HERE=$PACK/gpu
TREE=${TREE:-/home/xanewok/repos/qwen38-eval-worktrees/wt/s10-k11b}
WD=$TREE/codex/nsys
DRY_RUN=${DRY_RUN:-1}
CANARY=${CANARY:-0}
NP=2
ITEM=${ITEM:-k11b-nsys}
STACK=$TREE/build-cuda/bin
export GGML_SCHED_INPUT_STAGING=1
export LLAMA_ARG_KV_PAD=4096
# serve.sh swift15-iq4-duo: chain and fold C are off (they refuse n_seq > 1).
export LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=0
MODEL=$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
MODEL_BYTES=15475951296
DRAFT_VOCAB=$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json
DRAFT_VOCAB_SHA=e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
OUT=${OUT:-$PACK/runs/$ITEM/full}
PORT=${PORT:-${GPUQ_PORT:-8081}}
K=5
MEM_FREE_MIB=1500
BIN=$STACK/round-trace
CTX=204800; UB=512; BUDGET_S=${BUDGET_S:-1500}
MODEL_JSON=llama-swift15-iq4xs.json
ARMS=(base shared sharedfwd s1rev base2); WINDOWS=decode-8k
if [ "$CANARY" = 1 ]; then ARMS=(shared); BUDGET_S=${BUDGET_S_CANARY:-850}; RESERVE_S=${RESERVE_S:-250}; fi
EXTRA=(-ngl 99 -sm tensor --tensor-split 50,50
  --spec-type draft-mtp --spec-draft-n-max $K --spec-draft-n-min 0 --spec-draft-p-min 0
  --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-sampling probabilistic
  --spec-draft-backend-sampling --backend-sampling -t 8 -tb 8 --spec-draft-vocab "$DRAFT_VOCAB")
export CUDA_VISIBLE_DEVICES=$UUIDS
XID_SINCE=${XID_SINCE:-}
S=k11b$$
T0=$(date +%s)
mkdir -p "$OUT"
if compgen -G "$OUT/*" > /dev/null; then
  prev=$OUT/prev-$(date +%Y%m%d-%H%M%S); mkdir -p "$prev"; find "$OUT" -maxdepth 1 -mindepth 1 -not -path "$prev" -exec mv {} "$prev"/ \;
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

# The build window stamps build-cuda with the source trees it compiled; a tree edited since is not what was built.
source_stamp() { git -C "$TREE" rev-parse HEAD:ggml HEAD:src HEAD:common HEAD:tests | paste -sd+; }
cpu_checks() {
  (cd "$HERE/inputs" && sha256sum -c --quiet ../inputs.sha256) || { log "frozen corpus/template/client changed"; return 6; }
  command -v nsys > /dev/null || { log "nsys missing"; return 6; }
  [ -x "$BIN" ] || { log "no CUDA build: $BIN (run bw-s10-k11b first)"; return 6; }
  [ "$(cat "$STACK/../BUILD_STAMP" 2>/dev/null)" = "$(source_stamp)" ] || { log "build-cuda does not match the source trees of HEAD"; return 6; }
  [ "$(stat -c %s "$MODEL" 2>/dev/null)" = "$MODEL_BYTES" ] || { log "model missing or wrong size: $MODEL"; return 6; }
  [ "$(sha256sum "$DRAFT_VOCAB" | cut -d' ' -f1)" = "$DRAFT_VOCAB_SHA" ] || { log "draft vocab changed"; return 6; }
  python3 -m py_compile "$PACK/attrib_pack.py" "$HERE/llama_drive.py" "$WD/drive_np2.py" "$WD/k11b_analyze.py" || { log "python compile failed"; return 6; }
}
gates() {
  [ "$(xid)" = 0 ] || { log "REFUSE: Xid/fallen-off in the kernel log"; return 3; }
  local u; u=$(used)
  [ "$u" -le "$MEM_FREE_MIB" ] || { log "REFUSE: GPUs hold $u MiB (> $MEM_FREE_MIB)"; return 4; }
  port_busy && { log "REFUSE: port $PORT busy"; return 4; }
  log "gates ok: GPUs ${u} MiB, port $PORT free; NP=$NP arms ${ARMS[*]} window $WINDOWS; chain=$LLAMA_MTP_CHAIN fold=$LLAMA_MTP_FOLD_C"
}

# The only difference between the arms. The inherited env is scrubbed first so a leftover K11b variable cannot reach another arm.
arm_env() {  # ARM
  unset GGML_META_INPUT_STAGE GGML_META_INPUT_RANK_ORDER GGML_CUDA_INPUT_STAGE_ORDER GGML_META_INPUT_STAGE_MUTANT GGML_CUDA_INPUT_STAGE_MUTANT
  case $1 in
    base|base2) export GGML_META_INPUT_STAGE=0 ;;
    shared) ;;
    sharedfwd) export GGML_CUDA_INPUT_STAGE_ORDER=fwd ;;
    s1rev) export GGML_META_INPUT_STAGE=0 GGML_META_INPUT_RANK_ORDER=rev ;;
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
# Short boot of the same binary, argv and env at verbosity 5 plus one seeded request: the staging log lines and the arm's env are
# proven here, outside the measured server. The generated text is kept to compare arms (same seed, exact change => same text).
proof_boot() {  # ARM UB
  local arm=$1 plog=$OUT/$1/proof-ub$2.log rc
  args "$2"; S=
  LD_LIBRARY_PATH=$(dirname "$BIN") LLAMA_STACK7_PROOF=1 LLAMA_ARG_LOG_VERBOSITY=5 "$BIN" "${ARGS[@]}" > "$plog" 2>&1 &
  NPID=$!
  wait_health "$NPID"; rc=$?
  if [ "$rc" = 0 ]; then
    curl -s -m 120 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -o "$plog.resp" -d \
      '{"prompt":"Write a Rust function that parses a CSV line.","n_predict":96,"ignore_eos":true,"cache_prompt":false,"id_slot":0,"temperature":1,"top_k":20,"top_p":0.95,"min_p":0.05,"seed":1729}' || rc=2
    tr '\0' '\n' < "/proc/$NPID/environ" | grep -E '^(GGML|LLAMA)_' | sort > "$OUT/$arm/server-env.txt"
  fi
  stop_pid "$NPID"; NPID=
  [ "$rc" = 0 ] || { log "proof boot/request failed (arm $arm, rc $rc); tail:"; tail -5 "$plog" | tee -a "$CON"; return $(( rc == 7 ? 7 : 2 )); }
  for _ in $(seq 30); do port_busy || break; sleep 1; done
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["content"])' "$plog.resp" > "$OUT/$arm/proof-content.txt" 2>/dev/null \
    || { log "proof response has no content (arm $arm)"; return 6; }
  [ -s "$OUT/$arm/proof-content.txt" ] || { log "proof response is empty (arm $arm)"; return 6; }
  case $arm in
    base|base2|s1rev)
      [ "$(grep -cxF 'GGML_META_INPUT_STAGE=0' "$OUT/$arm/server-env.txt")" = 1 ] || { log "K11b env proof failed: GGML_META_INPUT_STAGE=0 not in the server environment (arm $arm)"; return 6; }
      [ "$(grep -ac 'one pinned copy per input' "$plog")" = 0 ] || { log "K11b engagement proof failed: shared staging engaged in arm $arm ($plog)"; return 6; } ;;
    shared|sharedfwd)
      [ "$(grep -ac '^GGML_META_INPUT_STAGE=' "$OUT/$arm/server-env.txt")" = 0 ] || { log "K11b env proof failed: stale GGML_META_INPUT_STAGE in arm $arm"; return 6; }
      [ "$(grep -ac 'one pinned copy per input' "$plog")" -ge 1 ] || { log "K11b engagement proof failed: no shared staging log line (arm $arm, $plog)"; return 6; } ;;
  esac
  case $arm in
    s1rev) [ "$(grep -cxF 'GGML_META_INPUT_RANK_ORDER=rev' "$OUT/$arm/server-env.txt")" = 1 ] || { log "K11b env proof failed: rank order env (arm $arm)"; return 6; } ;;
    sharedfwd)
      [ "$(grep -ac 'slowest host link first: CUDA0 ' "$plog")" -ge 1 ] || { log "K11b order proof failed: forward order not applied (arm $arm)"; return 6; } ;;
    shared)
      # GPU1 sits behind the Gen3 riser (X2 section 3): the measured order must put it first.
      [ "$(grep -ac 'slowest host link first: CUDA1 ' "$plog")" -ge 1 ] || { log "K11b order proof failed: measured order does not put CUDA1 first (arm $arm; $(grep -a 'one pinned copy per input' "$plog" | tail -1))"; return 6; } ;;
  esac
  [ "$(grep -acE "n_kv_pad +[=] $LLAMA_ARG_KV_PAD\b" "$plog")" -ge 2 ] || { log "kv_pad proof failed ($plog)"; return 6; }
  grep -aq 'copy-engine H2D streams per GPU: 2' "$plog" || { log "AR dual-stream proof failed ($plog)"; return 6; }
  [ "$(grep -acE "n_ubatch +[=] $2\b" "$plog")" -ge 2 ] || { log "ub proof failed: n_ubatch = $2 not logged for both contexts ($plog)"; return 6; }
  [ "$(grep -acE 'using backend candidates: n_cand = [1-9][0-9]*, n_ban = [1-9][0-9]*, top_k = 20\b' "$plog")" -ge 1 ] \
    || { log "cand-neginf proof failed: no 'using backend candidates ... n_ban' line ($plog)"; return 6; }
  [ "$(grep -acE 'set_sampler:.*using CPU' "$plog")" = 0 ] || { log "CPU sampler warning in the proof boot ($plog)"; return 6; }
  [ "$(grep -acE 'MTP chain: k=|mtp_fold_decode: engaged' "$plog")" = 0 ] || { log "chain/fold engaged at np2 (-duo has both off) ($plog)"; return 6; }
  log "proof ok (arm $arm, ub $2, np $NP): K11b env and staging engagement as intended, kv_pad $LLAMA_ARG_KV_PAD x2, AR H2D streams 2, backend candidates with ban, no CPU sampler, chain/fold off"
}
# boot ARM UB: proof boot, then the measured server under nsys launch session S; sets NPID.
boot() {
  proof_boot "$1" "$2" || return $?
  S=k11b$$$1
  args "$2"
  LD_LIBRARY_PATH=$(dirname "$BIN") nsys launch --session=$S --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
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
drive() {  # OUTDIR WINDOWS [extra args...]
  local o=$1 w=$2; shift 2
  local cap=$(( $(left) - ${RESERVE_S:-600} ))
  [ "$cap" -gt 120 ] || { log "over budget before driving $w"; return 7; }
  timeout -k 10 "$cap" python3 "$WD/drive_np2.py" run --port "$PORT" --session "$S" --out "$o" --windows "$w" --k "$K" "$@" >> "$CON" 2>&1
}

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN plan: NP=$NP ctx $CTX ub $UB boot $BIN (nsys launch) env KV_PAD=$LLAMA_ARG_KV_PAD STAGING=$GGML_SCHED_INPUT_STAGING CHAIN=$LLAMA_MTP_CHAIN FOLD_C=$LLAMA_MTP_FOLD_C; arms ${ARMS[*]}, window $WINDOWS -> $OUT/<arm>"
  cpu_checks; rc=$?; log "DRY_RUN cpu checks rc=$rc"; exit $rc
fi
trap cleanup EXIT
trap 'log "signal; stopping"; exit 130' INT TERM
cpu_checks || exit $?
gates || exit $?

for arm in "${ARMS[@]}"; do
  mkdir -p "$OUT/$arm"
  arm_env "$arm"
  log "arm $arm: META_INPUT_STAGE=${GGML_META_INPUT_STAGE-<unset>} RANK_ORDER=${GGML_META_INPUT_RANK_ORDER-<unset>} STAGE_ORDER=${GGML_CUDA_INPUT_STAGE_ORDER-<unset>}"
  boot "$arm" "$UB" || exit $?
  drive "$OUT/$arm" "$WINDOWS"; drc=$?
  [ "$(xid)" = 0 ] || { log "Xid after the cuda pass (arm $arm)"; exit 3; }
  log "arm $arm cuda pass driver rc=$drc"
  shutdown_server
  [ "$drc" = 0 ] || exit $(( drc == 5 ? 5 : drc == 7 ? 7 : 2 ))
done

# ---- CPU only from here
for arm in "${ARMS[@]}"; do
  f=$OUT/$arm/decode-8k.nsys-rep
  [ -s "$f" ] || { log "no report: $arm cuda pass"; exit 2; }
  [ "$(left)" -gt 90 ] || { log "over budget before exporting $arm"; exit 7; }
  timeout $(( $(left) - 60 )) nsys export --type sqlite --force-overwrite=true -o "$OUT/$arm/decode-8k.sqlite" "$f" > "$OUT/$arm/export.log" 2>&1 \
    || { log "$arm: sqlite export failed"; exit 2; }
done
python3 "$WD/k11b_analyze.py" "$OUT" "${ARMS[0]}" "${ARMS[@]:1}" > "$OUT/k11b-analysis.txt" 2>&1
log "analysis rc=$?: $(grep -a 'K11B-VERDICT' "$OUT/k11b-analysis.txt" | wc -l) verdict lines in $OUT/k11b-analysis.txt"
for arm in "${ARMS[@]:1}"; do
  if cmp -s "$OUT/${ARMS[0]}/proof-content.txt" "$OUT/$arm/proof-content.txt"; then
    log "TOKENS-MATCH $arm vs ${ARMS[0]}: yes (seeded proof request)"
  else
    log "TOKENS-MATCH $arm vs ${ARMS[0]}: NO (seeded proof request; base2 differing as well means the request is not deterministic across boots)"
  fi
done
[ "$(left)" -gt 0 ] || { log "over budget"; exit 7; }
log "done: $OUT/<arm>/{decode-8k.sqlite,runs.json}, $OUT/k11b-analysis.txt"
exit 0
