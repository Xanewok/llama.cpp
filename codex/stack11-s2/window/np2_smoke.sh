#!/usr/bin/env bash
# np2 joint-replay server smoke (copy of stack10's; stack10-pre-fix died here: GPF in memset under
# ggml_backend_meta_graph_compute_impl at the first joint replay after both 8k prefills). Server, argv and env of that probe.
# Arms, each one boot under gdb batch (backtrace on a fault):
#   fixed   3 joint replays: the server must stay healthy and both slots produce all tokens every time
#   mutant  GGML_META_SUBGRAPH_MUTANT=1 (pre-fix bound), 1 replay: the server must die, or the smoke is blind
# Exit: 0 pass | 8 fixed arm failed | 5 mutant survived (void control) | 4 GPUs or port busy | 3 Xid | 2 harness failure.
# Env: ROOT (tree with build-cuda), PORT, XID_SINCE, CUDA_VISIBLE_DEVICES. --preflight: CPU checks only.
set -uo pipefail

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../.." && pwd)}
BIN_DIR=$ROOT/build-cuda/bin
BIN=$BIN_DIR/round-trace
INPUTS=/home/xanewok/repos/attrib-pack/gpu/inputs
MODEL=$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
DRAFT_VOCAB=$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json
if [ "${1:-}" = --preflight ]; then
  command -v gdb > /dev/null && command -v curl > /dev/null && [ -s "$MODEL" ] && [ -s "$DRAFT_VOCAB" ] && [ -s "$INPUTS/chat-template.jinja" ] \
    && python3 -m py_compile "$HERE/np2_smoke.py" && exit 0
  exit 6
fi
OUT=${1:?out dir}
PORT=${PORT:?}
XID_SINCE=${XID_SINCE:-$(date '+%Y-%m-%d %H:%M:%S')}
K=5
CTX=204800; UB=512
mkdir -p "$OUT"
export TMPDIR=$OUT/tmp; mkdir -p "$TMPDIR"
T0=$(date +%s)
CON=$OUT/console.log
log() { echo "[$(date +%T) +$(( $(date +%s) - T0 ))s] $*" | tee -a "$CON"; }
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1; [[ $s != Z* ]]; }
port_busy() { ss -Hltn "sport = :$PORT" | grep -c . > /dev/null; }
healthy() { curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -c '"ok"' > /dev/null; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc; }
xid() { journalctl -k --since "$XID_SINCE" --no-pager | grep -cE '\bXid\b|fallen off'; }

ARGS=(-m "$MODEL" --alias local-benchmark -c "$CTX" -np 2 -ctk q8_0 -ctv q8_0 -fa on
  -ngl 99 -sm tensor --tensor-split 50,50
  --spec-type draft-mtp --spec-draft-n-max $K --spec-draft-n-min 0 --spec-draft-p-min 0
  --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-sampling probabilistic
  --spec-draft-backend-sampling --backend-sampling -t 8 -tb 8 --spec-draft-vocab "$DRAFT_VOCAB"
  -b 2048 -ub $UB --cache-ram 0 --jinja --chat-template-file "$INPUTS/chat-template.jinja"
  --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --host 127.0.0.1 --port "$PORT")

kids() { cat /proc/"$1"/task/*/children 2>/dev/null; }
stop_tree() {  # PID
  local p=$1 k
  alive "$p" || return 0
  for k in $(kids "$p"); do kill -TERM "$k" 2>/dev/null; done
  kill -TERM "$p" 2>/dev/null
  for _ in $(seq 60); do alive "$p" || return 0; sleep 1; done
  for k in $(kids "$p"); do kill -KILL "$k" 2>/dev/null; done
  kill -KILL "$p" 2>/dev/null
}

# ARM REPS ENV...: sets VERDICT to survived | died | boot-failed | client-error
run_arm() {
  local arm=$1 reps=$2; shift 2
  local d=$OUT/$arm pid
  mkdir -p "$d"
  VERDICT=harness
  [ "$(used)" -le 1500 ] || { log "REFUSE: GPUs hold $(used) MiB before arm $arm"; return 4; }
  port_busy && { log "REFUSE: port $PORT busy before arm $arm"; return 4; }
  (
    unset GGML_META_SUBGRAPH_MUTANT GGML_SSM_CONV_RS_OFF LLAMA_SYNC_ELIDE_DEBUG LLAMA_SYNC_ELIDE_MUTANT
    export GGML_SCHED_INPUT_STAGING=1 LLAMA_ARG_KV_PAD=4096 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=0 LLAMA_SYNC_ELIDE=0
    for kv in "$@"; do export "$kv"; done
    export LD_LIBRARY_PATH=$BIN_DIR
    exec gdb -q -batch -ex 'set pagination off' -ex 'set confirm off' \
      -ex 'handle SIGPIPE SIGUSR1 SIGUSR2 SIGCHLD SIGALRM SIGURG nostop noprint pass' -ex run \
      -ex 'bt 40' -ex 'info registers rip rsp rdi rsi rdx rcx' -ex 'x/6i $pc' -ex 'info threads' \
      --args "$BIN" "${ARGS[@]}" > "$d/server.log" 2>&1
  ) &
  pid=$!
  local i up=0
  for i in $(seq 420); do
    healthy && { up=1; break; }
    alive "$pid" || break
    sleep 1
  done
  if [ "$up" != 1 ]; then
    VERDICT=boot-failed
    log "arm $arm: server never healthy"; stop_tree "$pid"; wait "$pid" 2>/dev/null; return 0
  fi
  local sp; sp=$(kids "$pid"); [ -n "$sp" ] || sp=$pid
  tr '\0' '\n' < "/proc/${sp%% *}/environ" 2>/dev/null | grep -E '^(GGML|LLAMA)_' | sort > "$d/server-env.txt"
  log "arm $arm: healthy (env: $*)"
  timeout -k 10 900 python3 "$HERE/np2_smoke.py" --port "$PORT" --out "$d" --reps "$reps" >> "$CON" 2>&1
  sleep 3
  if ! healthy; then
    VERDICT=died
    sleep 10   # the dying process tears its GPU contexts down slowly
  elif python3 - "$d/replays.json" "$reps" <<'PY_OK'
import json, sys
r = json.load(open(sys.argv[1]))
ok = r['stage'] == 'done' and len(r['reps']) == int(sys.argv[2]) and \
     all(not any(x['error']) and x['tokens'] == [r['n_predict']] * 2 for x in r['reps'])
sys.exit(0 if ok else 1)
PY_OK
  then
    VERDICT=survived
  else
    VERDICT=client-error
  fi
  log "arm $arm: $VERDICT; $(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r['stage'],len(r['reps']),'reps',r.get('prefill'),r.get('client_exception',''))" "$d/replays.json" 2>&1)"
  stop_tree "$pid"
  wait "$pid" 2>/dev/null
  for _ in $(seq 60); do port_busy || break; sleep 1; done
  return 0
}

[ -x "$BIN" ] || { log "missing $BIN"; exit 2; }
[ "$(xid)" = 0 ] || { log "REFUSE: Xid"; exit 3; }
run_arm fixed 3 || exit $?
FIXED=$VERDICT
[ "$(xid)" = 0 ] || { log "Xid during the fixed arm"; exit 3; }
if [ "$FIXED" != survived ]; then
  log "FAIL fixed arm: $FIXED"; grep -a -A12 'received signal' "$OUT/fixed/server.log" >> "$CON"; exit 8
fi
run_arm mutant 1 GGML_META_SUBGRAPH_MUTANT=1 || exit $?
[ "$(xid)" = 0 ] || { log "Xid during the mutant arm"; exit 3; }
[ "$(grep -acxF 'GGML_META_SUBGRAPH_MUTANT=1' "$OUT/mutant/server-env.txt")" = 1 ] || { log "mutant env did not reach the server"; exit 2; }
if [ "$VERDICT" != died ]; then log "void control: the pre-fix mutant $VERDICT"; exit 5; fi
grep -aq 'ggml_backend_meta_graph_compute_impl' "$OUT/mutant/server.log" || log "note: mutant died outside graph_compute_impl (see server.log)"
log "PASS: fixed survived 3 joint replays; pre-fix mutant died"
exit 0
