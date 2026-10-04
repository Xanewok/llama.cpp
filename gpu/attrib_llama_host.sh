#!/usr/bin/env bash
# attrib-llama-host: which host functions fill llama.cpp's STEP time (GPU0 idle between kernels) in an MTP decode round.
# Same server launch as attrib_llama.sh (frozen stack2, S5 np1 flags, 2x RTX 5060 Ti) and the same nsys CUDA trace, plus CPU
# IP sampling with backtraces on the server process tree during each decode window: nsys start --sample=process-tree
# --backtrace=dwarf --sampling-period=2336904 --samples-per-backtrace=1.
#   backtrace dwarf: the frozen libs are built without frame pointers (274 of 7131 libllama functions set up rbp), so fp
#     unwinding skips frames (measured on a no-fp C chain: "[Broken backtraces]"); lbr needs Intel; dwarf walks .eh_frame.
#   period 2336904 cycles (nsys' non-dwarf default, ~1.8 kHz at ~4.2 GHz = 0.56 ms/sample, measured on this CPU) with a
#     chain on EVERY sample: spb 1, since the first llama/ggml frame needs the stack (the dwarf default would be 0.9 kHz, spb 4).
#   window 6 s (attrib-llama: 4 s): ~85 steady rounds at 8k, ~70 at 90k -> ~900 / ~1400 samples inside STEP (derived).
# Windows (gpu/llama_drive.py): decode-8k, decode-90k, one nsys start/stop each. CANARY=1: decode-8k only, short requests.
# Deviation from S5: --cache-ram 0 (a host prompt-cache save must not land inside a window); no `nice`
# (host gaps are an axis being measured).
# Contract: only our own PIDs are stopped (nsys session by name, then the exact PID); the pack must accept every window
# (>= 40 steady rounds per decode) and host_sampling.py must find >= 200 decode-thread samples inside STEP per window, or
# the item fails with exit 8 after keeping all raw files.
# Output: $OUT/{<window>.nsys-rep,sqlite,runs.json,manifest.json,pack/budget.{json,md},host/host.{json,md},server.log,console.log}.
# Exit: 0 ok, 2 server/nsys/driver failure or sampling unavailable, 3 Xid (the only hazard), 4 GPUs/port busy, 6 frozen input
#       changed, 7 wall budget exceeded (manifest maps it to timeout), 8 pack or host attribution rejected a trace.
# Env: DRY_RUN (1 = print the plan and run CPU preflight only), CANARY, OUT, BUDGET_S, PORT (gpuq {port}), XID_SINCE (journal
#      --since; default this boot), BACKTRACE / SAMPLING_PERIOD / SAMPLES_PER_BT / WINDOW_S / DELAY_S / N_PREDICT / TWIN_N (capture
#      shape), BASELINE_BUDGET (attrib-llama budget.json for the sampling-only delta), SERVER_BIN (CPU stub tests only),
#      CPU_REHEARSAL=1 (4B model on the CPU, no GPU checks, port 8394, small windows; the pack is then EXPECTED to reject:
#      exit 8 proves the negative control).
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
OUT=${OUT:-$PACK/runs/attrib-llama-host}
PORT=${PORT:-${GPUQ_PORT:-8081}}
K=5
BUDGET_S=${BUDGET_S:-1350}
MEM_FREE_MIB=1500
WINDOWS=decode-8k,decode-90k
CTX=102400
EXTRA=(-ngl 99 -sm tensor --tensor-split 50,50
  --spec-type draft-mtp --spec-draft-n-max $K --spec-draft-n-min 0 --spec-draft-p-min 0
  --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-sampling probabilistic
  --spec-draft-backend-sampling --backend-sampling -t 8 -tb 8 --spec-draft-vocab "$DRAFT_VOCAB")
BACKTRACE=${BACKTRACE:-dwarf}
SAMPLING_PERIOD=${SAMPLING_PERIOD:-2336904}
SAMPLES_PER_BT=${SAMPLES_PER_BT:-1}
WINDOW_S=${WINDOW_S:-6}
DELAY_S=${DELAY_S:-1.5}
N_PREDICT=${N_PREDICT:-4000}
TWIN_N=${TWIN_N:-1000}
if [ "$CANARY" = 1 ]; then WINDOWS=decode-8k; N_PREDICT=${N_PREDICT_CANARY:-1500}; TWIN_N=300; fi
SAMPLE_ARGS="--sample=process-tree --backtrace=$BACKTRACE --samples-per-backtrace=$SAMPLES_PER_BT --sampling-period=$SAMPLING_PERIOD"
BASELINE_BUDGET=${BASELINE_BUDGET:-$PACK/runs/attrib-llama/full-a1/pack/budget.json}
DRV_EXTRA=(--window "$WINDOW_S" --delay "$DELAY_S" --n-predict "$N_PREDICT" --twin-n "$TWIN_N" --nsys-sample-args "$SAMPLE_ARGS")
BIN=${SERVER_BIN:-$STACK/weekend-srv}
if [ -n "${CPU_REHEARSAL:-}" ]; then
  PORT=8394; MODEL=$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf; MODEL_BYTES=; CTX=8192; K=3
  EXTRA=(-ngl 0 -t 4); DRV_EXTRA+=(--tokens decode-8k=700,decode-90k=2000 --n-predict 400 --twin-n 100)
  BIN=${REHEARSAL_BIN:?set REHEARSAL_BIN to a renamed copy of the server binary (never named llama-server)}
  export CUDA_VISIBLE_DEVICES=
  BUDGET_S=${BUDGET_S:-900}
else
  export CUDA_VISIBLE_DEVICES=$UUIDS
fi
XID_SINCE=${XID_SINCE:-}
S=attribhost$$
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
  python3 -m py_compile "$PACK/attrib_pack.py" "$PACK/host_sampling.py" "$HERE/llama_drive.py" "$HERE/make_manifest.py" || { log "python compile failed"; return 6; }
  python3 "$PACK/attrib_pack.py" pack --help > /dev/null || { log "attrib_pack.py does not start"; return 6; }
  python3 "$PACK/host_sampling.py" --help > /dev/null || { log "host_sampling.py does not start"; return 6; }
  sampling_checks || return $?
}
# CPU sampling needs: perf_event access for user space, an nsys that knows the flags, and symbols + unwind tables in the
# frozen libs (nm -D: exported API; .symtab: static functions such as the graph-build helpers; .eh_frame: dwarf unwinding).
sampling_checks() {
  local p; p=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 9)
  [ "$p" -le 2 ] || { log "perf_event_paranoid=$p > 2: nsys CPU sampling is unavailable"; return 2; }
  # no `grep -q` after a pipe: under pipefail the early exit would SIGPIPE the producer and fail the check
  [ -n "$(nsys start --help 2>&1 | grep -e '--samples-per-backtrace')" ] || { log "this nsys has no --samples-per-backtrace"; return 2; }
  local l
  for l in libllama.so libllama-common.so libllama-server-impl.so libggml-base.so libggml-cuda.so libggml-cpu.so; do
    [ -f "$STACK/$l" ] || { log "missing $STACK/$l"; return 6; }
    [ -n "$(readelf -S "$STACK/$l" 2>/dev/null | grep -F '.symtab')" ] || { log "$l has no .symtab: frames would be unnamed"; return 6; }
    [ -n "$(readelf -S "$STACK/$l" 2>/dev/null | grep -F '.eh_frame')" ] || { log "$l has no .eh_frame: dwarf unwinding would fail"; return 6; }
  done
  [ -n "$(nm -D --defined-only "$STACK/libllama.so" | grep -w llama_decode)" ] || { log "libllama.so does not export llama_decode"; return 6; }
  [ -n "$(nm -D --defined-only "$STACK/libggml-base.so" | grep -w ggml_backend_sched_graph_compute)" ] || { log "libggml-base.so does not export ggml_backend_sched_graph_compute"; return 6; }
  log "sampling preflight ok: perf_event_paranoid=$p, $BACKTRACE backtraces, period $SAMPLING_PERIOD, libs have symbols and unwind tables"
}
gates() {
  [ "$(xid)" = 0 ] || { log "REFUSE: Xid/fallen-off in the kernel log"; return 3; }
  local u; u=$(used)
  [ "$u" -le "$MEM_FREE_MIB" ] || { log "REFUSE: GPUs hold $u MiB (> $MEM_FREE_MIB)"; return 4; }
  port_busy && { log "REFUSE: port $PORT busy"; return 4; }
  log "gates ok: GPUs ${u} MiB, port $PORT free; windows $WINDOWS"
}

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN plan: boot $BIN (nsys launch session) np1 k=$K ctx $CTX; windows $WINDOWS; nsys start $SAMPLE_ARGS; export sqlite; pack -> $OUT/pack; host attribution -> $OUT/host"
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
[ "$drc" = 0 ] || exit 2

# Export is CPU-only; the GPUs are free from here.
for w in ${WINDOWS//,/ }; do
  [ -s "$OUT/$w.nsys-rep" ] || { log "$w: no report"; exit 2; }
  [ "$(left)" -gt 60 ] || { log "over budget before exporting $w"; exit 7; }
  timeout $(( $(left) - 30 )) nsys export --type sqlite --force-overwrite=true -o "$OUT/$w.sqlite" "$OUT/$w.nsys-rep" \
    > "$OUT/export-$w.log" 2>&1 || { log "$w: sqlite export failed"; exit 2; }
done
python3 "$HERE/make_manifest.py" --engine llama --model llama-swift15-iq4xs.json --out-dir "$OUT" --k "$K" \
  --title "llama.cpp stack2 Swift 1.5 IQ4_XS np1 k=$K, S5 flags, 2x RTX 5060 Ti, nsys CPU sampling ($BACKTRACE, period $SAMPLING_PERIOD) (measured)" \
  --manifest "$OUT/manifest.json" || exit 2
python3 "$PACK/attrib_pack.py" pack "$OUT/manifest.json" --out "$OUT/pack" >> "$CON" 2>&1
prc=$?
log "pack rc=$prc"
[ "$prc" = 0 ] || exit 8
BL=(); [ ! -f "$BASELINE_BUDGET" ] || BL=(--baseline "$BASELINE_BUDGET")
python3 "$PACK/host_sampling.py" "$OUT/manifest.json" --out "$OUT/host" "${BL[@]}" >> "$CON" 2>&1
hrc=$?
log "host rc=$hrc"
[ "$hrc" = 0 ] || exit 8
[ "$(left)" -gt 0 ] || { log "over budget"; exit 7; }
log "done: $OUT/host/host.md"
