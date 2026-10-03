#!/usr/bin/env bash
# Attribution pack, HyperQwen arm: nsys captures of steady k=5 decode at 8k and ~90k plus one cold 32k prefill on the
# codexh checkpoint (launch = hq-codexh-2x94k.sh with MAX_SEQS=1), then attrib_pack.py -> budget.{json,md}. One boot, np1.
# One nsys capture range per window (gpu/hq_drive.py opens/closes them through /start_profile and /stop_profile);
# `repeat-shutdown:N:sync` writes each range's report as it closes and shuts the server down after the Nth.
# Contract: servers are stopped only via stop_hq.sh (nsys is the serve.pid wrapper; SIGINT to it, our own PID, is the
# only other signal); never pgrep/pkill -f; a server never outlives the script; the pack must accept every window
# (>= 40 steady steps per decode, a prefill marker run) or the item fails with exit 8 after keeping all raw files.
# Unverified until the canary runs: the per-range report file names (matched by content then range order, see
# make_manifest.py) and the HQ prefill kernel names (absent from the existing HQ decode traces, so they cannot match a
# verify step; the parser rejects the window if no marker run is found).
# Output: $OUT/{hq*.nsys-rep,sqlite,runs.json,manifest.json,pack/budget.{json,md},serve.log,console.log}.
# Exit: 0 ok, 2 server/profiler/driver failure, 3 Xid, 4 GPUs/port busy, 6 inputs missing/changed, 7 wall budget exceeded,
#       8 pack rejected a trace.
# CANARY=1 runs prefill-32k then decode-8k (repeat-shutdown:2:sync): the multi-range path, the HQ prefill markers
# (attrib_pack MARKERS['hq']) and make_manifest.assign_hq are all exercised before the full run.
# Fallback if the canary shows repeat-shutdown does not write one report per range: one boot per window with
# --capture-range-end=stop-shutdown (proven by engine-plan/step0), i.e. run attrib_hq.sh once per window with WINDOWS=<one name> (it then uses stop-shutdown).
# Env: DRY_RUN (1 = print the plan, CPU preflight only), CANARY, OUT, BUDGET_S, XID_SINCE.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACK=$(dirname "$HERE")
DRY_RUN=${DRY_RUN:-1}
CANARY=${CANARY:-0}
# ATTRIB_SP / ATTRIB_HQ / ATTRIB_PORT exist for tests/test_gpu_drivers.py (stubbed stop_hq.sh and HQ dir); the defaults are the real ones.
SP=${ATTRIB_SP:-/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad}
Q=$HOME/repos/qwen38-eval
HQ=${ATTRIB_HQ:-$HOME/repos/HyperQwen}
OV=$SP/vllm-both/overlay
VSP=$HQ/venv/lib/python3.12/site-packages/vllm
MODEL=$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh
OUT=${OUT:-$PACK/runs/attrib-hq}
PORT=${ATTRIB_PORT:-8081}
URL=http://127.0.0.1:$PORT
K=5
UNIT=16
MAX_LEN=102400
XID_SINCE=${XID_SINCE:-"2026-10-03 18:57:00"}
BUDGET_S=${BUDGET_S:-1650}
BOOT_WAIT_S=750
WINDOWS=${WINDOWS:-prefill-32k,decode-8k,decode-90k}
[ "$CANARY" = 1 ] && WINDOWS=prefill-32k,decode-8k
NWIN=$(echo "$WINDOWS" | tr ',' '\n' | wc -l)
END_MODE=repeat-shutdown:$NWIN:sync
[ "$NWIN" = 1 ] && END_MODE=stop-shutdown  # the mode step0 proved

T0=$(date +%s)
log() { echo "[$(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
die() { log "ABORT: $*"; exit "${2:-6}"; }
left() { echo $(( T0 + BUDGET_S - $(date +%s) )); }
xid_count() { journalctl -k --no-pager --since "$XID_SINCE" | grep -cE '\bXid\b|fallen off'; }
gpu_max_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc; }
ram_avail_mib() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
port_busy() { ss -Hltn "sport = :$PORT" | grep -q .; }
gfail() { if [ "$DRY_RUN" = 1 ]; then log "DRY note: the run gate would stop on: $*"; else die "$*" 4; fi; }

gates() {
  local x m r
  x=$(xid_count); m=$(gpu_max_used); r=$(ram_avail_mib)
  log "gates: kernel_log_faults=$x gpu_max_used=${m}MiB ram_avail=${r}MiB port_held=$(port_busy && echo yes || echo no)"
  [ "$x" = 0 ] || { if [ "$DRY_RUN" = 1 ]; then log "DRY note: the run gate would stop on kernel-log faults since $XID_SINCE"; else die "Xid/fallen-off since $XID_SINCE" 3; fi; }
  [ "${m:-99999}" -le 750 ] || gfail "GPU memory in use ${m} MiB > 750"
  [ "$r" -ge 2048 ] || gfail "RAM available ${r} MiB < 2048"
  port_busy && gfail "port $PORT held by another process"
  if [ -e "$HQ/serve.pid" ]; then
    local w; w=$(cat "$HQ/serve.pid")
    tr '\0' ' ' < "/proc/$w/cmdline" 2>/dev/null | grep -q 'start_qwen.sh\|vllm serve' && gfail "HyperQwen server $w still running"
    [ "$DRY_RUN" = 1 ] || rm -f "$HQ/serve.pid"
  fi
}

cpu_checks() {
  [ -f "$MODEL/config.json" ] && [ -f "$MODEL/mtp_draft_vocab_ids.pt" ] || die "$MODEL lacks config.json / mtp_draft_vocab_ids.pt"
  [ -d "$OV/vllm" ] || die "no overlay $OV"
  [ -f "$SP/vllm-ar/build/libhsar_sm120.so" ] || die "no libhsar_sm120.so"
  [ -f "$Q/corpora/templates/swift-leading-instructions-v1.jinja" ] || die "no chat template"
  (cd "$HERE/inputs" && sha256sum -c --quiet ../inputs.sha256) || die "frozen corpus/client changed"
  command -v nsys > /dev/null || die "nsys missing"
  [[ "$(nsys profile --help 2>&1)" == *repeat-shutdown* ]] || die "nsys lacks --capture-range-end=repeat-shutdown"
  grep -q 'profiler_config' "$VSP/engine/arg_utils.py" || die "this vLLM has no --profiler-config"
  patch -p1 -R --dry-run -s -d "$VSP" < "$HQ/patches/minp-spec-decode-v2-responses.patch" >/dev/null || die "min_p patch not applied to the venv"
  [ "$(cd "$HQ" && CUDA_VISIBLE_DEVICES= PYTHONPATH=$OV venv/bin/python -c '
import vllm, vllm.envs as e, vllm.v1.worker.gpu.sample.fast_topk as ft, vllm.distributed.device_communicators.host_staged_all_reduce as h
print("OK" if all("/overlay/vllm/" in m.__file__ for m in (vllm, e, ft, h)) and hasattr(e, "VLLM_FAST_TOPK") and hasattr(e, "VLLM_HOST_STAGED_AR") else "BAD")' 2>/dev/null | tail -1)" = OK ] \
    || die "overlay does not shadow the venv"
  python3 -m py_compile "$PACK/attrib_pack.py" "$HERE/hq_drive.py" "$HERE/make_manifest.py" || die "python compile failed"
  python3 "$PACK/attrib_pack.py" pack --help > /dev/null || die "attrib_pack.py does not start"
  log "preflight ok (model, draft vocab, overlay, min_p patch, profiler flag, nsys repeat-shutdown, inputs)"
}

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN plan: boot codexh k=$K MAX_SEQS=1 under nsys ($END_MODE); windows $WINDOWS; export sqlite; pack -> $OUT/pack"
  gates; cpu_checks; exit 0
fi

mkdir -p "$OUT" "$OUT/tmp"
if compgen -G "$OUT/*" > /dev/null; then
  prev=$OUT/prev-$(date +%Y%m%d-%H%M%S); mkdir -p "$prev"; find "$OUT" -maxdepth 1 -type f -exec mv {} "$prev"/ \;
fi
exec > >(tee -a "$OUT/console.log") 2>&1

# nsys moves the app into its own session, so stop_hq.sh may miss members: record (pid, start tick) of the wrapper's
# process tree and session while it is alive (pattern: engine-plan/step0/run.sh server_pids), and reap survivors by exact PID.
session_pids() { ps -e -o pid=,sid= | awk -v s="$1" '$2 == s { print $1 }'; }
server_pids() {
  { session_pids "$1"
    ps -e -o pid=,ppid= | awk -v root="$1" '{ parent[$1] = $2 } END {
      for (p in parent) { q = p; while (q in parent && q != root && q > 1) q = parent[q]; if (q == root) print p } }'
  } | sort -un
}
start_tick() { awk '{ print $22 }' "/proc/$1/stat" 2>/dev/null; }
MEMBERS=()
record_members() {
  local p t
  for p in $(server_pids "$1"); do
    t=$(start_tick "$p"); [ -n "$t" ] && MEMBERS+=("$p:$t")
  done
}
reap_members() {
  local m p t i
  for m in "${MEMBERS[@]}"; do
    p=${m%%:*}; t=${m##*:}
    [ "$(start_tick "$p")" = "$t" ] || continue
    log "survivor $p ($(cat /proc/"$p"/comm 2>/dev/null)) after stop_hq.sh: TERM"
    kill -TERM "$p" 2>/dev/null
    for i in $(seq 1 15); do [ "$(start_tick "$p")" = "$t" ] || break; sleep 1; done
    [ "$(start_tick "$p")" = "$t" ] && { log "survivor $p still alive: KILL"; kill -KILL "$p" 2>/dev/null; }
  done
  MEMBERS=()
}
stop_server() {
  bash "$SP/stop_hq.sh"
  reap_members
  local i
  for i in $(seq 1 90); do
    [ "$(gpu_used)" -lt 1500 ] && ! port_busy && return 0
    sleep 2
  done
  log "WARNING: GPU memory / port not free 180 s after stop_hq.sh (gpu_used=$(gpu_used) MiB)"
  return 1
}
# LAUNCHED is set only once launch_real wrote our own wrapper PID: a gate refusal must never stop a foreign HQ server.
LAUNCHED=
trap 'rc=$?; { [ -z "$LAUNCHED" ] || [ ! -e "$HQ/serve.pid" ]; } || stop_server; reap_members; log "exit rc=$rc"' EXIT
trap 'log "signal"; exit 130' INT TERM

gates
cpu_checks

launch_real() {
  local extra="--tensor-parallel-size 2 --disable-custom-all-reduce --prefix-match-unit $UNIT --served-model-name local-benchmark --chat-template $Q/corpora/templates/swift-leading-instructions-v1.jinja --override-generation-config {\"temperature\":1.0,\"top_k\":20,\"top_p\":0.95,\"min_p\":0.05,\"max_new_tokens\":32768} --profiler-config {\"profiler\":\"cuda\",\"max_iterations\":1000}"
  cd "$HQ" || die "cd $HQ"
  # TMPDIR: nsys stages in it and / has < 1 GB free; the app keeps the default.
  setsid nohup env PYTHONPATH=$OV VLLM_HOST_STAGED_AR=1 VLLM_HOST_STAGED_AR_VARIANT=classic VLLM_FAST_TOPK=1 \
    VLLM_HOST_STAGED_AR_LIB=$SP/vllm-ar/build/libhsar_sm120.so \
    CUDA_HOME=/usr/local/cuda-13.2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 \
    VLLM_SERVER_DEV_MODE=1 VLLM_USE_V2_MODEL_RUNNER=1 DRAFT_TOKENS=$K HOST=127.0.0.1 PORT=$PORT GPU_UTIL=0.86 \
    MODEL="$MODEL" SPEC=mtp CTX=long MAX_LEN=$MAX_LEN MAX_SEQS=1 PREFIX_CACHE=1 \
    EXTRA_ARGS="$extra" TMPDIR="$OUT/tmp" \
    nsys profile -o "$OUT/hq" --force-overwrite=true -t cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true \
      --capture-range=cudaProfilerApi --capture-range-end=$END_MODE --kill=sigterm --sample=none --cpuctxsw=none \
      env -u TMPDIR bash single-user/start_qwen.sh > "$OUT/serve.log" 2>&1 < /dev/null & echo $! > "$HQ/serve.pid"
  LAUNCHED=1
  sleep 2
  tr '\0' ' ' < "/proc/$(cat "$HQ/serve.pid")/cmdline" 2>/dev/null | grep -q 'start_qwen.sh\|vllm serve' \
    || { log "serve.pid $(cat "$HQ/serve.pid") is not the launch wrapper"; return 1; }
}

launch_real || { [ -z "$LAUNCHED" ] || stop_server; exit 2; }
WRAP=$(cat "$HQ/serve.pid")
for i in $(seq 1 $(( BOOT_WAIT_S / 5 ))); do
  curl -s -m 3 "$URL/health" >/dev/null 2>&1 && break
  kill -0 "$WRAP" 2>/dev/null || { tail -30 "$OUT/serve.log"; log "died during boot"; stop_server; exit 2; }
  [ "$(left)" -gt 0 ] || { log "over budget while booting"; stop_server; exit 7; }
  sleep 5
done
curl -sf -m 3 "$URL/health" >/dev/null || { tail -20 "$OUT/serve.log"; log "not healthy in ${BOOT_WAIT_S}s"; stop_server; exit 2; }
log "healthy after $(( $(date +%s) - T0 )) s"
record_members "$WRAP"
log "recorded ${#MEMBERS[@]} server process(es) for the survivor reap"

slog=$OUT/serve.log
grep -q 'Using V2 Model Runner' "$slog" || die "not on the V2 runner" 2
be=$(grep -m1 -o "Using \[.*\] all-reduce backends" "$slog")
n=$(grep -c 'host-staged all-reduce: rank' "$slog")
dv=$(grep -c 'MTP drafter uses a [0-9]*-token draft head' "$slog")
blk=$(grep -m1 -oE 'Setting attention block size to [0-9]+' "$slog" | grep -oE '[0-9]+$')
kv=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$slog" | grep -oE '[0-9,]+' | tr -d ',')
grep -q 'host-staged all-reduce disabled' "$slog" && die "host-staged setup failed" 2
[ "$n" -ge 2 ] && [[ "$be" == *HOST_STAGED* ]] || die "host-staged not active" 2
[ "$dv" -ge 2 ] || die "MTP draft vocab head not in use" 2
[ -n "$blk" ] && [ $(( blk % UNIT )) = 0 ] || die "unit $UNIT does not divide block '${blk:-?}'" 2
grep -q "num_speculative_tokens': $K" "$slog" || die "speculative k != $K" 2
[ "${kv:-0}" -ge 93000 ] || die "KV pool ${kv:-?} tokens < 93000: the 90k window cannot fit" 2
log "proof: '$be'; host-staged lines $n; draft-head lines $dv; block $blk; k=$K; KV pool $kv"

chk=$(curl -s -m 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d \
  '{"model":"local-benchmark","messages":[{"role":"user","content":"Write a Python function that checks if a number is prime."}],"max_tokens":400}')
echo "$chk" | python3 -c "import sys,json; d=json.load(sys.stdin); print('warm-up (server-default min_p) ok, tokens', d['usage']['completion_tokens'])" \
  || die "warm-up request failed: $chk" 2
sleep 2
for kk in _candidates_kernel _select_kernel; do
  grep -q "JIT compilation during inference: $kk" "$slog" || die "fast-topk kernel $kk not launched" 2
done
[ "$(xid_count)" = 0 ] || die "Xid after boot" 3

record_members "$WRAP"
timeout -k 10 $(( $(left) - 300 )) python3 "$HERE/hq_drive.py" --url "$URL" --out "$OUT" --windows "$WINDOWS"
drc=$?
log "driver rc=$drc"
# After the last range nsys writes its report and shuts the server down; only then are the files complete.
for w in $(seq 1 60); do kill -0 "$WRAP" 2>/dev/null || break; sleep 2
  if [ "$w" = 30 ]; then log "nsys still alive 60 s after the driver; SIGINT to nsys (our own PID)"; kill -INT "$WRAP" 2>/dev/null; fi
done
stop_server
[ "$(xid_count)" = 0 ] || die "Xid after the captures" 3
[ "$drc" = 0 ] || exit 2

reps=$(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l)
[ "$reps" = "$NWIN" ] || { log "expected $NWIN nsys reports, found $reps: $(ls "$OUT"/*.nsys-rep 2>/dev/null | tr '\n' ' ')"; exit 2; }
for rep in "$OUT"/*.nsys-rep; do
  [ "$(left)" -gt 60 ] || { log "over budget before exporting $rep"; exit 7; }
  timeout $(( $(left) - 30 )) nsys export --type sqlite --force-overwrite=true -o "${rep%.nsys-rep}.sqlite" "$rep" \
    > "${rep%.nsys-rep}.export.log" 2>&1 || { log "$rep: sqlite export failed"; exit 2; }
done
python3 "$HERE/make_manifest.py" --engine hq --model hq-swift15-codexh.json --out-dir "$OUT" --k "$K" \
  --title "HyperQwen codexh np1 k=$K, host-staged AR + fast top-k, 2x RTX 5060 Ti (measured)" --manifest "$OUT/manifest.json" || exit 2
python3 "$PACK/attrib_pack.py" pack "$OUT/manifest.json" --out "$OUT/pack"
prc=$?
log "pack rc=$prc"
[ "$prc" = 0 ] || exit 8
[ "$(left)" -gt 0 ] || { log "over budget"; exit 7; }
log "done: $OUT/pack/budget.md"
