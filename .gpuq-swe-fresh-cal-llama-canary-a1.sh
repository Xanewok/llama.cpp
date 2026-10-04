#!/usr/bin/env bash
# Parent-only (gpuq) stage runner: boot one server, run run_swe_fresh.py against it, stop the server.
# Server launch, idle/hazard gates and PID handling are copied from two-agent/codex/gpu_ab.sh
# (contract cases 6-9, 11, 13); it differs in: AGENTS slots, no assets/baseline, a disk gate.
# Never run on a GPU outside gpuq. CPU checks: preflight.sh.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENGINE=${ENGINE:?llama or hq}
PINS=${PINS:?identity JSON from make_manifest.py pin}
TASKS_FILE=${TASKS_FILE:?JSON with .tasks[].instance_id}
CANARY_TASK=${CANARY_TASK:?one short instance id}
OUT=${OUT:?fresh output directory inside the repo}
ONLY=${ONLY:-all} # all | canary | measure (measure requires this OUT's passed, pinned canary)
BUDGET_S=${BUDGET_S:-14400}
CAP_S=${CAP_S:-1200}
CANARY_CAP_S=${CANARY_CAP_S:-300}
AGENTS=${AGENTS:-1}
REPEAT=${REPEAT:-1}
PROVIDER=${PROVIDER:-local}
HARBOR_VERSION=${HARBOR_VERSION:-0.23.0}
DATASET_PATH=${DATASET_PATH:?pinned Harbor dataset snapshot (harbor download), see CALIBRATION.md}
MIN_FREE_GB=${MIN_FREE_GB:-30}
PORT=${PORT:-8081}
XID_SINCE=${XID_SINCE:?acknowledged kernel-log cutoff}
Q=/home/xanewok/repos/qwen38-eval
HQ=/home/xanewok/repos/HyperQwen
SP=/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad
HS=/home/xanewok/repos/two-agent/codex/harness_support.py
BINROOT=${LLAMA_BINROOT:-/home/xanewok/repos/qwen38-weekend/frozen/stack2/build/bin}
TEMPLATE=$Q/corpora/templates/swift-leading-instructions-v1.jinja
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
URL=http://127.0.0.1:$PORT
SERVER_CTX=102400 # window 98304 + room for one largest tool output (Codex does not recover from a 400 overflow)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
PY=(nice -n 19 python3)
case "$ENGINE:$ONLY" in llama:all|llama:canary|llama:measure|hq:all|hq:canary|hq:measure) ;; *) exit 6 ;; esac
case "$AGENTS" in 1|2) ;; *) exit 6 ;; esac
OUT=$(realpath -m "$OUT")
PINS=$(realpath -e "$PINS")
TASKS_FILE=$(realpath -e "$TASKS_FILE")
DATASET_PATH=$(realpath -e "$DATASET_PATH")
case "$OUT/" in "$ROOT/"*) ;; *) printf 'OUT must be inside the repo\n' >&2; exit 6 ;; esac
mkdir -p "$OUT"
T0=$(date +%s)
DEADLINE=$(( T0 + BUDGET_S ))
SPID=; SERVER_START=; STAGE_DEADLINE=$DEADLINE
left() { echo $(( STAGE_DEADLINE - $(date +%s) )); }
fail() { printf '%s\n' "$1" >> "$OUT/status.txt"; exit "${2:-2}"; }
budget() { [ "$(left)" -gt "$1" ] || fail budget 7; }
hazard() {
  journalctl -k --no-pager --since "$XID_SINCE" > "$OUT/kernel.log" || fail kernel-log-unreadable 3
  local count
  count=$("${PY[@]}" "$HS" xid --input "$OUT/kernel.log") || fail xid-parser 3
  [ "$count" = 0 ] || fail hazard 3
}
idle() {
  while true; do
    budget 15; hazard
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv > "$OUT/memory.csv" || fail memory-query 4
    local memory ram
    memory=$("${PY[@]}" "$HS" memory --input "$OUT/memory.csv") || fail memory-parser 4
    ram=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    if awk "BEGIN {exit !($memory <= 750 && $ram >= 4096)}"; then break; fi
    sleep 2
  done
  if ss -Hltn "sport = :$PORT" | grep -q .; then fail port-busy 4; fi
}
disk() {
  # Harbor leaves the pulled base layers in BuildKit's cache after --delete (measured, CALIBRATION.md).
  local free_gb
  free_gb=$(df -B1G --output=avail /mnt/scratch | tail -1 | tr -d ' ')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || fail "disk-low-${free_gb}GB" 5
}
identity() {
  "${PY[@]}" "$ROOT/make_manifest.py" verify --pins "$PINS" || fail pinned-input-changed 6
}
docker_ready() {
  docker info > /dev/null 2>&1 || fail docker-unavailable 5
  timeout 120 uvx "harbor==$HARBOR_VERSION" --version > "$OUT/harbor.version" 2>&1 || fail harbor-unavailable 5
}
start_tick() { awk '{print $22}' "/proc/$1/stat"; }
owned_alive() {
  [ -n "$SPID" ] && [ -e "/proc/$SPID/stat" ] && [ "$(start_tick "$SPID")" = "$SERVER_START" ] &&
    [ "$(awk '{print $3}' "/proc/$SPID/stat")" != Z ]
}
stop_server() {
  if owned_alive; then
    if [ "$ENGINE" = hq ]; then
      [ "$(cat "$HQ/serve.pid")" = "$SPID" ] || fail hq-pid-file-changed 6
      timeout -k 2 15 bash "$SP/stop_hq.sh" || fail hq-stop-failed 2
      for _ in $(seq 1 20); do owned_alive || break; sleep 1; done
      owned_alive && fail hq-stop-incomplete 2
    else
      kill -TERM "$SPID" 2>/dev/null || true
      for _ in $(seq 1 20); do owned_alive || break; sleep 1; done
      if owned_alive; then kill -KILL "$SPID" 2>/dev/null || true; fi
    fi
  fi
  if [ -n "$SPID" ]; then wait "$SPID" 2>/dev/null || true; fi
  SPID=; SERVER_START=
}
RPID=; RNAME=
harbor_backstop() { # recorded Harbor groups/compose projects of unfinished attempts, if the runner died before its own cleanup
  "${PY[@]}" "$ROOT/run_swe_fresh.py" cleanup --campaign "$OUT/$1" >> "$OUT/cleanup.log" 2>&1 || true
}
cleanup() {
  local rc=$?; trap - EXIT
  if [ -n "$RPID" ]; then
    kill -INT "$RPID" 2>/dev/null || true; wait "$RPID" 2>/dev/null || true; harbor_backstop "$RNAME"
  fi
  stop_server
  printf 'exit %s\n' "$rc" >> "$OUT/status.txt"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

launch() {
  local stage=$1
  if [ "$ENGINE" = llama ]; then
    (cd "$BINROOT" && sha256sum -c --quiet SHA256SUMS) || fail frozen-binary-changed 6
    CUDA_VISIBLE_DEVICES=$UUIDS LD_LIBRARY_PATH=$BINROOT nice -n 19 prlimit --stack=unlimited:unlimited "$BINROOT/weekend-srv" \
      -m "$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf" --alias local-benchmark \
      -c $(( SERVER_CTX * AGENTS )) -np "$AGENTS" -ctk q8_0 -ctv q8_0 -ngl 99 -fa on -sm tensor --tensor-split 50,50 \
      -b 2048 -ub 128 --cache-ram 0 --jinja --chat-template-file "$TEMPLATE" \
      --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --host 127.0.0.1 --port "$PORT" --metrics \
      --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-n-min 0 --spec-draft-p-min 0 \
      --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-sampling probabilistic \
      --spec-draft-backend-sampling --backend-sampling -t 8 -tb 8 \
      --spec-draft-vocab "$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json" \
      > "$OUT/$stage.server.log" 2>&1 &
    SPID=$!
  else
    if [ -e "$HQ/serve.pid" ]; then fail existing-hq-pid-file 4; fi
    local extra
    extra="--tensor-parallel-size 2 --disable-custom-all-reduce --prefix-match-unit 16 --served-model-name local-benchmark --chat-template $TEMPLATE --override-generation-config {\"temperature\":1.0,\"top_k\":20,\"top_p\":0.95,\"min_p\":0.05,\"max_new_tokens\":32768}"
    cd "$HQ"
    setsid nohup env PYTHONPATH="$SP/vllm-both/overlay" VLLM_HOST_STAGED_AR=1 VLLM_HOST_STAGED_AR_VARIANT=classic \
      VLLM_FAST_TOPK=1 VLLM_HOST_STAGED_AR_LIB="$SP/vllm-ar/build/libhsar_sm120.so" \
      CUDA_HOME=/usr/local/cuda-13.2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
      NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_SERVER_DEV_MODE=1 VLLM_USE_V2_MODEL_RUNNER=1 \
      DRAFT_TOKENS=5 DRAFT_SAMPLE=probabilistic HOST=127.0.0.1 PORT="$PORT" GPU_UTIL=0.86 \
      MODEL="$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh" \
      SPEC=mtp CTX=long MAX_LEN=$SERVER_CTX MAX_SEQS="$AGENTS" PREFIX_CACHE=1 EXTRA_ARGS="$extra" \
      nice -n 19 bash single-user/start_qwen.sh > "$OUT/$stage.server.log" 2>&1 < /dev/null &
    SPID=$!
    printf '%s\n' "$SPID" > "$HQ/serve.pid"
    cd "$ROOT"
  fi
  SERVER_START=$(start_tick "$SPID")
  printf '%s %s\n' "$SPID" "$SERVER_START" > "$OUT/$stage.pid"
}

stage() {
  local name=$1 rc=0 flags=() tasks=()
  STAGE_DEADLINE=$DEADLINE
  if [ "$name" = canary ]; then
    STAGE_DEADLINE=$(( $(date +%s) + 1500 )) # boot (HQ up to 900) + image + Codex install (~170 s measured) + cap
    [ "$STAGE_DEADLINE" -le "$DEADLINE" ] || STAGE_DEADLINE=$DEADLINE
    flags=(--canary --cap-s "$CANARY_CAP_S" --hard-timeout-s $(( CANARY_CAP_S + 900 )))
    tasks=(--tasks "$CANARY_TASK")
  else
    flags=(--cap-s "$CAP_S" --hard-timeout-s $(( CAP_S + 900 )))
    tasks=(--tasks-file "$TASKS_FILE" --repeat "$REPEAT")
  fi
  if [ -e "$OUT/$name" ]; then fail output-already-exists 6; fi
  idle; disk
  budget 60
  launch "$name"
  until curl -sf --max-time 2 "$URL/health" > "$OUT/$name.health.json"; do
    budget 60; hazard; owned_alive || fail server-exited 2
    sleep 1
  done
  cp "/proc/$SPID/limits" "$OUT/$name.limits.txt" 2>/dev/null || true
  local seconds=$(( $(left) - 60 ))
  # The harness writes its own budget verdict; timeout is the outer guard.
  timeout -k 90 "$(( seconds + 30 ))" "${PY[@]}" "$ROOT/run_swe_fresh.py" run \
    --campaign "$OUT/$name" --upstream-port "$PORT" --metrics-url "$URL" --engine "$ENGINE" \
    --agents "$AGENTS" --provider "$PROVIDER" --harbor-version "$HARBOR_VERSION" \
    --dataset-path "$DATASET_PATH" --budget-s "$seconds" --min-free-gb "$MIN_FREE_GB" "${tasks[@]}" "${flags[@]}" \
    > "$OUT/$name.console.log" 2>&1 &
  RPID=$!; RNAME=$name
  while kill -0 "$RPID" 2>/dev/null; do
    hazard
    [ "$(left)" -gt 0 ] || fail budget 7
    sleep 5
  done
  wait "$RPID" || rc=$?
  RPID=
  printf '%s rc=%s\n' "$name" "$rc" >> "$OUT/status.txt"
  stop_server
  hazard
  if [ "$rc" = 124 ] || [ "$rc" = 137 ] || [ "$rc" -gt 128 ]; then harbor_backstop "$name"; fi
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then fail budget 7; fi
  [ "$rc" = 5 ] && fail "$name-disk-low" 5
  [ "$rc" = 0 ] || fail "$name-invalid" 2
  if [ "$name" = canary ]; then sha256sum "$PINS" | cut -d' ' -f1 > "$OUT/canary.pins"; fi
}

identity; docker_ready
PINS_SHA=$(sha256sum "$PINS" | cut -d' ' -f1)
if [ "$ONLY" = measure ]; then
  "${PY[@]}" - "$OUT/canary/SUMMARY.json" "$OUT/canary.pins" "$PINS_SHA" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))['valid'] is True
assert open(sys.argv[2]).read().strip() == sys.argv[3]
PY
fi
if [ "$ONLY" != measure ]; then stage canary; fi
if [ "$ONLY" != canary ]; then identity; stage measure; fi
printf 'complete\n' >> "$OUT/status.txt"
