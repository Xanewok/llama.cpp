#!/usr/bin/env bash
# Driver-side serving control: everything between "I have a reachable runner" and "a
# verified server is answering". Provider-independent on purpose -- OpenSSH returns 255 for
# any client-side failure anywhere, so this is not a vast quirk and must not be filed under
# one, or a non-vast experiment never finds it. Renting and ssh transport live in box.sh.
#
# Source this; it defines functions only. Callers set LIB_ROOT to the repo root.
#
# Every wait loop in this repo used to ask "is the server still there?" with
#   ssh box 'pgrep -x llama-server' || return 1
# which cannot tell a dead server from a dropped ssh. A transient drop then marked a
# context-ladder rung as refused, and the wrong max_ctx went into a results table.

# proc_state <runner> <process-name>
#   0 = running, 1 = gone, 2 = unknown (ssh itself failed -- do NOT treat as gone).
# <runner> is the caller's own remote-exec function, e.g. rssh.
LIB_ROOT=${LIB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

proc_state () {
  local out rc
  out=$("$1" "pgrep -x $2 >/dev/null && echo UP || echo DOWN" 2>/dev/null); rc=$?
  [ "$rc" -eq 255 ] && return 2
  case "$out" in
    *UP*)   return 0 ;;
    *DOWN*) return 1 ;;
    *)      return 2 ;;
  esac
}

# served_model <local-port>
#   What llama-server says it is actually serving, so a row can be tied to a model rather
#   than to the label we intended. A stale server that outlived its kill answers /health
#   with "ok" from the previous config, and the row is then recorded against the new one.
served_model () {
  curl -s -m 10 "http://127.0.0.1:$1/props" 2>/dev/null \
    | python3 -c "import sys,json
try: print(json.load(sys.stdin).get('model_path',''))
except Exception: print('')" 2>/dev/null
}

# await_server <local-port> <tries> <sleep> <runner> <process-name> [expect-substring]
#   Returns 0 only when the endpoint answers AND, if expect-substring is given, the server
#   reports serving something matching it. Returns 1 when the process is confirmed gone.
#   An ssh failure is retried rather than counted as death.
await_server () {
  local port=$1 tries=$2 nap=$3 runner=$4 proc=$5 expect=${6:-}
  local i
  for ((i = 0; i < tries; i++)); do
    if curl -s -m 10 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok; then
      [ -z "$expect" ] && return 0
      case "$(served_model "$port")" in
        *"$expect"*) return 0 ;;
        "")          return 0 ;;   # server does not expose /props; identity unverifiable
        *)           return 3 ;;   # a different model is answering: stale server
      esac
    fi
    proc_state "$runner" "$proc"
    case $? in
      1) return 1 ;;               # confirmed gone
      2) : ;;                      # ssh flake: keep waiting
    esac
    sleep "$nap"
  done
  return 1
}

# say <words...>
#   Timestamped progress on stdout. Defined once here because 24 orchestration scripts
#   each defined their own, which is the same reason one bug needed fixing in eight files.
# no_argv "$@"
#   Every driver that sources this is configured by env var and rents a box within seconds,
#   so a positional argument is a mistake, not a request -- refuse before spending. --help
#   prints the caller's own header block, which is where the env vars are documented.
no_argv () {
  case "${1:-}" in
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    "") return 0 ;;
    *) echo "$(basename "$0") takes no arguments; configure it with env vars (see --help)." >&2
       exit 2 ;;
  esac
}

say () { echo "$(date -u '+%H:%M:%S') $*"; }

# ship_serving <runner> [remote-dir]
#   Put the versioned serving layer on the box. These scripts used to be authored on the
#   box and existed nowhere else, so each teardown destroyed them.
ship_serving () {
  local runner=$1 dir=${2:-/root/work} f
  "$runner" "mkdir -p $dir" || return 1
  for f in "$LIB_ROOT"/serving/*.sh; do
    "$runner" "cat > $dir/$(basename "$f") && chmod +x $dir/$(basename "$f")" < "$f" || return 1
  done
}

# serve <runner> <local-port> <model-basename> <ctx> [engine flags...]
#   Launch and return only once the endpoint answers AND reports serving this model.
#   Returns 1 if it never came up, 3 if something else is answering on that port.
#
#   The identity check is the point: a server that outlived its kill answers /health from
#   its previous config, and the measurement is then filed against a model that never
#   loaded. Provider-independent, so it lives here rather than in box.sh.
serve () {
  local runner=$1 port=$2 model=$3 ctx=$4; shift 4
  "$runner" "WORK=${REMOTE_WORK:-/root/work} PORT=${REMOTE_PORT:-8000} KV=${KV:-q8_0} \
    PARALLEL=${PARALLEL:-1} \
    ${REMOTE_WORK:-/root/work}/serve.sh $model $ctx $*" >/dev/null \
    || { say "serve.sh refused to launch $model (gpu not free, or no such model)"; return 1; }
  await_server "$port" "${SERVE_TRIES:-40}" "${SERVE_NAP:-10}" "$runner" llama-server "$model"
  local rc=$?
  [ "$rc" -eq 3 ] && say "STALE SERVER on port $port -- not measuring $model"
  return $rc
}

# ensure_hf <runner>
#   Print the path to the HF CLI on the box, installing it if absent. The base CUDA image
#   has no pip, and Ubuntu 24.04 refuses a system-wide pip install without the flag.
ensure_hf () {
  local runner=$1 hf
  hf=$("$runner" 'PATH=$PATH:$HOME/.local/bin; command -v hf || command -v huggingface-cli' | tr -d '\r')
  [ -n "$hf" ] && { echo "$hf"; return 0; }
  "$runner" 'export DEBIAN_FRONTEND=noninteractive
    python3 -m pip --version >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq python3-pip; }
    python3 -m pip install -q --break-system-packages "huggingface_hub[cli]"' >/dev/null || return 1
  hf=$("$runner" 'PATH=$PATH:$HOME/.local/bin; command -v hf || command -v huggingface-cli' | tr -d '\r')
  [ -n "$hf" ] || return 1
  echo "$hf"
}

# fetch_model <runner> <hf-path> <repo> <file> [dest-dir]
#   xet is disabled: its token endpoint threw a ConnectionError mid-download and cost a
#   whole provisioned box.
fetch_model () {
  local runner=$1 hf=$2 repo=$3 file=$4 dest=${5:-/root/work/models/qwen38} attempt
  for attempt in 1 2 3 4 5; do
    "$runner" "HF_HUB_DISABLE_XET=1 $hf download '$repo' '$file' --local-dir '$dest'" >/dev/null \
      && { say "got $file"; return 0; }
    say "download attempt $attempt failed for $file"
    # The remote downloader outlives the ssh that started it, so a plain retry adds a
    # second one that blocks forever on the first one's .lock. Matched out of /proc, never
    # with pkill -f, which would count this very command line as a hit.
    "$runner" "for p in \$(ls /proc | grep -E '^[0-9]+\$'); do
        tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -q 'download.*$(basename "$file")' && kill \$p
      done
      rm -f '$dest/.cache/huggingface/download/$file.lock'" >/dev/null 2>&1
    sleep $((attempt * 20))
  done
  say "DOWNLOAD FAILED $file"; return 1
}
