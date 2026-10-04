#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
# The server inherits this niceness and host time is the measured axis (attrib_llama.sh does not nice its server):
# only CPU-only invocations are niced.
case " $* " in *" --execute-parent "*) exec python3 codex/gpu_ab.py "$@" ;; esac
exec nice -n 19 python3 codex/gpu_ab.py "$@"
