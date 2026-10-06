#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
here=$(cd "$(dirname "$0")" && pwd)
if [[ ${GPUQ_ACTIVE:-} != 1 ]]; then
    echo 'mmq-smallj: GPU queue environment required' >&2
    exit 6
fi
python3 "$here/kernel_item.py" perf --stage preflight
python3 - "$here" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from kernel_item import require_memcheck
require_memcheck()
PY
if [[ $# -lt 2 ]]; then
    echo 'usage: gpu_ab.sh {iq4-np2|j5q-np2|j5q-np1|j5q-prefill} {--canary|--execute} [harness args]' >&2
    exit 6
fi
experiment="$here/items/$1.toml"
shift
exec bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$experiment" "$@"
