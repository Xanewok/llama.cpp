#!/usr/bin/env bash
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:---canary}
shift || true
if [[ $mode != --preflight && ${GPUQ_ACTIVE:-} != 1 ]]; then
    echo 'GPUQ_ACTIVE=1 required; this item is for the parent to queue.' >&2
    exit 6
fi
bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$here/depth-experiment.toml" "$mode" "$@"
if [[ $mode == --canary ]]; then
    exec nice -n 19 python3 "$here/canary_identity.py" "$@"
fi
