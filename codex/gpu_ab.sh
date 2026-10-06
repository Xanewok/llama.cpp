#!/usr/bin/env bash
set -euo pipefail
if [[ ${GPUQ_ACTIVE:-} != 1 ]]; then
    echo 'GPUQ_ACTIVE=1 required; this item is for the parent to queue.' >&2
    exit 6
fi
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:---canary}
shift || true
exec bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$here/depth-experiment.toml" "$mode" "$@"
