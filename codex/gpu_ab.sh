#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
# Run only in gpuq; the parent selects an already validated item.
if [[ ${GPUQ_ACTIVE:-} != 1 ]]; then
    echo 'probe: integrity: queue environment missing' >&2
    exit 6
fi
quant=${1:?use iq4 or j5q}
stage=${2:---canary}
case "$quant" in iq4|j5q) ;; *) exit 6 ;; esac
case "$stage" in --canary|--execute) ;; *) exit 6 ;; esac
exec bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$root/items/depth-$quant.toml" "$stage" "${@:3}"
