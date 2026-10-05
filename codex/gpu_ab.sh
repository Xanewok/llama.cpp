#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
mode=${1:---preflight}
case "$mode" in
    --preflight|--capture|--canary|--execute) ;;
    *) echo "usage: $0 --preflight|--capture|--canary|--execute [harness options]" >&2; exit 2 ;;
esac
shift || true
if [[ "$mode" = --canary || "$mode" = --execute ]]; then
    [[ ${GPUQ_ACTIVE:-0} = 1 ]] || { echo 'GPU execution requires gpuq' >&2; exit 8; }
fi
exec bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$here/gdn-state-movers.experiment.toml" "$mode" "$@"
