#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
while true; do
    status=$(gpuq status --brief)
    printf '%s\n' "$status"
    if [[ "$status" != *"RUN "* ]]; then
        break
    fi
    sleep 30
done
script=${2:-}
if [[ "${1:-}" == python3 ]]; then
    case "${script##*/}" in
        host-cpu-identity.py|host-negative.py|host-input-count.py|host-draft-identity.py)
            exec /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate "$@" ;;
    esac
fi
command=${1:-}
if [[ "${command##*/}" == gpuq ]]; then
    exec /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate "$@"
fi
exec /home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate python3 "$(dirname "$0")/host-pause.py" "$@"
