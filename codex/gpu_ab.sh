#!/usr/bin/env bash
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/xanewok/repos/qwen38-eval
mode=${1:?use --preflight, --canary, or --execute}
case "$mode" in
    --preflight) export CUDA_VISIBLE_DEVICES="" ;;
    --canary|--execute) [[ ${GPUQ_ACTIVE:-0} = 1 && -n ${CUDA_VISIBLE_DEVICES:-} ]] ;;
    *) exit 6 ;;
esac
exec bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$(dirname "$0")/depth-experiment.toml" "$@"
