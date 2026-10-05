#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
experiment="$root/codex/depth-experiment.toml"
item="$root/codex/depth-item.toml"
eval_repo=/home/xanewok/repos/qwen38-eval
case "${1:---check}" in
    --check)
        export CUDA_VISIBLE_DEVICES=""
        cd "$eval_repo"
        nice -n 19 python3 -m harness.mkitem "$experiment" --check "$item"
        nice -n 19 /home/xanewok/repos/gpuq/bin/gpuq submit --check "$item"
        ;;
    --preflight)
        export CUDA_VISIBLE_DEVICES=""
        exec nice -n 19 bash "$eval_repo/harness/run.sh" depth "$experiment" --preflight --out "$root/codex/gpu-results/depth/preflight-plan"
        ;;
    --canary|--execute)
        [[ ${GPUQ_ACTIVE:-0} == 1 ]] || { echo 'Run GPU stages only through gpuq.' >&2; exit 6; }
        mode=$1
        shift
        exec bash "$eval_repo/harness/run.sh" depth "$experiment" "$mode" "$@"
        ;;
    *) echo 'usage: gpu_ab.sh --check|--preflight|--canary|--execute' >&2; exit 2 ;;
esac
