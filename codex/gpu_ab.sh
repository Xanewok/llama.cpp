#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
harness=/home/xanewok/repos/qwen38-eval/harness
experiment="$here/host-bundle.experiment.toml"
case ${1:-} in
    --parsers)
        export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
        exec nice -n 19 python3 "$here/host_bundle_parsers.py"
        ;;
    --capture|--preflight|--reanalyze)
        exec bash "$harness/run.sh" depth "$experiment" "$@"
        ;;
    --canary)
        [[ ${GPUQ_ACTIVE:-} = 1 ]] || { echo 'Use gpuq for GPU execution; queue gates and the wall budget are required.' >&2; exit 8; }
        export PYTHONPATH="$(dirname "$harness")" PYTHONDONTWRITEBYTECODE=1
        exec python3 "$here/host_bundle_canary.py" "$@"
        ;;
    --execute)
        [[ ${GPUQ_ACTIVE:-} = 1 ]] || { echo 'Use gpuq for GPU execution; queue gates and the wall budget are required.' >&2; exit 8; }
        exec bash "$harness/run.sh" depth "$experiment" "$@"
        ;;
    *)
        echo 'Usage: gpu_ab.sh --parsers | --capture | --preflight | --canary | --execute | --reanalyze RUN_DIR [--out DIR] [--port N]' >&2
        exit 8
        ;;
esac
