#!/usr/bin/env bash
set -euo pipefail
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
mode=${1:?mode required}
case $mode in
    --preflight) export CUDA_VISIBLE_DEVICES="" ;;
    --canary|--run) [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] ;;
    *) exit 8 ;;
esac
exec python3 "$(dirname "$0")/probe.py" "${mode#--}" --out "${OUT:?OUT required}" --budget-s "${BUDGET_S:-600}"
