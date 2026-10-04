#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
runner=/home/xanewok/repos/qwen38-eval/harness/run.sh
experiment="$PWD/codex/mask-np2.experiment.toml"
stage=${1:---preflight}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
    --fixtures)
        export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/xanewok/repos/qwen38-eval
        exec nice -n 19 python3 -P codex/mask_np2_harness_check.py
        ;;
    --preflight|--capture)
        export CUDA_VISIBLE_DEVICES=""
        exec nice -n 19 bash "$runner" depth "$experiment" "$stage" "$@"
        ;;
    --canary) cap=600 ;;
    --execute) cap=7200 ;;
    *) echo "usage: gpu_ab.sh --fixtures|--preflight|--capture|--canary|--execute [--out DIR]" >&2; exit 8 ;;
esac
[[ ${GPUQ_ACTIVE:-0} == 1 ]] || { echo "GPU stages require gpuq" >&2; exit 8; }
out="$PWD/codex/results/mask-np2-${stage#--}-$(date +%Y%m%d-%H%M%S)"
args=("$@")
for ((i=0; i<${#args[@]}; ++i)); do
    if [[ ${args[i]} == --out ]]; then out=${args[i+1]}; fi
done
mkdir -p "$out"
guard_dir="${out}.guard"
mkdir -p "$guard_dir"
check_kernel() {
    journalctl -k -b --no-pager > "$guard_dir/kernel-${1}.txt" || { echo "kernel journal unreadable" >&2; return 3; }
    python3 - "$guard_dir/kernel-${1}.txt" <<'PY'
from pathlib import Path
import re
import sys
if re.search(r"\bXid\b|fallen off the bus", Path(sys.argv[1]).read_text()):
    print("GPU fault in current boot; stop the queue", file=sys.stderr)
    sys.exit(3)
PY
}
check_kernel before
if timeout --signal=TERM --kill-after=30 "$cap" bash "$runner" depth "$experiment" "$stage" --out "$out" "$@"; then
    rc=0
else
    rc=$?
fi
check_kernel after
if [[ $rc == 124 || $rc == 137 ]]; then
    echo "budget: ${cap}s wall cap exceeded" >&2
    exit 7
fi
exit "$rc"
