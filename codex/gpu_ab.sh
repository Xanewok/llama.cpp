#!/usr/bin/env bash
# Parent-only gpuq entry. The standard depth harness owns frozen-byte, identity, budget, Xid and exact-PID cleanup gates.
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
H=/home/xanewok/repos/qwen38-eval/harness
export PYTHONPATH="$H/.."
experiment=${1:?experiment}; shift
mode= out=
args=("$@")
while (($#)); do
    case $1 in
        --preflight|--canary|--execute) mode=$1; shift ;;
        --out) out=${2:?out}; shift 2 ;;
        *) shift ;;
    esac
done
[[ -n $mode && -n $out ]] || exit 8
if [[ $mode == --preflight ]]; then
    # No model or GPU process; real recorded artifact checks.
    nice -n 19 python3 "$ROOT/codex/trace_gap.py" --fixtures
    exec bash "$H/run.sh" depth "$experiment" "${args[@]}"
fi
[[ ${GPUQ_ACTIVE:-} == 1 ]] || { echo 'execution requires GPUQ_ACTIVE=1'; exit 8; }
bash "$H/run.sh" depth "$experiment" "${args[@]}"
extra=()
[[ $mode != --canary ]] || extra=(--canary)
python3 "$ROOT/codex/trace_gap.py" --experiment "$experiment" --out "$out" "${extra[@]}"
