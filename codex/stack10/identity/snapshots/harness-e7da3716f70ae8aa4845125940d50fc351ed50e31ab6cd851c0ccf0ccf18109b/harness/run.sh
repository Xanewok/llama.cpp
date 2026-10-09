#!/usr/bin/env bash
# gpuq entry: bash run.sh <kind> experiment.toml --execute|--canary|--preflight [...]
set -euo pipefail
unset PYTHONOPTIMIZE LD_PRELOAD LD_LIBRARY_PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 GIT_NO_LAZY_FETCH=1
here=$(cd "$(dirname "$0")" && pwd)
code=$(cd "$here/.." && pwd)
data=$code
if [[ -f $code/source-root.txt ]]; then data=$(cat "$code/source-root.txt"); fi
export HARNESS_DATA_ROOT="$data" PYTHONPATH="$code:$data"
# Host time is a measured axis: only CPU-only stages run niced; the server inherits the launcher's nice.
case " $* " in *" --preflight "*|*" --capture "*) exec nice -n 19 python3 -P -m harness "$@" ;; esac
exec python3 -P -m harness "$@"
