#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec nice -n 19 python3 "$HERE/replay/sweep.py" "$@"
