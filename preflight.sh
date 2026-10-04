#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=""
nice -n 19 bash -n "$ROOT/gpu_swe_fresh.sh"
cd "$ROOT"
nice -n 19 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 python3 -m unittest -q test_swe_fresh
