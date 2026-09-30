#!/usr/bin/env bash
# Parent-only. Plan by default; --execute also requires RUN_GPU=1.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 python3 "$(dirname "$0")/gpu_harness.py" trace "$@"
