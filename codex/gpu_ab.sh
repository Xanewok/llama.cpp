#!/usr/bin/env bash
# Parent-only up/patch/patch/up. Both arms run with tracing off.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 python3 "$(dirname "$0")/gpu_harness.py" ab "$@"
