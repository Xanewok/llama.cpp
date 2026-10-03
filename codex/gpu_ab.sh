#!/usr/bin/env bash
# Parent-only input staging A/B: IQ4_XS S5 by default, up/patch/patch/up, trace off.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 python3 "$(dirname "$0")/gpu_harness.py" ab "$@"
