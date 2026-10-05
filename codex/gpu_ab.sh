#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 python3 "$(dirname "$0")/mtp_gpu.py" "$@"
