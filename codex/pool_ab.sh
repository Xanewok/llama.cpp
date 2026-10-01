#!/usr/bin/env bash
# DO NOT MERGE. Parent-only; default is a read-only plan. Execution needs RUN_GPU=1.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 python3 "$(dirname "$0")/pool_harness.py" "$@"
