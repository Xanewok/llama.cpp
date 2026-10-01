#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
exec nice -n 19 bash "$(dirname "$0")/gpu_scope.sh" "$@"
