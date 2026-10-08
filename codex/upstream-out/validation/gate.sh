#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
exec python3 "$(dirname "$0")/model_gate.py" "$@"
