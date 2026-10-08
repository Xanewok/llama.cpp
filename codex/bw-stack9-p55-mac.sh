#!/usr/bin/env bash
# Package 55 is CUDA-only. This gate proves the merge compiles and runs on CPU, that test-mmvq-params passes and its mutant aborts,
# and re-runs the stack9 CPU stages that need no hour-long model run; stages 23 24 32-35 test sources p55 does not touch.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 S7_ONLY='20 21 29 31 36'
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/codex/bw-stack9-mac.sh" "$@"
