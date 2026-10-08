#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 S10_ONLY=31
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/codex/bw-stack10-cand-mac.sh" "$@"
