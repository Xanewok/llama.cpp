#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 S7_ONLY='20 21'
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/codex/bw-stack8-mac.sh" "$@"
