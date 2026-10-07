#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 S7_ONLY=${S7_ONLY:-'20 21 31'}
export S8_PRIMARY=$HOME/qwen38-bw/stack8/bdae9b6aa3f29c761b70e6a133a7a0a54bbc3c9a/run.935rrX
[[ -f $S8_PRIMARY/exit-code && $(cat "$S8_PRIMARY/exit-code") == 0 ]]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/codex/bw-stack8-mac.sh" "$@"
