#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
exec nice -n 19 "${S7_BUILDGATE:-$HOME/repos/qwen38-eval-worktrees/bin/buildgate}" "$@"
