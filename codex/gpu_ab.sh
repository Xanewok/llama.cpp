#!/usr/bin/env bash
set -euo pipefail
cd /home/xanewok/repos/qwen38-eval-worktrees/wt/spec-tf
exec python3 codex/gpu_payload.py "$@"
