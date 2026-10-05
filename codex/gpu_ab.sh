#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 GIT_NO_LAZY_FETCH=1
export OUT=${OUT:-"$PWD/codex/runs/q8-probe7/manual-$(date +%s)"}
exec nice -n 19 bash codex/q8-probe7/run.sh "${1:---preflight}"
