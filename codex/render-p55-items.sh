#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/codex/render"
OUT=${1:?contract log directory}
mkdir -p "$OUT"
for name in p55-depth-iq4 p55-np2-real-iq4 p55-depth-j5q; do
    nice -n 19 python3 -m harness.mkitem "codex/experiments/$name.toml.in" --out "codex/items/$name.gpuq.toml" > "$OUT/$name-render.log" 2>&1
    nice -n 19 python3 -m harness.mkitem "codex/experiments/$name.toml.in" --check "codex/items/$name.gpuq.toml" > "$OUT/$name-render-check.log" 2>&1
    nice -n 19 gpuq submit --check "$ROOT/codex/items/$name.gpuq.toml" > "$OUT/$name-contract.log" 2>&1
done
