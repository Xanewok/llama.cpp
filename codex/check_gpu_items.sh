#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
python3 codex/render_depth_item.py
python3 codex/render_memcheck_item.py
python3 codex/verify_gpu_parsers.py
python3 codex/memcheck/memcheck.py --stage preflight
bash codex/gpu_ab.sh --preflight --out "$PWD/codex/gpu-results/depth/preflight-plan"
/home/xanewok/.local/bin/gpuq submit --check "$PWD/codex/memcheck/item.toml"
/home/xanewok/.local/bin/gpuq submit --check "$PWD/codex/depth-item.toml"
