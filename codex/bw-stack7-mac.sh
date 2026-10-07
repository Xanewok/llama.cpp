#!/usr/bin/env bash
# Runs bw-stack7 stages 10 and 20-31 (CPU only) on the Mac mini; subset: S7_ONLY="10 20 24". Usage: bw-stack7-mac.sh [OUT | --preflight]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
S=$HOME/qwen38-stack7
export PATH="$S/tools/local/bin:$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
export S7_W=$S/w S7_EVAL=$S/eval S7_MODEL=$S/models/Qwen3.5-4B-UD-Q4_K_XL.gguf S7_VOCAB=$S/models/swift_draft_vocab_ids.json
export S7_GATE=$ROOT/codex/hostgate.sh S7_BUILDGATE=$ROOT/codex/hostgate.sh S7_CCACHE=OFF S7_CPU_ONLY=1
export S7_JOBS=${S7_JOBS:-8} S7_BRANCH=$(git -C "$ROOT" branch --show-current)
export GPUQ_BUILD_WINDOW=1 CMAKE_GENERATOR=Ninja
exec bash "$ROOT/codex/bw-stack7.sh" "${1:-$S7_W/work/build-window-1007/stack7-a1}"
