#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
bin=/home/xanewok/repos/qwen38-eval-worktrees/bin
mkdir -p codex/logs
for stage in cpu cuda negative resources; do
    printf 'Starting gated stage: %s\n' "$stage"
    bash codex/build.sh "$stage" > "codex/logs/final-$stage.log" 2>&1
done
for spec in 'mmq-smallj build-cuda' 'mmq-smallj-negative build-cuda-negative'; do
    read -r name build <<< "$spec"
    QWEN38_WT_ROOT="$PWD/codex" "$bin/wt" freeze "$PWD" "$name" --build "$build" \
        --desc 'Small-J MMQ CPU-prepared validation; CUDA 13.2; NCCL OFF; no device execution by author'
    cp "$build/CMakeCache.txt" "codex/frozen/$name/CMakeCache.txt"
    stamp=$(sed -n 's/^#define GGML_COMMIT *"\([^"]*\)".*/\1/p' "$build/ggml/src/ggml-version.h")
    sha=$(git rev-parse HEAD)
    [[ -n $stamp && ${sha:0:${#stamp}} == "$stamp" ]]
    strings "codex/frozen/$name/build/bin/libggml-base.so" | rg -Fx "$stamp" > "codex/frozen/$name/ggml-stamp.txt"
    {
        printf 'source: '; git rev-parse HEAD
        printf 'CUDA_VISIBLE_DEVICES=""; see codex/build.sh %s\n' "$name"
        rg '^CMAKE_CUDA_FLAGS:|^CMAKE_CUDA_ARCHITECTURES:|^GGML_BACKEND_DL:|^GGML_CCACHE:|^GGML_CUDA_NCCL:' "$build/CMakeCache.txt"
    } > "codex/frozen/$name/RECIPE.txt"
done
# The resource reader uses the dedicated TU logs, not the interleaved full-build log.
python3 codex/gen_items.py
for item in codex/items/*.gpuq.toml; do
    gpuq submit --check "$item"
done
printf 'Gated builds, host tests, freezes and item contract checks PASS\n'
