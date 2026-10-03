#!/usr/bin/env bash
# Build and freeze only. This script never runs a CUDA executable.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
export GIT_DIR="$PWD/codex/local.git"
mkdir -p "$TMPDIR"
if [[ ! -d codex/base-src ]]; then
    mkdir -p codex/base-src
    git archive "$(cat codex/BASE)" | tar -x -C codex/base-src
fi
# CMake otherwise runs a CUDA device probe, even with explicit architectures.
flags=(-DCMAKE_CUDA_COMPILER_FORCED=ON -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
if [[ ! -d codex/base.git ]]; then
    git init --bare --quiet codex/base.git
    printf '%s\n' "$PWD/codex/local.git/objects" > codex/base.git/objects/info/alternates
    git --git-dir=codex/base.git config core.bare false
    git --git-dir=codex/base.git config core.worktree "$PWD/codex/base-src"
    git --git-dir=codex/base.git update-ref refs/heads/base "$(cat codex/BASE)"
    git --git-dir=codex/base.git symbolic-ref HEAD refs/heads/base
    git --git-dir=codex/base.git read-tree HEAD
fi
export GIT_DIR="$PWD/codex/base.git"
cmake -S codex/base-src -B build-base-cuda "${flags[@]}" > codex/cmake-base-cuda-safe.log 2>&1
cmake --build build-base-cuda --target llama-server -j8 > codex/build-base-cuda-safe.log 2>&1
export GIT_DIR="$PWD/codex/local.git"
cmake -B build-cuda "${flags[@]}" > codex/cmake-cuda-safe.log 2>&1
cmake --build build-cuda --target llama-server -j8 > codex/build-cuda-safe.log 2>&1
freeze() {
    local build=$1 dest=$2 source=$3
    [[ ! -e "$dest" ]] || { echo "freeze destination already exists: $dest" >&2; exit 1; }
    mkdir -p "$dest/bin"
    cp "$build/bin/llama-server" "$dest/bin/round-trace"
    for path in "$build"/bin/*.so*; do
        cp -L "$path" "$dest/bin/$(basename "$path")"
    done
    printf '%s\n' "$source" > "$dest/SOURCE"
    (cd "$dest/bin"; sha256sum round-trace *.so* > SHA256SUMS)
    (cd "$dest"; sha256sum SOURCE bin/SHA256SUMS > SHA256SUMS)
}
freeze build-base-cuda codex/frozen/graph-up "$(cat codex/BASE)"
freeze build-cuda codex/frozen/graph-patch "$(git rev-parse HEAD)"
printf 'M: frozen CUDA builds; no CUDA executable was run\n'
