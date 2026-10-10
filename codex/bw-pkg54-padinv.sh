#!/usr/bin/env bash
# GPU execution belongs to gpuq. Build with hidden devices, then seal one binary for both routes.
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=/home/xanewok/repos/qwen38-eval-worktrees
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
BASE=a588f2a44382a361df3e2a456be0c80ca10dd1d6
preflight() {
    [[ $(git branch --show-current) == topic/pkg54-padinv ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    git diff --quiet HEAD -- . ':!codex/PROMPT.md' ':!codex/PURPOSE' || return 1
    local p
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc /usr/bin/patchelf codex/padinv_summary.py; do
        [[ -e $p ]] || return 1
    done
    python3 codex/padinv_summary.py selftest codex/padinv-evidence --read-only || return 1
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-pkg54-padinv: FAIL preflight'; exit 8; }
    echo 'bw-pkg54-padinv: PASS preflight'
    exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-pkg54-padinv && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
[[ $VISIBLE == GPU-* ]] || exit 6
OUT=$1
case "$OUT" in /mnt/scratch/pkg54-padinv/a[0-9]*) ;; *) exit 8 ;; esac
[[ ! -e $OUT ]] || { echo 'bw-pkg54-padinv: FAIL output already exists'; exit 6; }
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
export TMPDIR=/mnt/scratch/tmp-build
mkdir -p "$TMPDIR"
BUILD=$OUT/build
SEALED=$OUT/sealed
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
# D: total cap 7200 s; build 3000, canary 600, full 3300, cleanup/analysis reserve 300.
nice -n 19 timeout --kill-after=20 3000 bash -c '
    set -euo pipefail
    gate=$1; root=$2; build=$3; shift 3
    "$gate" cmake -S "$root" -B "$build" "$@"
    "$gate" cmake --build "$build" --target test-backend-ops -j4
' bash "$W/bin/buildgate" "$ROOT" "$BUILD" "${flags[@]}" || { rc=$?; [[ $rc == 124 ]] && exit 7; exit 8; }
[[ $(git rev-parse HEAD) == "$SHA" ]] || exit 6
git diff --quiet HEAD -- . ':!codex/PROMPT.md' ':!codex/PURPOSE' || exit 6
mkdir -p "$SEALED"
cp -L "$BUILD/bin/test-backend-ops" "$BUILD"/bin/*.so* "$SEALED/"
for f in "$SEALED/test-backend-ops" "$SEALED"/*.so*; do
    patchelf --set-rpath '$ORIGIN' "$f"
done
(cd "$SEALED" && sha256sum test-backend-ops *.so* > SHA256SUMS)
printf '%s\n' "$SHA" > "$OUT/SOURCE"
printf '%s\n' "${flags[@]}" > "$OUT/RECIPE.txt"
sealed_check() { (cd "$SEALED" && sha256sum --check SHA256SUMS); }
sealed_check || exit 6
rc=0
CUDA_VISIBLE_DEVICES="$VISIBLE" nice -n 19 python3 codex/padinv_summary.py execute "$OUT/canary" \
    --binary "$SEALED/test-backend-ops" --budget-s 600 --canary || rc=$?
if [[ $rc != 0 ]]; then
    [[ ! -f $OUT/canary/summary.json ]] || cp "$OUT/canary/summary.json" "$OUT/summary.json"
    exit "$rc"
fi
sealed_check || exit 6
rc=0
CUDA_VISIBLE_DEVICES="$VISIBLE" nice -n 19 python3 codex/padinv_summary.py execute "$OUT/run" \
    --binary "$SEALED/test-backend-ops" --budget-s 3300 || rc=$?
[[ ! -f $OUT/run/summary.json ]] || cp "$OUT/run/summary.json" "$OUT/summary.json"
sealed_check || exit 6
[[ $rc == 0 ]] || exit "$rc"
echo 'bw-pkg54-padinv: PASS'
