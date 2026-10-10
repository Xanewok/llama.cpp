#!/usr/bin/env bash
# Parent queue only: build without visible GPUs, freeze, then render/check measurement items.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GGML_CCACHE=ON
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=vanilla-08246a28f
BASE=08246a28f6000100433d297c4e037c02e9d2d464
cd "$ROOT"
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
preflight() {
    [[ $(git branch --show-current) == topic/vanilla-master ]] || return 6
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 6
    [[ -z $(git diff --name-only "$BASE" HEAD -- . ':!codex') ]] || return 6
    [[ ! -e $W/frozen/$NAME ]] || return 6
    if git show-ref --verify --quiet "refs/tags/frozen/$NAME"; then return 6; fi
    [[ -d $CCCL && -x /usr/local/cuda-13.2/bin/nvcc ]] || return 6
    for tool in cmake git python3 nice gpuq ccache strings patchelf taskset; do command -v "$tool" > /dev/null || return 6; done
    for path in "$W/bin/buildgate" "$W/bin/wt" "$W/work/aa-load/ccx01.sh" "$W/frozen/stack9-p55/SOURCE" "$W/frozen/stack10/SOURCE"; do
        [[ -f $path ]] || return 6
    done
    bash -n codex/bw-vanilla.sh codex/gpu_ab.sh || return 6
    python3 - <<'PY' || return 6
from pathlib import Path
for p in Path('codex').glob('*.py'):
    compile(p.read_text(), str(p), 'exec')
assert len(list(Path('codex/items').glob('*.toml.in'))) == 6
PY
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-vanilla: FAIL preflight'; exit 6; }
    echo 'bw-vanilla: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-vanilla && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/vanilla-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
stage() {
    local code=$1 name=$2 rc=0 began=$SECONDS
    shift 2
    (set -e; "$@") > "$OUT/$name.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-began))" >> "$OUT/stages.tsv"
    echo "bw-vanilla: $name rc=$rc wall_s_M=$((SECONDS-began))"
    [[ $rc == 0 ]] || { echo "bw-vanilla: FAIL $name"; case $rc in 3|4|5|6|7|8) exit "$rc" ;; *) exit 8 ;; esac; }
}
build() {
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake -S . -B build-cuda "${flags[@]}" || return 8
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake --build build-cuda --target llama-server -j4 || return 8
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 8
}
stamps() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null || return 6
    strings build-cuda/bin/libllama-common.so | grep -F -- '--spec-draft-sampling' > /dev/null || return 6
    if strings build-cuda/bin/libllama-common.so | grep -F -- '--spec-draft-vocab' > /dev/null; then return 6; fi
    grep -ax 'GGML_CUDA:BOOL=ON' build-cuda/CMakeCache.txt || return 6
    grep -ax 'GGML_CUDA_NCCL:BOOL=OFF' build-cuda/CMakeCache.txt || return 6
    grep -aEx 'CMAKE_CUDA_ARCHITECTURES:(STRING|UNINITIALIZED)=120a' build-cuda/CMakeCache.txt || return 6
    cp build-cuda/CMakeCache.txt "$OUT/CMakeCache.txt" || return 6
    printf 'source_M=%s\nbase_D=%s\nCUDA_D=13.2\nCCCL_D=3.4.3\narch_D=120a\nNCCL_D=OFF\n' "$SHA" "$BASE" > "$OUT/STAMPS.txt"
}
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    nice -n 19 "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda --desc 'unpatched upstream 08246a28f engine; CUDA 13.2 120a CCCL343 NCCL OFF; no GPU execution in build window' || return 6
    [[ $(sed -n '1p' "$W/frozen/$NAME/SOURCE") == "$SHA" ]] || return 6
    {
        printf 'commit %s\nupstream %s\nconfigure: cmake -S . -B build-cuda ' "$SHA" "$BASE"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\nGPU execution: none\ntag: refs/tags/frozen/%s\n' "$NAME"
    } > "$W/frozen/$NAME/RECIPE.txt" || return 6
    cp "$OUT/STAMPS.txt" "$W/frozen/$NAME/STAMPS.txt" || return 6
    git tag "frozen/$NAME" "$SHA" || return 6
}
items() {
    nice -n 19 taskset -c 6-11,18-23 python3 codex/render_vanilla_items.py --out "$ROOT/codex/rendered"
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 48 freeze freeze
stage 49 render-and-check items
python3 - "$OUT" "$SHA" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1], 'summary.json').write_text(json.dumps({'source': sys.argv[2], 'frozen': 'vanilla-08246a28f', 'gpu_executed': False, 'items_submitted': False}, indent=2)+'\n')
PY
