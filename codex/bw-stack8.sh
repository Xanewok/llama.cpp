#!/usr/bin/env bash
# Parent queue only: CUDA compilation and freeze, then item render/check. No CUDA binary runs.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
BUILDGATE=$W/bin/buildgate
preflight() {
    [[ $(git branch --show-current) == topic/stack8 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    [[ ! -e $W/frozen/stack8 ]] || return 1
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf flock; do
        command -v "$tool" >/dev/null || return 1
    done
    git merge-base --is-ancestor 48b82b0d1 HEAD || return 1
    for p in "$CCCL" "$BUILDGATE" "$W/bin/wt" "$W/frozen/stack6/RECIPE.txt" "$W/frozen/stack7/SOURCE" "$W/frozen/stack7/build/bin/round-trace" "$W/work/aa-load/ccx01.sh" /usr/local/cuda-13.2/bin/nvcc codex/mac-evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY' || return 1
import json
from pathlib import Path
r = json.loads(Path('codex/mac-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20, 21, 23, 29, 32, 33, 34, 31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
PY
    [[ ${S8_DUO_FOLD_C:-} == 0 || ${S8_DUO_FOLD_C:-} == 1 ]]
}
if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-stack8: PASS preflight'; exit 0; fi
    echo 'bw-stack8: FAIL preflight'; exit 10
fi
[[ ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1")
case $OUT in "$W"/work/build-window-1007/stack8-a[0-9]*) ;; *) exit 10 ;; esac
mkdir -p "$OUT"
preflight
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    [[ -z ${S7_ONLY:-} || " $S7_ONLY " == *" $code "* ]] || return 0
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-stack8: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$BUILDGATE" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$BUILDGATE" cmake --build build-cuda --target llama-server -j4
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
freeze() {
    "$W/bin/wt" freeze "$ROOT" stack8 --build build-cuda --desc 'stack7 + np2 chain fallback and backend candidates under banning bias; Mac gates passed; CUDA 13.2 120a CCCL343 NCCL OFF'
    [[ $(head -1 "$W/frozen/stack8/SOURCE") == "$SHA" ]]
    {
        printf 'commit %s\nconfigure: cmake -B build-cuda ' "$SHA"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\nCUDA_VISIBLE_DEVICES=""; no CUDA binary executed\n'
    } > "$W/frozen/stack8/RECIPE.txt"
}
items() {
    nice -n 19 python3 codex/render_stack8_items.py --duo-fold-c "$S8_DUO_FOLD_C" --out "$ROOT/codex/items/rendered"
}
stage 40 cuda-build build
stage 41 cuda-stamps-and-clean-tree stamps
stage 42 freeze-stack8 freeze
stage 43 render-and-check-items items
python3 - "$OUT" "$SHA" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps({'commit':sys.argv[2], 'result':'PASS', 'GPU_execution':False, 'items':'codex/items/rendered; checked, NOT submitted'}, indent=2)+'\n')
PY
echo 'bw-stack8: PASS; frozen and items checked, NOT submitted'
