#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export CPUGATE_MIN_GIB=10 BUILDGATE_MIN_GIB=10
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
W=/home/xanewok/repos/qwen38-eval-worktrees
EVAL=/home/xanewok/repos/qwen38-eval
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
GPUS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
cd "$ROOT"
preflight() {
    local path tool
    for path in "$CCCL" "$W/bin/wt" "$W/bin/buildgate" "$W/bin/cpugate" "$W/frozen/stack6/RECIPE.txt" "$W/frozen/stack7/SOURCE" "$EVAL/harness/mkitem.py" /usr/local/cuda-13.2/bin/nvcc; do
        [[ -e $path ]] || return 1
    done
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf taskset; do
        command -v "$tool" >/dev/null || return 1
    done
    [[ $(git branch --show-current) == topic/stack7-p55 ]] || return 1
    [[ ! -e $W/frozen/stack7-p55 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0302404cb HEAD || return 1
    for path in codex/{bw-stack7-p55,gpu-p55,render-p55-items}.sh; do bash -n "$path" || return 1; done
}
if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-stack7-p55: PASS preflight'; exit 0; fi
    echo 'bw-stack7-p55: FAIL preflight'; exit 10
fi
[[ ${GPUQ_BUILD_WINDOW:-0} == 1 && ${GPUQ_ACTIVE:-0} == 1 && $# == 1 ]] || exit 8
OUT=$(readlink -m "$1")
[[ $OUT =~ ^$W/work/build-window-1007/stack7-p55-a[0-9]+$ ]] || exit 10
mkdir -p "$OUT"
run_stage() {
    local name=$1 code=$2 rc
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' "$name" "$code" "$rc" >> "$OUT/stages.tsv"
    if [[ $rc != 0 ]]; then echo "bw-stack7-p55: FAIL $name rc=$code command_rc=$rc"; exit "$code"; fi
    echo "bw-stack7-p55: PASS $name"
}
run_stage preflight 10 preflight
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
printf '%s\n' "$SHA" > "$OUT/source-commit.txt"
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_build() {
    nice -n 19 "$W/bin/buildgate" cmake -S codex/p55-test -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target llama-server test-backend-ops -j4
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
run_stage cuda-build 40 cuda_build
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
run_stage cuda-stamps-and-clean-tree 41 stamps
correctness() {
    local op backend
    # This stage is parent-queued GPU work. The build and freeze keep GPUs hidden.
    for op in MUL_MAT MUL_MAT_ID; do
        for backend in CUDA0 CUDA1; do
            nice -n 19 "$W/bin/cpugate" env CUDA_VISIBLE_DEVICES="$GPUS" taskset -c 0-3,12-15 timeout --kill-after=30s 600s \
                build-cuda/bin/test-backend-ops -o "$op" -b "$backend" --output sql \
                > "$OUT/$op-$backend.sql" 2> "$OUT/$op-$backend.err"
            nice -n 19 python3 codex/p55_checks.py "$OUT/$op-$backend.sql" "$op" "$backend" "$SHA"
            PYTHONPATH="$EVAL" nice -n 19 python3 - "$OUT/$op-$backend.journal" <<'PY'
import os,sys
from pathlib import Path
from harness import lib
n = lib.record_journal(float(os.environ['P55_CORRECTNESS_SINCE']), Path(sys.argv[1]))
if n is None or n:
    raise SystemExit('kernel fault gate failed; see saved journal')
PY
        done
    done
}
P55_CORRECTNESS_SINCE=$(date +%s)
export P55_CORRECTNESS_SINCE
run_stage cuda-correctness 42 correctness
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    "$W/bin/wt" freeze "$ROOT" stack7-p55 --build build-cuda --desc 'stack7 + package 55 per-type MMVQ; CUDA MUL_MAT and MUL_MAT_ID gates passed on both GPUs; CUDA 13.2 120a CCCL343 NCCL OFF'
    [[ $(head -n 1 "$W/frozen/stack7-p55/SOURCE") == "$SHA" ]]
    printf 'commit %s\nconfigure: cmake -S codex/p55-test -B build-cuda ' "$SHA" > "$OUT/RECIPE.txt"
    printf '%q ' "${flags[@]}" >> "$OUT/RECIPE.txt"
    printf '\nbuild: nice -n 19 bin/buildgate cmake --build build-cuda --target llama-server test-backend-ops -j4\n' >> "$OUT/RECIPE.txt"
    cp "$OUT/RECIPE.txt" "$W/frozen/stack7-p55/RECIPE.txt"
}
run_stage freeze-stack7-p55 43 freeze
run_stage generate-and-check-gpu-items 44 bash codex/render-p55-items.sh "$OUT"
nice -n 19 python3 - "$OUT" "$SHA" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
stages=[dict(name=n,exit_code=int(c),command_rc=int(r)) for n,c,r in (l.split('\t') for l in (out/'stages.tsv').read_text().splitlines())]
assert all(s['command_rc']==0 for s in stages)
(out/'summary.json').write_text(json.dumps(dict(commit=sys.argv[2],stages=stages,items='contract checks PASS; NOT submitted'),indent=2)+'\n')
PY
echo 'bw-stack7-p55: PASS; frozen; items checked, NOT submitted'
