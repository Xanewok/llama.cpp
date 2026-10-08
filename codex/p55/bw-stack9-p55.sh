#!/usr/bin/env bash
# Parent queue only: CUDA build, stamps, test-backend-ops on both GPUs, freeze, then item render/check. No server runs.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
GPUS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
BUILDGATE=$W/bin/buildgate
STACK9=cd776e104dbd516c7b74506992d2e8efd7600155
preflight() {
    [[ $(git branch --show-current) == topic/stack9-p55 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    [[ ! -e $W/frozen/stack9-p55 ]] || return 1
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf flock taskset; do
        command -v "$tool" >/dev/null || return 1
    done
    git merge-base --is-ancestor "$STACK9" HEAD || return 1
    for p in "$CCCL" "$BUILDGATE" "$W/bin/cpugate" "$W/bin/wt" "$W/frozen/stack6/RECIPE.txt" "$W/work/aa-load/ccx01.sh" /usr/local/cuda-13.2/bin/nvcc \
             "$EVAL/harness/mkitem.py" "$W/frozen/stack9/SOURCE" "$W/frozen/stack7/SOURCE" "$W/frozen/stack7-p55/SOURCE" \
             "$W/frozen/stack7-p55/build/bin/round-trace" "$W/frozen/stack9/build/bin/round-trace" "$W/frozen/stack7/build/bin/round-trace" \
             codex/stack9-p55-evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    [[ $(head -n 1 "$W/frozen/stack9/SOURCE") == "$STACK9" ]] || return 1
    # Everything outside codex/ that differs from stack9 must be package 55; the Mac evidence for stages 23 24 32-35 carries over only then.
    [[ $(git diff --name-only "$STACK9" HEAD -- . ':!codex' | LC_ALL=C sort | tr '\n' ' ') == \
       'ggml/src/ggml-cuda/mmvq-params.h ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmvq.cuh tests/CMakeLists.txt tests/test-backend-ops.cpp tests/test-mmvq-params.cpp ' ]] || return 1
    for script in codex/bw-stack9-p55.sh codex/bw-stack9-mac.sh codex/bw-stack9-p55-mac.sh codex/gpu_ab.sh; do bash -n "$script" || return 1; done
    python3 codex/p55_checks.py --fixtures > /dev/null || return 1
    python3 - <<'PY' || return 1
import hashlib, json, tomllib
from pathlib import Path
for name in ('codex/render_s9p55_items.py', 'codex/p55_checks.py', 'codex/gpu_stack9.py'):
    compile(Path(name).read_text(), name, 'exec')
for p in Path('codex/items').glob('s9p55-*.toml.in'):
    tomllib.loads(p.read_text())
r = json.loads(Path('codex/stack9-p55-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
d = hashlib.sha256()
for name in r['source_files']:
    d.update(name.encode() + b'\0' + Path(name).read_bytes())
assert d.hexdigest() == r['source_sha256'], 'source differs from Mac gates'
assert 'ggml/src/ggml-cuda/mmvq-params.h' in r['source_files'] and 'tests/test-mmvq-params.cpp' in r['source_files']
assert {20, 21, 29, 31, 36} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
PY
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-stack9-p55: PASS preflight'; exit 0; fi
    echo 'bw-stack9-p55: FAIL preflight'; exit 10
fi
[[ ${GPUQ_BUILD_WINDOW:-0} == 1 && ${GPUQ_ACTIVE:-0} == 1 && $# == 1 ]] || exit 8
OUT=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1")
case $OUT in "$ROOT"/codex/window/stack9-p55-a[0-9]*) ;; *) exit 10 ;; esac
mkdir -p "$OUT"
preflight || { echo 'bw-stack9-p55: FAIL preflight'; exit 10; }
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-stack9-p55: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
# Same flags as bw-stack9.sh, plus LLAMA_BUILD_TESTS=ON for test-backend-ops (the stack7-p55 window froze such a build).
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$BUILDGATE" cmake -S . -B build-cuda "${flags[@]}"
    nice -n 19 "$BUILDGATE" cmake --build build-cuda --target llama-server test-backend-ops -j4
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
# test-backend-ops under the GPU wrapper. Contract: rc 0 or 1 only (1 = a test case failed; p55_checks.py judges which); a crash or timeout is rc >= 2.
bops() {
    local op=$1 backend=$2 out=$3 err=$4 limit=$5 rc=0
    shift 5
    nice -n 19 "$W/bin/cpugate" env CUDA_VISIBLE_DEVICES="$GPUS" taskset -c 0-3,12-15 timeout --kill-after=30s "$limit" \
        build-cuda/bin/test-backend-ops -o "$op" -b "$backend" "$@" --output sql > "$out" 2> "$err" || rc=$?
    echo "test-backend-ops $op $backend rc_M=$rc"
    [[ $rc == 0 || $rc == 1 ]]
}
# The only stage that sees the GPUs. Contract: every MUL_MAT / MUL_MAT_ID case of the changed quant types (any n) and of every type at n=5..8
# passes on both GPUs, no Xid. A failing case elsewhere is excused only if it passes RETRIES fresh reruns (inputs are random per run).
correctness() {
    local op backend base params re attempt
    [[ ${GPUQ_ACTIVE:-0} == 1 ]]
    python3 codex/p55_checks.py --fixtures
    for op in MUL_MAT MUL_MAT_ID; do
        for backend in CUDA0 CUDA1; do
            base=$OUT/$op-$backend
            bops "$op" "$backend" "$base.sql" "$base.err" 600s
            nice -n 19 python3 codex/p55_checks.py "$base.sql" "$op" "$backend" "$SHA"
            while IFS= read -r -u 3 params; do
                [[ -n $params ]] || continue
                re=$(python3 codex/p55_checks.py --regex "$params")
                for attempt in 1 2 3; do
                    bops "$op" "$backend" "$base.rerun$attempt.sql" "$base.rerun$attempt.err" 300s -p "$re"
                    python3 codex/p55_checks.py --rerun-passed "$base.rerun$attempt.sql" "$op" "$backend" "$SHA" "$params" \
                        || { echo "rerun $attempt failed: $params"; return 1; }
                done
                echo "excused after 3 clean reruns: $params"
            done 3< "$base.sql.retry"
            PYTHONPATH="$EVAL" nice -n 19 python3 - "$base.journal" <<'PY'
import os, sys
from pathlib import Path
from harness import lib
n = lib.record_journal(float(os.environ['P55_CORRECTNESS_SINCE']), Path(sys.argv[1]))
if n is None or n:
    raise SystemExit('kernel fault gate failed; see saved journal')
PY
        done
    done
}
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    nice -n 19 "$W/bin/wt" freeze "$ROOT" stack9-p55 --build build-cuda --desc 'stack9 + package 55 per-type MMVQ; CUDA MUL_MAT and MUL_MAT_ID gates passed on both GPUs; Mac CPU gates passed; CUDA 13.2 120a CCCL343 NCCL OFF'
    [[ $(head -n 1 "$W/frozen/stack9-p55/SOURCE") == "$SHA" ]]
    {
        printf 'commit %s\nconfigure: cmake -S . -B build-cuda ' "$SHA"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server test-backend-ops -j4\nbuild stage: CUDA_VISIBLE_DEVICES=""; test-backend-ops ran in stage 42 only\n'
    } > "$W/frozen/stack9-p55/RECIPE.txt"
}
items() {
    nice -n 19 python3 codex/render_s9p55_items.py --out "$ROOT/codex/items/rendered"
}
stage 40 cuda-build build
stage 41 cuda-stamps-and-clean-tree stamps
P55_CORRECTNESS_SINCE=$(date +%s)
export P55_CORRECTNESS_SINCE
stage 42 cuda-correctness correctness
stage 43 freeze-stack9-p55 freeze
stage 44 render-and-check-items items
python3 - "$OUT" "$SHA" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps({'commit':sys.argv[2], 'result':'PASS', 'GPU_execution':'test-backend-ops MUL_MAT and MUL_MAT_ID only', 'items':'codex/items/rendered; checked, NOT submitted'}, indent=2)+'\n')
PY
echo 'bw-stack9-p55: PASS; frozen and items checked, NOT submitted'
