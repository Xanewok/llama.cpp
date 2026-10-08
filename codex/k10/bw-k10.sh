#!/usr/bin/env bash
# K10 window: CUDA build, batched top-k order/engagement/negative-control gates on both GPUs, perf arms. Never freezes.
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
preflight() {
    [[ $(git branch --show-current) == topic/s10-k10 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0cfc0723e HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc codex/k10/evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY_GATE' || return 1
import hashlib, json, subprocess
from pathlib import Path
r = json.loads(Path('codex/k10/evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {10, 11, 12, 20, 21, 22, 31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
h = hashlib.sha256()
for name in subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines():
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-k10: FAIL preflight'; exit 8; }
    echo 'bw-k10: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-k10 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/k10-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-k10: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || { echo "bw-k10: FAIL $name"; exit 8; }
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target test-backend-ops -j4
    grep -F 'set(CMAKE_CXX_COMPILER_ID "GNU")' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
    grep -E 'set\(CMAKE_CXX_COMPILER_VERSION "13\.' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
}
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
tbo() {
    local dev=$1 out=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o TOP_K -j 1 > "$out" 2>&1
}
# gates on the exact path
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev
    for dev in CUDA0 CUDA1; do
        tbo "$dev" "$OUT/topk-$dev.log"
        grep -aE '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/topk-$dev.log" > /dev/null
        python3 codex/k10/k10_perf.py engaged "$OUT/topk-$dev.log"
        tbo "$dev" "$OUT/topk-log-$dev.log" GGML_CUDA_TOPK_LOG=1
        [[ $(grep -ac 'top_k_batched: engaged' "$OUT/topk-log-$dev.log") -gt 0 ]]
    done
}
# the canonical cases must fail when batching is off or any control is on: they discriminate
controls() {
    local dev arm rc
    for dev in CUDA0 CUDA1; do
        for arm in off m1 m2 m3 m4 m5 m6; do
            case $arm in
                off) envs=(GGML_CUDA_TOPK_BATCHED=0) ;;
                m*) envs=(GGML_CUDA_TOPK_MUTANT="${arm#m}") ;;
            esac
            rc=0
            tbo "$dev" "$OUT/control-$arm-$dev.log" "${envs[@]}" || rc=$?
            [[ $rc != 0 ]] || { echo "FAIL control $arm escaped on $dev"; return 1; }
            python3 codex/k10/k10_perf.py failures "$OUT/control-$arm-$dev.log" "$arm"
        done
    done
}
perf() {
    local dev arm rep
    local shapes='ne=\[(248320|124160|42|1024),(1|2|6|12|13),1,1\],k=(21|41)'
    for dev in CUDA0 CUDA1; do
        for rep in 1 2; do
            for arm in auto off force; do
                case $arm in
                    auto) envs=() ;;
                    off) envs=(GGML_CUDA_TOPK_BATCHED=0) ;;
                    force) envs=(GGML_CUDA_TOPK_BATCHED=force) ;;
                esac
                env CUDA_VISIBLE_DEVICES="$VISIBLE" "${envs[@]}" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops perf -b "$dev" -o TOP_K -p "$shapes" > "$OUT/perf-$arm-$dev-$rep.log" 2>&1
            done
        done
        python3 codex/k10/k10_perf.py perf "$OUT" "$dev" > "$OUT/perf-$dev.json"
    done
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 topk-order-and-engagement ops
stage 43 topk-controls controls
stage 44 topk-perf perf
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True,note='perf tables in perf-CUDA*.json are measurements (M), not gates'),indent=2)+'\n')
PY_RESULT
echo 'bw-k10: PASS'
