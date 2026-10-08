#!/usr/bin/env bash
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
source_stamp() { git rev-parse HEAD:ggml HEAD:src HEAD:common HEAD:tests | paste -sd+; }
preflight() {
    [[ $(git branch --show-current) == topic/s10-k11b ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0cfc0723e HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc codex/k11b-evidence/summary.json codex/nsys/k11b_nsys.toml; do
        [[ -e $p ]] || return 1
    done
    python3 - "$(git rev-parse HEAD:ggml)" "$(git rev-parse HEAD:tests)" <<'PY_GATE' || return 1
import json, sys
from pathlib import Path
r = json.loads(Path('codex/k11b-evidence/summary.json').read_text())
assert r['result'] == 'PASS' and r['local_mutant_stage'] == 'PASS'
assert r['ggml_tree'] == sys.argv[1] and r['tests_tree'] == sys.argv[2], 'sources differ from the Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-s10-k11b: FAIL preflight'; exit 8; }
    echo 'bw-s10-k11b: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-s10-k11b && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/k11b-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-s10-k11b: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    rm -f build-cuda/BUILD_STAMP
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target llama-server test-meta-input-stage -j4
    grep -F 'set(CMAKE_CXX_COMPILER_ID "GNU")' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
    grep -E 'set\(CMAKE_CXX_COMPILER_VERSION "13\.' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    source_stamp > build-cuda/BUILD_STAMP
}
# Both GPUs, both visibility orders: the measured order must follow the physical link, not the rank index.
t=build-cuda/bin/test-meta-input-stage
run_test() {  # LOG VISIBLE [ENV=VAL...]
    local log=$1 vis=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$vis" INPUT_STAGE_REQUIRE_CUDA=1 "$@" timeout 900 nice -n 19 "$t" > "$log" 2>&1
}
ops() {
    [[ $VISIBLE == GPU-*,GPU-* ]] || return 1
    local rev=${VISIBLE#*,},${VISIBLE%%,*}
    run_test "$OUT/stage-fwd-vis.log" "$VISIBLE"
    grep -aq 'cuda input stage: OK' "$OUT/stage-fwd-vis.log"
    grep -aq 'test-meta-input-stage: PASS' "$OUT/stage-fwd-vis.log"
    grep -aq 'slowest host link first: CUDA1 ' "$OUT/stage-fwd-vis.log"
    run_test "$OUT/stage-rev-vis.log" "$rev"
    grep -aq 'cuda input stage: OK' "$OUT/stage-rev-vis.log"
    grep -aq 'slowest host link first: CUDA0 ' "$OUT/stage-rev-vis.log"
    run_test "$OUT/stage-order-fwd.log" "$VISIBLE" GGML_CUDA_INPUT_STAGE_ORDER=fwd
    grep -aq 'slowest host link first: CUDA0 ' "$OUT/stage-order-fwd.log"
    run_test "$OUT/stage-order-rev.log" "$VISIBLE" GGML_CUDA_INPUT_STAGE_ORDER=rev
    grep -aq 'slowest host link first: CUDA1 ' "$OUT/stage-order-rev.log"
    if run_test "$OUT/stage-mutant-cuda.log" "$VISIBLE" GGML_CUDA_INPUT_STAGE_MUTANT=1; then
        echo 'FAIL CUDA staging mutant escaped'; return 1
    fi
    grep -aq 'tensor [AB] differs' "$OUT/stage-mutant-cuda.log"
    if run_test "$OUT/stage-mutant-glue.log" "$VISIBLE" GGML_META_INPUT_STAGE_MUTANT=1; then
        echo 'FAIL meta glue mutant escaped'; return 1
    fi
}
items() {
    cd "$HOME/repos/gpuq"
    python3 -m gpuq.shlint "$ROOT/codex/bw-s10-k11b.sh" "$ROOT/codex/bw-s10-k11b-mac.sh" "$ROOT/codex/nsys/k11b_nsys.sh"
    gpuq submit --check "$ROOT/codex/nsys/k11b_nsys.toml"
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 cuda-stage-identity-and-controls ops
stage 43 check-nsys-item items
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True,items='codex/nsys/k11b_nsys.toml; checked, NOT submitted'),indent=2)+'\n')
PY_RESULT
echo 'bw-s10-k11b: PASS; build-cuda stamped, nsys item checked, NOT submitted'
