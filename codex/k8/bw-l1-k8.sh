#!/usr/bin/env bash
# Build window (gpuq filler, never submits): K8 CUDA build of test-backend-ops, bitwise/correctness gates and must-fail mutants on both GPUs,
# k8-bin assembly, perf item render + gpuq submit --check. No freeze, no llama-server.
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
preflight() {
    [[ $(git branch --show-current) == topic/l1-k8 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) && ! -e k8-bin ]] || return 1
    git merge-base --is-ancestor ace9a6d01 HEAD || return 1
    [[ ! -e $EVAL/orchestration/gpuq/items/drafts/k8-mmq-perf.toml ]] || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc /usr/bin/patchelf "$W/frozen/stack8/SOURCE" "$W/work/aa-load/ccx01.sh" codex/k8/mac-evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY_GATE' || return 1
import hashlib, json
from pathlib import Path
r = json.loads(Path('codex/k8/mac-evidence/summary.json').read_text())
assert r['result'] == 'PASS' and {s['name'] for s in r['stages'] if s['rc'] == 0} >= {'1-sim', '2-build', '3-cases', '4-hash'}
h = hashlib.sha256()
for name in r['source_files']:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-l1-k8: FAIL preflight'; exit 8; }
    echo 'bw-l1-k8: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-l1-k8 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/k8/window/l1-k8-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || { echo 'bw-l1-k8: FAIL preflight'; exit 8; }
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
    echo "bw-l1-k8: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || { echo "bw-l1-k8: FAIL $name rc=$rc"; exit 8; }
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target test-backend-ops -j4
    grep -F 'set(CMAKE_CXX_COMPILER_ID "GNU")' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake > /dev/null
    grep -E 'set\(CMAKE_CXX_COMPILER_VERSION "13\.' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake > /dev/null
}
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null
    grep -aq 'GGML_CUDA_MMQ_SK_MUTANT' build-cuda/bin/libggml-cuda.so
    grep -aq 'units=%d parts=%d blocks=%d' build-cuda/bin/libggml-cuda.so
    grep -aq 'mmq-hash' build-cuda/bin/test-backend-ops
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
# Same layout as bin/wt freeze (RUNPATH $ORIGIN, SHA256SUMS, SOURCE), without the freeze: the parent combines components later.
assemble() {
    local f so real
    mkdir k8-bin
    cp -p build-cuda/bin/test-backend-ops k8-bin/
    for so in build-cuda/bin/*.so*; do cp -a "$so" k8-bin/; done
    for f in k8-bin/*; do
        [[ -L $f ]] || patchelf --set-rpath '$ORIGIN' "$f"
    done
    real=$(cd k8-bin && pwd -P)
    while read -r f; do
        [[ $(readlink -f "$f") == "$real"/* ]] || { echo "resolves outside k8-bin: $f"; return 1; }
    done < <(env -u LD_LIBRARY_PATH ldd k8-bin/test-backend-ops | awk '/libggml/ {print $3}')
    [[ -z $(env -u LD_LIBRARY_PATH ldd k8-bin/test-backend-ops | grep 'not found' || true) ]]
    (cd k8-bin && find . -maxdepth 1 \( -type f -o -type l \) | sed 's#^\./##' | grep -v '^SHA256SUMS$' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS)
    printf '%s CUDA 13.2 120a CCCL343 NCCL OFF, K8 test-backend-ops build\n' "$SHA" > k8-bin/SOURCE
}
checks() {
    [[ $VISIBLE == GPU-* ]] || return 1
    export CUDA_VISIBLE_DEVICES="$VISIBLE"
    COMMIT=$SHA OUT=$OUT/checks-canary BUDGET_S=600 bash codex/k8/gpu/k8-gpu.sh checks --canary
    COMMIT=$SHA OUT=$OUT/checks BUDGET_S=9000 bash codex/k8/gpu/k8-gpu.sh checks --run
}
items() { python3 codex/k8/render_k8_items.py "$SHA" "$OUT"; }
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 bin-assemble assemble
stage 43 cuda-gates-and-controls checks
stage 44 render-and-check-items items
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p / 'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, items='checked, NOT submitted'), indent=2) + '\n')
PY_RESULT
echo 'bw-l1-k8: PASS; perf item checked, NOT submitted'
