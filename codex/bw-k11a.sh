#!/usr/bin/env bash
# K11a window: CUDA build of the getter sync elision, then cudaStreamSynchronize per np2 round with the elision off / on / debug-checked.
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
preflight() {
    [[ $(git branch --show-current) == topic/s10-k11a ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0cfc0723e HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc codex/k11a/mac-evidence/summary.json \
             "$HOME/repos/attrib-pack/attrib_pack.py" "$HOME/repos/attrib-pack/gpu/llama_drive.py"; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY_GATE' || return 1
import hashlib, json
from pathlib import Path
r = json.loads(Path('codex/k11a/mac-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20, 21, 22, 23} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
h = hashlib.sha256()
for name in r['source_files']:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
PY_GATE
    DRY_RUN=1 ROOT="$ROOT" OUT="$ROOT/codex/window/preflight" bash codex/k11a/sync_count.sh > /dev/null || return 1
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-k11a: FAIL preflight'; exit 8; }
    echo 'bw-k11a: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-k11a && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/k11a-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
STARTED=$(date '+%Y-%m-%d %H:%M:%S')
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-k11a: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target llama-server -j4
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
    [[ $(strings build-cuda/bin/libllama.so | grep -c 'debug check engaged') -ge 1 ]]
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
sync_count() {
    [[ $VISIBLE == GPU-* ]] || return 1
    env CUDA_VISIBLE_DEVICES="$VISIBLE" DRY_RUN=0 ROOT="$ROOT" OUT="$OUT/sync" PORT="${PORT:?}" XID_SINCE="$STARTED" BUDGET_S=3000 \
        bash codex/k11a/sync_count.sh
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 sync-count sync_count
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, sync_count=json.loads((p/'sync'/'sync-count.json').read_text())), indent=2)+'\n')
PY_RESULT
echo 'bw-k11a: PASS'
