#!/usr/bin/env bash
# Build window for K6: CUDA build of test-backend-ops, byte-identity gates and must-fail mutants on both GPUs.
# Parent-only: this payload is never submitted by its author. No freeze: the parent combines components later.
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
    [[ $(git branch --show-current) == topic/s10-k6 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0cfc0723e HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc /usr/local/cuda-13.2/bin/compute-sanitizer codex/k6/evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY_GATE' || return 1
import hashlib, json, subprocess
from pathlib import Path
r = json.loads(Path('codex/k6/evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20, 21, 22, 31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-s10-k6: FAIL preflight'; exit 8; }
    echo 'bw-s10-k6: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-s10-k6 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/k6/window/k6-a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-s10-k6: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 8
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
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
# run LOG ENV... -- ARGS...: one test-backend-ops invocation with the GPUs exposed; returns its rc.
run() {
    local log=$1
    shift
    local envs=()
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ${envs[@]+"${envs[@]}"} timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test "$@" -j 1 > "$log" 2>&1
}
count() { grep -ac -- "$1" "$2" || true; }
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev n
    for dev in CUDA0 CUDA1; do
        run "$OUT/shared-$dev.log" -- -b "$dev" -o MUL_MAT_SHARED
        grep -aq 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS' "$OUT/shared-$dev.log"
        grep -aq 'q8_reuse:.* rows=12 skipped=[1-9]' "$OUT/shared-$dev.log"
        grep -aq 'q8_reuse:.* rows=16 skipped=[1-9]' "$OUT/shared-$dev.log"
        n=$(count 'q8_reuse:.* rows=17 ' "$OUT/shared-$dev.log"); [[ $n == 0 ]] || { echo "FAIL 17-row input engaged reuse ($n)"; return 1; }
        run "$OUT/shared-off-$dev.log" GGML_CUDA_Q8_REUSE=0 -- -b "$dev" -o MUL_MAT_SHARED
        grep -aq 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS' "$OUT/shared-off-$dev.log"
        n=$(count 'q8_reuse:' "$OUT/shared-off-$dev.log"); [[ $n == 0 ]] || { echo "FAIL reuse off still engaged ($n)"; return 1; }
        if run "$OUT/shared-stale-$dev.log" GGML_CUDA_Q8_REUSE_STALE_MUTANT=1 -- -b "$dev" -o MUL_MAT_SHARED; then
            echo 'FAIL stale-cache mutant escaped'; return 1
        fi
        grep -aq 'FAIL MUL_MAT_SHARED byte identity' "$OUT/shared-stale-$dev.log"
        run "$OUT/addrms-$dev.log" -- -b "$dev" -o RMS_NORM_MUL_ADD,ADD_RMS_NORM
        grep -aq 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS' "$OUT/addrms-$dev.log"
        grep -aq 'add_rms_norm: .* rows=12 ' "$OUT/addrms-$dev.log"
        grep -aq 'add_rms_norm: .* rows=16 ' "$OUT/addrms-$dev.log"
        n=$(count 'add_rms_norm: .* rows=\(17\|33\) ' "$OUT/addrms-$dev.log"); [[ $n == 0 ]] || { echo "FAIL rows above the cap fused ($n)"; return 1; }
        run "$OUT/addrms-off-$dev.log" GGML_CUDA_ADD_RMS_NORM=0 -- -b "$dev" -o RMS_NORM_MUL_ADD,ADD_RMS_NORM
        grep -aq 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS' "$OUT/addrms-off-$dev.log"
        n=$(count 'add_rms_norm:' "$OUT/addrms-off-$dev.log"); [[ $n == 0 ]] || { echo "FAIL residual fusion off still engaged ($n)"; return 1; }
        if run "$OUT/addrms-mutant-$dev.log" GGML_CUDA_ADD_RMS_NORM_MUTANT=1 -- -b "$dev" -o RMS_NORM_MUL_ADD,ADD_RMS_NORM; then
            echo 'FAIL residual numerical mutant escaped'; return 1
        fi
        grep -aq 'FAIL ADD_RMS_NORM byte identity' "$OUT/addrms-mutant-$dev.log"
    done
}
memcheck() {
    [[ $VISIBLE == GPU-* ]] || return 1
    env CUDA_VISIBLE_DEVICES="${VISIBLE%%,*}" timeout 1800 nice -n 19 /usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 9 \
        build-cuda/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_SHARED,RMS_NORM_MUL_ADD,ADD_RMS_NORM -j 1 > "$OUT/memcheck.log" 2>&1
    grep -aq 'ERROR SUMMARY: 0 errors' "$OUT/memcheck.log"
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 cuda-op-identity-and-controls ops
stage 43 cuda-memcheck memcheck
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True), indent=2) + '\n')
PY_RESULT
echo 'bw-s10-k6: PASS'
