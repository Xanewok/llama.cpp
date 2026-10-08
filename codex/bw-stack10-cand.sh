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
preflight() {
    [[ $(git branch --show-current) == topic/stack10-cand ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) && ! -e $W/frozen/stack10-cand ]] || return 1
    git merge-base --is-ancestor cd776e104 HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" "$W/bin/wt" "$W/frozen/stack9/SOURCE" /usr/local/cuda-13.2/bin/nvcc codex/stack10-evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY_GATE' || return 1
import hashlib, json
from pathlib import Path
r = json.loads(Path('codex/stack10-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20,21,22,31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
h = hashlib.sha256()
for name in r['source_files']:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from Mac CPU gate'
s = r['static_replay']
assert s['result'] == 'PASS' and any(t['code'] == 31 and t['rc'] == 0 for t in s['stages'])
ph = hashlib.sha256()
for name in s['payload_files']:
    ph.update(name.encode()+b'\0'+Path(name).read_bytes())
assert ph.hexdigest() == s['payload_sha256'], 'payload differs from final Mac static replay'
for tested in Path('codex/parser-harness/harness').glob('*.py'):
    assert tested.read_bytes() == (Path.home()/'repos/qwen38-eval/harness'/tested.name).read_bytes(), 'parent parser changed since Mac replay'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-stack10-cand: FAIL preflight'; exit 8; }
    echo 'bw-stack10-cand: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-stack10-cand && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/stack10-cand-a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-stack10-cand: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target llama-server test-backend-ops -j4
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
}
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    for dev in CUDA0 CUDA1; do
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o ADD_SOFTPLUS_MUL -j 1 > "$OUT/alpha-$dev.log" 2>&1
        grep -q 'same-backend byte identity and changing-input replay PASS' "$OUT/alpha-$dev.log"
        grep -q 'alpha_gate_fusion:' "$OUT/alpha-$dev.log"
        if env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_ALPHA_GATE_MUTANT=1 timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o ADD_SOFTPLUS_MUL -j 1 > "$OUT/alpha-mutant-$dev.log" 2>&1; then
            echo 'FAIL alpha mutant escaped'; return 1
        fi
        grep -q 'FAIL ADD_SOFTPLUS_MUL byte identity' "$OUT/alpha-mutant-$dev.log"
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p 'hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0' -j 1 > "$OUT/q8-$dev.log" 2>&1
        grep -q 'fattn_q8_decode:' "$OUT/q8-$dev.log"
        grep -E '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/q8-$dev.log" >/dev/null
        env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_Q8_DECODE=0 timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p 'hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0' -j 1 > "$OUT/q8-control-$dev.log" 2>&1
        if grep -q 'fattn_q8_decode:' "$OUT/q8-control-$dev.log"; then
            echo 'FAIL grouped q8 off control engaged'; return 1
        fi
    done
}
freeze() {
    nice -n 19 "$W/bin/wt" freeze "$ROOT" stack10-cand --build build-cuda --desc 'stack9 + grouped q8 decode, persistent rotations, alpha gate; Mac CPU and CUDA op gates passed'
    [[ $(sed -n '1p' "$W/frozen/stack10-cand/SOURCE") == "$SHA" ]]
}
items() { nice -n 19 python3 codex/render_stack10_items.py --out "$ROOT/codex/items/rendered"; }
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 cuda-op-identity-and-controls ops
stage 43 freeze-stack10-cand freeze
stage 44 render-and-check-items items
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True,items='codex/items/rendered; checked, NOT submitted'),indent=2)+'\n')
PY_RESULT
echo 'bw-stack10-cand: PASS; frozen and items checked, NOT submitted'
