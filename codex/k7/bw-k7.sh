#!/usr/bin/env bash
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
EVIDENCE=codex/k7/mac-evidence/summary.json
preflight() {
    [[ $(git branch --show-current) == topic/s10-k7 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0cfc0723e HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc "$EVIDENCE"; do
        [[ -e $p ]] || return 1
    done
    python3 - "$EVIDENCE" <<'PY_GATE' || return 1
import hashlib, json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['result'] == 'PASS' and {20, 21, 22, 23} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
h = hashlib.sha256()
for name in r['source_files']:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-k7: FAIL preflight'; exit 8; }
    echo 'bw-k7: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-k7 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/k7/window/k7-a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-k7: $name rc=$rc wall_s_M=$((SECONDS-t0))"
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
# run NAME DEVICE [VAR=VALUE...]: test-backend-ops on the pair case, log to $OUT/NAME-DEVICE.log, return its status
run_pair() {
    local name=$1 dev=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o RMS_NORM_SCALE_PAIR -j 1 > "$OUT/$name-$dev.log" 2>&1
}
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev mask msg
    for dev in CUDA0 CUDA1; do
        # byte identity with CUDA graphs (default) and without, and engagement of the paired launch
        run_pair pair "$dev"
        grep -c 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$OUT/pair-$dev.log" > /dev/null
        grep -c 'rms_norm_scale_pair_fusion:' "$OUT/pair-$dev.log" > /dev/null
        run_pair pair-eager "$dev" GGML_CUDA_DISABLE_GRAPHS=1
        grep -c 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$OUT/pair-eager-$dev.log" > /dev/null
        grep -c 'rms_norm_scale_pair_fusion:' "$OUT/pair-eager-$dev.log" > /dev/null
        # off switch: identical bytes, no paired launch
        run_pair pair-off "$dev" GGML_CUDA_RMS_NORM_SCALE_PAIR=0
        grep -c 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$OUT/pair-off-$dev.log" > /dev/null
        if grep -c 'rms_norm_scale_pair_fusion:' "$OUT/pair-off-$dev.log" > /dev/null; then
            echo 'FAIL pair off switch engaged'; return 1
        fi
        # numerical mutants must be caught by the byte comparison while the paired launch is engaged
        for mask in 1 2; do
            if run_pair "mutant-$mask" "$dev" GGML_CUDA_RMS_NORM_SCALE_PAIR_MUTANT=$mask; then
                echo "FAIL pair mutant $mask escaped"; return 1
            fi
            grep -c 'FAIL RMS_NORM_SCALE_PAIR byte identity' "$OUT/mutant-$mask-$dev.log" > /dev/null
            grep -c 'rms_norm_scale_pair_fusion:' "$OUT/mutant-$mask-$dev.log" > /dev/null
        done
        # each matcher guard must be what rejects its layout
        for mask in 1 2 4 8; do
            case $mask in
                1) msg='first output over the second input' ;;
                2) msg='second output over the first input' ;;
                4) msg='outputs overlap' ;;
                8) msg='second input is a view of the first output' ;;
            esac
            if run_pair "alias-mutant-$mask" "$dev" GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT=$mask; then
                echo "FAIL alias mutant $mask escaped"; return 1
            fi
            grep -c "FAIL RMS_NORM_SCALE_PAIR matcher accepted: $msg" "$OUT/alias-mutant-$mask-$dev.log" > /dev/null
        done
        # neighbouring RMS norm fusions keep passing
        for op in RMS_NORM RMS_NORM_MUL_ADD RMS_NORM_MUL_ROPE SCALE; do
            env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o "$op" -j 1 > "$OUT/neighbour-$op-$dev.log" 2>&1
            grep -E '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/neighbour-$op-$dev.log" > /dev/null
        done
    done
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 cuda-pair-identity-and-controls ops
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True), indent=2)+'\n')
PY_RESULT
echo 'bw-k7: PASS'
