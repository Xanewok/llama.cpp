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
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
preflight() {
    [[ $(git branch --show-current) == topic/s10-k4 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0cfc0723e HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc "$MODEL" codex/k4-evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY_GATE' || return 1
import hashlib, json
from pathlib import Path
r = json.loads(Path('codex/k4-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20, 21, 22, 23, 24, 31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
h = hashlib.sha256()
for name in r['source_files']:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-k4: FAIL preflight'; exit 8; }
    echo 'bw-k4: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-k4 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/k4-a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-k4: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target test-backend-ops test-recurrent-state-rollback -j4
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
tbo() { # DEV LOG ENV...: test-backend-ops on one device, rc in $RC
    local dev=$1 log=$2
    shift 2
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o SSM_CONV_RS -j 1 > "$log" 2>&1 || RC=$?
}
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev m
    for dev in CUDA0 CUDA1; do
        tbo "$dev" "$OUT/ops-$dev.log" GGML_SSM_CONV_RS_LOG=1
        [[ $RC == 0 ]]
        grep -aq 'byte identity with the gathered path and changing-id replay PASS cases=[1-9][0-9]* unsupported=[1-9]' "$OUT/ops-$dev.log"
        grep -aq 'ssm_conv_rs engaged:.* silu=1' "$OUT/ops-$dev.log"
        grep -aq 'ssm_conv_rs engaged:.* silu=0' "$OUT/ops-$dev.log"
        # the fused silu epilogue must be exercised on the whole-graph cases too
        grep -aEq '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/ops-$dev.log"
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o SSM_CONV -j 1 > "$OUT/ssm-conv-$dev.log" 2>&1
        for m in 1 2; do
            tbo "$dev" "$OUT/ops-mutant$m-$dev.log" GGML_SSM_CONV_RS_MUTANT=$m
            if [[ $RC == 0 ]]; then echo "FAIL ssm_conv_rs mutant $m escaped on $dev"; return 1; fi
            grep -aq 'FAIL SSM_CONV_RS byte identity' "$OUT/ops-mutant$m-$dev.log"
        done
        tbo "$dev" "$OUT/ops-off-$dev.log" GGML_SSM_CONV_RS_OFF=1 GGML_SSM_CONV_RS_LOG=1
        [[ $RC == 0 ]]
        grep -aq 'PASS cases=0 ' "$OUT/ops-off-$dev.log"
        if grep -aq 'ssm_conv_rs engaged' "$OUT/ops-off-$dev.log"; then echo "FAIL off control engaged on $dev"; return 1; fi
    done
}
# arm DEV NAME EXPECT REF ENV...: whole model on one GPU; raw logits of the gathered arm are the reference of the others
arm() {
    local dev=$1 name=$2 expect=$3 ref=$4
    shift 4
    local cmp=(--getter-logits "$OUT/model/$dev-$name.bin")
    [[ -z $ref ]] || cmp+=(--getter-reference "$OUT/model/$dev-$ref.bin")
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-recurrent-state-rollback --getter-gate "${cmp[@]}" \
        --conv-expect "$expect" -m "$MODEL" -c 4096 -t 4 -ngl 99 -dev "$dev" -lv 5 > "$OUT/model/$dev-$name.log" 2>&1 || RC=$?
}
model() {
    mkdir -p "$OUT/model"
    local dev m
    for dev in CUDA0 CUDA1; do
        arm "$dev" gathered gathered '' GGML_SSM_CONV_RS_OFF=1
        [[ $RC == 0 ]]
        arm "$dev" rs rs gathered GGML_SSM_CONV_RS_LOG=1
        [[ $RC == 0 ]]
        cmp "$OUT/model/$dev-gathered.bin" "$OUT/model/$dev-rs.bin"
        grep -aq 'ssm_conv_rs engaged' "$OUT/model/$dev-rs.log"
        for m in 1 2; do
            arm "$dev" mutant$m rs gathered GGML_SSM_CONV_RS_MUTANT=$m
            if [[ $RC == 0 ]]; then echo "FAIL model mutant $m escaped on $dev"; return 1; fi
            grep -aq 'logits differ' "$OUT/model/$dev-mutant$m.log"
        done
    done
}
# Tensor-parallel slicing is the least certain part: recorded, not gating.
meta() {
    mkdir -p "$OUT/meta"
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_SSM_CONV_RS_OFF=1 timeout 900 nice -n 19 build-cuda/bin/test-recurrent-state-rollback --getter-gate \
        --getter-logits "$OUT/meta/off.bin" -m "$MODEL" -c 4096 -t 4 -ngl 99 -sm tensor -lv 5 > "$OUT/meta/off.log" 2>&1 || RC=$?
    echo "meta off rc=$RC"
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_SSM_CONV_RS_LOG=1 timeout 900 nice -n 19 build-cuda/bin/test-recurrent-state-rollback --getter-gate \
        --getter-reference "$OUT/meta/off.bin" -m "$MODEL" -c 4096 -t 4 -ngl 99 -sm tensor -lv 5 > "$OUT/meta/on.log" 2>&1 || RC=$?
    echo "meta on rc=$RC"
    return 0
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 cuda-op-identity-and-mutants ops
stage 43 cuda-model-logits-and-mutants model
stage 44 meta-recorded meta
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True),indent=2)+'\n')
PY_RESULT
echo 'bw-k4: PASS'
