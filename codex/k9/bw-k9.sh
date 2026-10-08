#!/usr/bin/env bash
# K9 window: CUDA build of this tree and of a baseline (stack10-cand base + the test harness only), byte-identity, engagement,
# wait-removal mutants and the 12/24-column perf arms on both RTX 5060 Ti. No freeze, no server, no model.
# rc: 0 pass | 5 negative control void | 6 integrity | 8 a correctness or determinism gate fails
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
BASE=0cfc0723e7e991600ca01004efa7cd377400c37b
SCR=/mnt/scratch/s10-k9
K=$ROOT/build-cuda/bin/test-backend-ops
B=$SCR/base/build/bin/test-backend-ops
PRODUCTION='ggml/src/ggml-cuda/common.cuh ggml/src/ggml-cuda/mmq.cuh ggml/src/ggml-cuda/quantize.cu tests/test-backend-ops.cpp'

preflight() {
    [[ $(git branch --show-current) == topic/s10-k9 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc codex/k9-evidence/summary.json codex/k9_summary.py /mnt/scratch; do
        [[ -e $p ]] || return 1
    done
    [[ -w /mnt/scratch ]] || return 1
    [[ $(git diff --name-only "$BASE" HEAD -- src ggml tests common | tr '\n' ' ') == "${PRODUCTION} " ]] || return 1
    python3 - <<'PY_GATE' || return 1
import hashlib, json, subprocess
from pathlib import Path
r = json.loads(Path('codex/k9-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20, 21, 31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-k9: FAIL preflight'; exit 8; }
    echo 'bw-k9: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-k9 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/k9-a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-k9: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] && return 0
    [[ $rc == 5 || $rc == 6 ]] && exit "$rc"
    echo "bw-k9: FAIL $name"
    exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target test-backend-ops -j4
}
build_base() {
    rm -rf "$SCR/base"
    mkdir -p "$SCR/base/src"
    git archive "$BASE" | tar -x -C "$SCR/base/src"
    git show HEAD:tests/test-backend-ops.cpp > "$SCR/base/src/tests/test-backend-ops.cpp"
    nice -n 19 "$W/bin/buildgate" cmake -S "$SCR/base/src" -B "$SCR/base/build" "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build "$SCR/base/build" --target test-backend-ops -j4
}
stamps() {
    local n
    for lib in build-cuda/bin/libggml-cuda.so "$SCR/base/build/bin/libggml-cuda.so"; do
        n=$(strings "$lib" | grep -c DeviceTopK)
        echo "DeviceTopK_count_M=$n $lib"
        [[ $n == 25 ]]
    done
    grep -c 'mmq_pdl:' build-cuda/bin/libggml-cuda.so > /dev/null
    [[ $(grep -ac 'mmq_pdl:' "$SCR/base/build/bin/libggml-cuda.so") == 0 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
RC=0
chain() { # TAG DEV BIN KIND ITERS [ENV=VAL ...]: stdout -> $OUT/TAG.out, stderr -> $OUT/TAG.err, status in RC
    local tag=$1 dev=$2 bin=$3 kind=$4 iters=$5
    shift 5
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 "$bin" mmq-chain "$dev" "$kind" "$iters" > "$OUT/$tag.out" 2> "$OUT/$tag.err" || RC=$?
    printf '%s\t%s\n' "$tag" "$RC" >> "$OUT/runs.tsv"
}
suite() { # TAG DEV TIMEOUT ARGS...: existing test-backend-ops cases on the K9 build
    local tag=$1 dev=$2 tmo=$3
    shift 3
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout "$tmo" nice -n 19 "$K" test -b "$dev" "$@" > "$OUT/$tag.log" 2>&1 || RC=$?
    [[ $RC == 0 ]] || { echo "FAIL $tag rc=$RC"; return 8; }
    [[ $(grep -ac 'FAIL' "$OUT/$tag.log") == 0 ]] || { echo "FAIL $tag has FAIL lines"; return 8; }
    grep -aEq '^ +[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/$tag.log" || { echo "FAIL $tag ran no cases"; return 8; }
}
ops() {
    [[ $VISIBLE == GPU-* ]] || return 6
    local dev
    for dev in CUDA0 CUDA1; do
        # K9 default: engagement of every launch in the chain, then byte identity against every other configuration.
        chain "k9-$dev" "$dev" "$K" hash 200 GGML_CUDA_MMQ_PDL_LOG=1
        [[ $RC == 0 ]] || { echo "FAIL k9-$dev rc=$RC"; return 8; }
        grep -aq '^mmq_pdl: kernel=mmq .* pdl=1' "$OUT/k9-$dev.err" || { echo "FAIL no PDL mmq launch"; return 6; }
        grep -aq '^mmq_pdl: kernel=quantize .* pdl=1' "$OUT/k9-$dev.err" || { echo "FAIL no PDL quantize launch"; return 6; }
        grep -aq '^mmq_pdl: kernel=fixup .* pdl=1' "$OUT/k9-$dev.err" || { echo "FAIL no PDL fixup launch: no shape needs a fixup"; return 6; }
        [[ $(grep -ac 'pdl=0' "$OUT/k9-$dev.err") == 0 ]] || { echo "FAIL a launch ran without PDL"; return 6; }
        [[ $(grep -ac '^chain ' "$OUT/k9-$dev.out") == 36 ]] || { echo "FAIL chain case count"; return 8; }
        awk '$1 == "chain" && $5 == $6 {bad = 1} END {exit bad}' "$OUT/k9-$dev.out" || { echo "FAIL alternating inputs hash alike"; return 5; }

        chain "off-$dev" "$dev" "$K" hash 200 GGML_CUDA_MMQ_PDL=0 GGML_CUDA_MMQ_PDL_LOG=1
        [[ $RC == 0 ]] || { echo "FAIL off-$dev rc=$RC"; return 8; }
        grep -aq '^mmq_pdl: kernel=mmq .* pdl=0' "$OUT/off-$dev.err" || { echo "FAIL off switch did not disable PDL"; return 5; }
        [[ $(grep -ac 'pdl=1' "$OUT/off-$dev.err") == 0 ]] || { echo "FAIL off arm still engaged PDL"; return 5; }
        cmp "$OUT/k9-$dev.out" "$OUT/off-$dev.out" || { echo "FAIL PDL on differs from MMQ-PDL off"; return 8; }

        chain "base-$dev" "$dev" "$B" hash 200
        [[ $RC == 0 ]] || { echo "FAIL base-$dev rc=$RC"; return 8; }
        cmp "$OUT/k9-$dev.out" "$OUT/base-$dev.out" || { echo "FAIL K9 differs from the stock-MMQ baseline binary"; return 8; }

        chain "nopdl-$dev" "$dev" "$K" hash 200 GGML_CUDA_PDL=0
        [[ $RC == 0 ]] || { echo "FAIL nopdl-$dev rc=$RC"; return 8; }
        cmp "$OUT/k9-$dev.out" "$OUT/nopdl-$dev.out" || { echo "FAIL global PDL off differs"; return 8; }

        chain "nograph-$dev" "$dev" "$K" hash 200 GGML_CUDA_DISABLE_GRAPHS=1
        [[ $RC == 0 ]] || { echo "FAIL nograph-$dev rc=$RC"; return 8; }
        cmp "$OUT/k9-$dev.out" "$OUT/nograph-$dev.out" || { echo "FAIL stream launches (no CUDA graph) differ"; return 8; }

        # Slow producers with correct waits must still match: the waits really order the chain.
        chain "stress-$dev" "$dev" "$K" hash 200 GGML_CUDA_MMQ_PDL_MUTANT=24
        [[ $RC == 0 ]] || { echo "FAIL stress-$dev rc=$RC"; return 8; }
        cmp "$OUT/k9-$dev.out" "$OUT/stress-$dev.out" || { echo "FAIL slow producers change the result"; return 8; }

        # Negative controls: the same slow producers without the wait must fail the recompute check.
        chain "mutant-main-$dev" "$dev" "$K" hash 50 GGML_CUDA_MMQ_PDL_MUTANT=9
        [[ $RC == 1 ]] || { echo "negative control void: MMQ without wait passed (rc=$RC)"; return 5; }
        grep -aq 'mmq-chain: FAIL' "$OUT/mutant-main-$dev.err" || { echo "negative control void: no FAIL line"; return 5; }
        chain "mutant-fixup-$dev" "$dev" "$K" hash 50 GGML_CUDA_MMQ_PDL_MUTANT=18
        [[ $RC == 1 ]] || { echo "negative control void: fixup without wait passed (rc=$RC)"; return 5; }
        grep -aq 'mmq-chain: FAIL' "$OUT/mutant-fixup-$dev.err" || { echo "negative control void: no FAIL line"; return 5; }
        # Informational: the unslowed wait-removal mutant (a race that may or may not fire).
        chain "mutant-raw-$dev" "$dev" "$K" hash 200 GGML_CUDA_MMQ_PDL_MUTANT=3
    done
}
suites() {
    [[ $VISIBLE == GPU-* ]] || return 6
    suite mm-CUDA0 CUDA0 3000 -o MUL_MAT
    suite id-CUDA0 CUDA0 3000 -o MUL_MAT_ID
    suite mm-CUDA1 CUDA1 1500 -o MUL_MAT -p 'type_a=(iq4_xs|q4_K|q6_K|q8_0),type_b=f32,m=(3072|5120|6144|8704),n=(9|12|16|24),k=(256|1024|5120),'
    suite id-CUDA1 CUDA1 1500 -o MUL_MAT_ID -p 'type_a=(iq4_xs|q4_K|q6_K|q8_0),type_b=f32,n_mats=4,n_used=2,b=0,m=8704,n=(12|24),k=512,'
}
perf() {
    [[ $VISIBLE == GPU-* ]] || return 6
    local dev arm i=0 arms
    for dev in CUDA0 CUDA1; do
        # A/A on the baseline brackets drift; arms interleave so a slow clock hits every arm alike.
        arms=(base k9 off base)
        [[ $dev == CUDA0 ]] && arms=(base k9 off base k9 off base)
        for arm in "${arms[@]}"; do
            i=$((i+1))
            case $arm in
                base) chain "perf-$dev-base-$i" "$dev" "$B" perf 200 ;;
                k9) chain "perf-$dev-k9-$i" "$dev" "$K" perf 200 ;;
                off) chain "perf-$dev-off-$i" "$dev" "$K" perf 200 GGML_CUDA_MMQ_PDL=0 ;;
            esac
            [[ $RC == 0 ]] || { echo "FAIL perf $dev $arm rc=$RC"; return 8; }
        done
    done
    python3 codex/k9_summary.py "$OUT" > "$OUT/perf.tsv"
}
stage 40 cuda-build build
stage 41 baseline-build build_base
stage 42 cuda-stamps stamps
stage 43 byte-identity-engagement-mutants ops
stage 44 existing-suites suites
stage 45 perf-arms perf
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, perf='perf.tsv'), indent=2) + '\n')
PY_RESULT
echo 'bw-k9: PASS'
