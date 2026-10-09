#!/usr/bin/env bash
# K9-fix window: CUDA builds of this tree (k9-fix), the stock baseline and the old K9 (topic/s10-k9 1039f834f); SASS gate
# (fixed MMQ/fixup/quantize kernels vs stock, old K9 as the must-fail control); byte identity, engagement, wait-removal mutants
# and the full MUL_MAT/MUL_MAT_ID suites on the fix; perf arms base/k9old/fix/off on both RTX 5060 Ti. No freeze, no server.
# rc: 0 pass | 5 negative control void | 6 integrity | 8 a correctness, determinism or SASS gate fails
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
K9OLD=1039f834f65044f24d34821186382e24f8620a59
SCR=/mnt/scratch/k9-fix
K=$ROOT/build-cuda/bin/test-backend-ops
B=$SCR/base/build/bin/test-backend-ops
OLD=$SCR/k9old/build/bin/test-backend-ops
CUDA_FILES='ggml/src/ggml-cuda/common.cuh ggml/src/ggml-cuda/mmq.cuh ggml/src/ggml-cuda/quantize.cu'
OBJ=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
CUOBJDUMP=/usr/local/cuda-13.2/bin/cuobjdump

preflight() {
    [[ $(git branch --show-current) == topic/k9-fix ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    [[ $(git cat-file -t "$K9OLD") == commit ]] || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc "$CUOBJDUMP" codex/k9-evidence/summary.json \
             codex/k9fix_summary.py codex/k9fix_sass.py /mnt/scratch; do
        [[ -e $p ]] || return 1
    done
    [[ -w /mnt/scratch ]] || return 1
    # The fix touches CUDA sources only, which the Mac CPU gate does not compile: its evidence (for the test harness) stays
    # valid when it matches the old K9 tree and nothing else differs.
    [[ $(git diff --name-only "$K9OLD" HEAD -- src ggml tests common | tr '\n' ' ') == "${CUDA_FILES} " ]] || return 1
    python3 - "$K9OLD" <<'PY_GATE' || return 1
import hashlib, json, subprocess, sys
from pathlib import Path
r = json.loads(Path('codex/k9-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20, 21, 22, 31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
rev = sys.argv[1]
names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', rev, '--', 'src', 'ggml', 'tests', 'common'],
                                text=True).splitlines()
cat = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
h = hashlib.sha256()
for name in names:
    cat.stdin.write(f'{rev}:{name}\n'.encode())
    cat.stdin.flush()
    size = int(cat.stdout.readline().split()[2])
    data = cat.stdout.read(size)
    cat.stdout.read(1)
    h.update(name.encode() + b'\0' + data)
cat.stdin.close()
cat.wait()
assert h.hexdigest() == r['source_sha256'], 'old K9 source differs from the Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-k9-fix: FAIL preflight'; exit 8; }
    echo 'bw-k9-fix: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-k9-fix && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/k9fix-a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-k9-fix: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] && return 0
    [[ $rc == 5 || $rc == 6 ]] && exit "$rc"
    echo "bw-k9-fix: FAIL $name"
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
build_tree() { # NAME REV: git archive REV + this tree's test file, same flags
    local name=$1 rev=$2
    rm -rf "${SCR:?}/$name"
    mkdir -p "$SCR/$name/src"
    git archive "$rev" | tar -x -C "$SCR/$name/src"
    git show HEAD:tests/test-backend-ops.cpp > "$SCR/$name/src/tests/test-backend-ops.cpp"
    nice -n 19 "$W/bin/buildgate" cmake -S "$SCR/$name/src" -B "$SCR/$name/build" "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build "$SCR/$name/build" --target test-backend-ops -j4
}
builds() {
    build
    build_tree base "$BASE"
    build_tree k9old "$K9OLD"
}
stamps() {
    local n lib
    for lib in build-cuda/bin/libggml-cuda.so "$SCR/base/build/bin/libggml-cuda.so" "$SCR/k9old/build/bin/libggml-cuda.so"; do
        n=$(strings "$lib" | grep -c DeviceTopK)
        echo "DeviceTopK_count_M=$n $lib"
        [[ $n == 25 ]]
    done
    grep -c 'mmq_pdl:' build-cuda/bin/libggml-cuda.so > /dev/null
    grep -c 'mmq_pdl:' "$SCR/k9old/build/bin/libggml-cuda.so" > /dev/null
    [[ $(grep -ac 'mmq_pdl:' "$SCR/base/build/bin/libggml-cuda.so") == 0 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null
    # same build recipe: identical CUDA flags in all three trees
    local f=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make
    [[ $(grep '^CUDA_FLAGS' "build-cuda/$f") == $(grep '^CUDA_FLAGS' "$SCR/base/build/$f") ]]
    [[ $(grep '^CUDA_FLAGS' "build-cuda/$f") == $(grep '^CUDA_FLAGS' "$SCR/k9old/build/$f") ]]
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
sass() {
    # MMQ main + stream-k fixup for the chain types (ggml_type ids: q8_0 8, q4_K 12, q6_K 14, iq4_xs 23) at the served tile
    # widths J=16/24, and every quantize_mmq_q8_1 instance. The fix must match stock; the old K9 must not (control).
    local t d tree id j
    mkdir -p "$OUT/sass"
    for tree in fix base k9old; do
        case $tree in fix) d=$ROOT/build-cuda ;; *) d=$SCR/$tree/build ;; esac
        for t in iq4_xs q4_k q6_k q8_0; do
            (ulimit -v 6000000; nice -n 19 "$CUOBJDUMP" -sass "$d/$OBJ/template-instances/mmq-instance-$t.cu.o") > "$OUT/sass/$tree-mmq-$t.sass"
        done
        (ulimit -v 6000000; nice -n 19 "$CUOBJDUMP" -sass "$d/$OBJ/quantize.cu.o") > "$OUT/sass/$tree-quantize.sass"
        (ulimit -v 6000000; nice -n 19 "$CUOBJDUMP" -res-usage "$d/$OBJ/template-instances/mmq-instance-iq4_xs.cu.o") > "$OUT/sass/$tree-res-iq4_xs.txt"
    done
    for tree in fix k9old; do
        for t in iq4_xs:23 q4_k:12 q6_k:14 q8_0:8; do
            id=${t#*:}
            local kernels=()
            for j in 16 24; do
                kernels+=("_Z9mul_mat_qIL9ggml_type${id}ELi${j}ELb0" "_Z24mul_mat_q_stream_k_fixupIL9ggml_type${id}ELi${j}ELb0")
            done
            python3 codex/k9fix_sass.py "$OUT/sass/base-mmq-${t%%:*}.sass" "$OUT/sass/$tree-mmq-${t%%:*}.sass" "${kernels[@]}" \
                > "$OUT/sass/$tree-vs-base-${t%%:*}.txt" || true
        done
        python3 codex/k9fix_sass.py "$OUT/sass/base-quantize.sass" "$OUT/sass/$tree-quantize.sass" _Z17quantize_mmq_q8_1 \
            > "$OUT/sass/$tree-vs-base-quantize.txt" || true
    done
    local fix_bad fix_ok old_bad
    fix_bad=$(cat "$OUT"/sass/fix-vs-base-*.txt | grep -c '^FAIL' || true)
    fix_ok=$(cat "$OUT"/sass/fix-vs-base-*.txt | grep -c '^PASS _Z' || true)
    old_bad=$(cat "$OUT"/sass/k9old-vs-base-*.txt | grep -c '^FAIL _Z' || true)
    echo "sass: fix kernels PASS=$fix_ok FAIL=$fix_bad; old K9 kernels FAIL=$old_bad"
    grep -aE '^(PASS|FAIL) ' "$OUT"/sass/fix-vs-base-*.txt
    # control: the gate must reject the old K9 (restrict dropped), at least on the iq4_xs J=16 kernel of the a3 regression
    grep -aq '^FAIL _Z9mul_mat_qIL9ggml_type23ELi16ELb0' "$OUT/sass/k9old-vs-base-iq4_xs.txt" || { echo "negative control void: the SASS gate passes the old K9"; return 5; }
    [[ $fix_ok -ge 22 ]] || { echo "FAIL SASS gate compared too few fixed kernels ($fix_ok)"; return 6; }
    [[ $fix_bad == 0 ]] || { echo "FAIL fixed kernels differ from stock beyond the PDL instructions"; return 8; }
}
RC=0
chain() { # TAG DEV BIN KIND ITERS [ENV=VAL ...]: stdout -> $OUT/TAG.out, stderr -> $OUT/TAG.err, status in RC
    local tag=$1 dev=$2 bin=$3 kind=$4 iters=$5
    shift 5
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 "$bin" mmq-chain "$dev" "$kind" "$iters" > "$OUT/$tag.out" 2> "$OUT/$tag.err" || RC=$?
    printf '%s\t%s\n' "$tag" "$RC" >> "$OUT/runs.tsv"
}
suite() { # TAG DEV TIMEOUT ARGS...: existing test-backend-ops cases on the fix build
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
        local fix_rc off_rc base_rc
        chain "fix-$dev" "$dev" "$K" hash 200 GGML_CUDA_MMQ_PDL_LOG=1
        fix_rc=$RC
        chain "off-$dev" "$dev" "$K" hash 200 GGML_CUDA_MMQ_PDL=0 GGML_CUDA_MMQ_PDL_LOG=1
        off_rc=$RC
        chain "base-$dev" "$dev" "$B" hash 200
        base_rc=$RC
        if [[ $fix_rc != 0 || $off_rc != 0 || $base_rc != 0 ]]; then
            echo "FAIL fix-$dev rc=$fix_rc off-$dev rc=$off_rc base-$dev rc=$base_rc"
            [[ $off_rc == 0 && $base_rc == 0 ]] || echo "FAIL also without K9 PDL: stock or test defect"
            return 8
        fi
        grep -aq '^mmq_pdl: kernel=mmq .* pdl=1' "$OUT/fix-$dev.err" || { echo "FAIL no PDL mmq launch"; return 6; }
        grep -aq '^mmq_pdl: kernel=quantize .* pdl=1' "$OUT/fix-$dev.err" || { echo "FAIL no PDL quantize launch"; return 6; }
        grep -aq '^mmq_pdl: kernel=fixup .* pdl=1' "$OUT/fix-$dev.err" || { echo "FAIL no PDL fixup launch: no shape needs a fixup"; return 6; }
        [[ $(grep -ac 'pdl=0' "$OUT/fix-$dev.err") == 0 ]] || { echo "FAIL a launch ran without PDL"; return 6; }
        [[ $(grep -ac '^chain ' "$OUT/fix-$dev.out") == 36 ]] || { echo "FAIL chain case count"; return 8; }
        awk '$1 == "chain" && $5 == $6 {bad = 1} END {exit bad}' "$OUT/fix-$dev.out" || { echo "FAIL alternating inputs hash alike"; return 5; }

        grep -aq '^mmq_pdl: kernel=mmq .* pdl=0' "$OUT/off-$dev.err" || { echo "FAIL off switch did not disable PDL"; return 5; }
        [[ $(grep -ac 'pdl=1' "$OUT/off-$dev.err") == 0 ]] || { echo "FAIL off arm still engaged PDL"; return 5; }
        cmp "$OUT/fix-$dev.out" "$OUT/off-$dev.out" || { echo "FAIL PDL on differs from MMQ-PDL off"; return 8; }

        cmp "$OUT/fix-$dev.out" "$OUT/base-$dev.out" || { echo "FAIL the fix differs from the stock-MMQ baseline binary"; return 8; }

        chain "nopdl-$dev" "$dev" "$K" hash 200 GGML_CUDA_PDL=0
        [[ $RC == 0 ]] || { echo "FAIL nopdl-$dev rc=$RC"; return 8; }
        cmp "$OUT/fix-$dev.out" "$OUT/nopdl-$dev.out" || { echo "FAIL global PDL off differs"; return 8; }

        chain "nograph-$dev" "$dev" "$K" hash 200 GGML_CUDA_DISABLE_GRAPHS=1
        [[ $RC == 0 ]] || { echo "FAIL nograph-$dev rc=$RC"; return 8; }
        cmp "$OUT/fix-$dev.out" "$OUT/nograph-$dev.out" || { echo "FAIL stream launches (no CUDA graph) differ"; return 8; }

        # Slow producers with correct waits must still match: the waits really order the chain.
        chain "stress-$dev" "$dev" "$K" hash 200 GGML_CUDA_MMQ_PDL_MUTANT=24
        [[ $RC == 0 ]] || { echo "FAIL stress-$dev rc=$RC"; return 8; }
        cmp "$OUT/fix-$dev.out" "$OUT/stress-$dev.out" || { echo "FAIL slow producers change the result"; return 8; }

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
    suite mm-CUDA1 CUDA1 3000 -o MUL_MAT
    suite id-CUDA1 CUDA1 3000 -o MUL_MAT_ID
}
perf() {
    [[ $VISIBLE == GPU-* ]] || return 6
    local dev arm i=0 arms
    for dev in CUDA0 CUDA1; do
        # A/A on the baseline brackets drift; arms interleave so a slow clock hits every arm alike.
        arms=(base k9old fix off base)
        [[ $dev == CUDA0 ]] && arms=(base k9old fix off base fix k9old off base)
        for arm in "${arms[@]}"; do
            i=$((i+1))
            case $arm in
                base) chain "perf-$dev-base-$i" "$dev" "$B" perf 200 ;;
                k9old) chain "perf-$dev-k9old-$i" "$dev" "$OLD" perf 200 ;;
                fix) chain "perf-$dev-fix-$i" "$dev" "$K" perf 200 ;;
                off) chain "perf-$dev-off-$i" "$dev" "$K" perf 200 GGML_CUDA_MMQ_PDL=0 ;;
            esac
            [[ $RC == 0 ]] || { echo "FAIL perf $dev $arm rc=$RC"; return 8; }
        done
    done
    python3 codex/k9fix_summary.py "$OUT" > "$OUT/perf.tsv" 2> "$OUT/perf-verdict.txt"
    cat "$OUT/perf-verdict.txt"
}
stage 40 cuda-builds builds
stage 42 cuda-stamps stamps
stage 43 sass-gate sass
stage 44 byte-identity-engagement-mutants ops
stage 45 existing-suites suites
stage 46 perf-arms perf
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
verdict = (p/'perf-verdict.txt').read_text().splitlines()
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, perf='perf.tsv',
                                              perf_verdict=verdict), indent=2) + '\n')
PY_RESULT
echo 'bw-k9-fix: PASS'
