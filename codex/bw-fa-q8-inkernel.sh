#!/usr/bin/env bash
# CUDA window for in-kernel q8_0 dequantization in the MMA flash-attention kernel at prefill batch sizes (topic/fa-q8-inkernel).
# Source delta over BASE: fattn.cu, fattn-mma-f16.cuh, cp-async.cuh, the q8_0 template instances and tests/test-backend-ops.cpp;
# no CPU-graph source changes, so no Mac gate is repeated (preflight enforces the file set).
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
# the gpuq daemon PATH has no CUDA tools (cuobjdump, nvcc)
export PATH="$HOME/.local/bin:$PATH:/usr/local/cuda-13.2/bin" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BASE=def4a4e20a689047e68e8edd81c97790b69a1de7
REF=$W/frozen/stack10-cand2
REF_LIB=$REF/build/bin
CAND_LIB=$W/frozen/stack10-cand/build/bin
CCX=$W/work/aa-load/ccx01.sh
preflight() {
    [[ $(git branch --show-current) == topic/fa-q8-inkernel ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) && ! -e $W/frozen/fa-q8-inkernel ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    local changed f ref_sha
    mapfile -t changed < <(git diff --name-only "$BASE" HEAD)
    for f in "${changed[@]}"; do
        case $f in
            codex/*|tests/test-backend-ops.cpp) ;;
            ggml/src/ggml-cuda/fattn.cu|ggml/src/ggml-cuda/fattn-mma-f16.cuh|ggml/src/ggml-cuda/cp-async.cuh) ;;
            ggml/src/ggml-cuda/template-instances/*) ;;
            *) return 1 ;;
        esac
    done
    for p in "$CCCL" "$W/bin/buildgate" "$W/bin/wt" "$REF/SOURCE" "$W/frozen/stack10-cand/SOURCE" "$CCX" \
             "$REF_LIB/libggml-cuda.so" "$REF_LIB/test-backend-ops" "$REF_LIB/round-trace" "$CAND_LIB/libggml-cuda.so" \
             /usr/local/cuda-13.2/bin/nvcc /usr/local/cuda-13.2/bin/cuobjdump /usr/bin/g++ \
             codex/fa_q8_sass_gate.py codex/fa-q8-identity.cpp codex/fa-q8-bench.cpp codex/fa_q8_prefill_check.py \
             codex/fa_q8_bench_summary.py codex/fa_q8_reserve.py codex/render_fa_q8_items.py codex/items/fa-q8-depth-j5q.toml.in \
             "$HOME/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf"; do
        [[ -e $p ]] || return 1
    done
    # the reference build must be the base source: only codex/ may differ between it and BASE
    ref_sha=$(sed -n '1p' "$REF/SOURCE")
    git cat-file -e "$ref_sha^{commit}" || return 1
    git merge-base --is-ancestor "$ref_sha" HEAD || return 1
    git diff --quiet "$ref_sha" "$BASE" -- . ':(exclude)codex' || return 1
    [[ $(sed -n '1p' "$W/frozen/stack10-cand/SOURCE") == 0cfc0723e7e991600ca01004efa7cd377400c37b ]] || return 1
    python3 - <<'PY_GATE' || return 1
from pathlib import Path
for tested in Path('codex/parser-harness/harness').glob('*.py'):
    assert tested.read_bytes() == (Path.home()/'repos/qwen38-eval/harness'/tested.name).read_bytes(), 'parent parser changed since Mac replay'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-fa-q8-inkernel: FAIL preflight'; exit 8; }
    echo 'bw-fa-q8-inkernel: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-fa-q8-inkernel && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/fa-q8-inkernel-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
FAIL_RC=8
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-fa-q8-inkernel: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    if [[ $rc != 0 ]]; then
        echo "bw-fa-q8-inkernel: FAIL $name"
        tail -n 40 "$OUT/$name.log" || true
        exit "$FAIL_RC"
    fi
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
# Verify instances <256,256,4|8,8,Q8_0> must keep the base SASS (the control, stack10-cand with its measured regression, must
# fail the same gate); the ten new instances are listed with their resources and spills.
sass() {
    python3 codex/fa_q8_sass_gate.py "$OUT/sass" "$REF_LIB/libggml-cuda.so" build-cuda/bin/libggml-cuda.so --control "$CAND_LIB/libggml-cuda.so"
}
stamps() {
    local n n_ref
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    n_ref=$(strings "$REF_LIB/libggml-cuda.so" | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n ref=$n_ref"
    [[ $n == "$n_ref" && $n -gt 0 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
all_passed() {
    local n m
    read -r n m < <(sed -n 's#.*[^0-9]\([0-9][0-9]*\)/\([0-9][0-9]*\) tests passed.*#\1 \2#p' "$1" | tail -n 1) || true
    echo "$1: passed=${n:-none} total=${m:-none}"
    [[ -n ${n:-} && ${n:-0} == "${m:-x}" && $n -gt 0 ]]
}
# Every FLASH_ATTN_EXT case of the selection must pass. A failing case is tolerated only if the base build (stack10-cand2) fails the
# same case on this device (a pre-existing failure, listed in the log); anything else, or a crash or timeout without a failing case,
# fails. The base build ignores the cap knob, so its failure list is that of the unmodified dispatch.
fa_run() { # DEVICE TAG PATTERN|- [ENV=VALUE...]
    local dev=$1 tag=$2 pat=$3 base=$REF_LIB/test-backend-ops sel=()
    shift 3
    [[ $pat == - ]] || sel=(-p "$pat")
    if env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 3000 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT "${sel[@]}" -j 1 > "$OUT/$tag-$dev.log" 2>&1; then
        all_passed "$OUT/$tag-$dev.log"
        return 0
    fi
    echo "$tag-$dev: failing cases, comparing with the base build"
    grep -a 'FAIL' "$OUT/$tag-$dev.log" > "$OUT/$tag-$dev.fail.raw" || { echo "FAIL $tag-$dev: no failing case line (crash or timeout)"; return 1; }
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag-$dev.fail.raw" | LC_ALL=C sort -u > "$OUT/$tag-$dev.fail"
    env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 3000 nice -n 19 "$base" test -b "$dev" -o FLASH_ATTN_EXT "${sel[@]}" -j 1 > "$OUT/$tag-base-$dev.log" 2>&1 || true
    grep -a 'FAIL' "$OUT/$tag-base-$dev.log" > "$OUT/$tag-base-$dev.fail.raw" || true
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag-base-$dev.fail.raw" | LC_ALL=C sort -u > "$OUT/$tag-base-$dev.fail"
    LC_ALL=C comm -23 "$OUT/$tag-$dev.fail" "$OUT/$tag-base-$dev.fail" > "$OUT/$tag-$dev.new-failures"
    cat "$OUT/$tag-$dev.new-failures"
    [[ ! -s $OUT/$tag-$dev.new-failures ]]
}
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev
    for dev in CUDA0 CUDA1; do
        # default cap: the full suite including the new prefill cases
        fa_run "$dev" fa - 
        # every q8_0 K/V case with the in-kernel path wherever the dispatch can use it (cap 1000000), and with it off (cap 0)
        fa_run "$dev" q8-all 'type_K=q8_0,type_V=q8_0' GGML_CUDA_FA_Q8_QTILES=1000000
        grep -aq 'fattn_q8_mq:' "$OUT/q8-all-$dev.log"
        fa_run "$dev" q8-off 'type_K=q8_0,type_V=q8_0' GGML_CUDA_FA_Q8_QTILES=0
        if grep -aq 'fattn_q8_mq:' "$OUT/q8-off-$dev.log"; then
            echo 'FAIL cap-0 control engaged'; return 1
        fi
        # single-query decode instances are unchanged but must still run
        fa_run "$dev" q8-decode 'hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0'
        grep -aq 'fattn_q8_decode:' "$OUT/q8-decode-$dev.log"
    done
}
# fa-q8-identity prints one FNV64 digest of the raw f32 output per case, with inputs from a fixed generator, through the public
# ggml API: the same binary runs against the new and the base library (LD_LIBRARY_PATH).
#   decode : new == base   (the single-query path keeps its bytes)
#   verify : new == base   (3-8 queries: the same instances at the same occupancy; one tile is always within the cap)
#   other  : new == base   (f16 K/V and q8_0 with 1 or 2 queries are untouched)
#   prefill: see fa_q8_prefill_check.py (B == base, engagement, NMSE, mutant)
runid() { # DEVICE GROUP TAG LIBDIR [ENV...]
    local dev=$1 group=$2 tag=$3 lib=$4
    shift 4
    env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" "$@" timeout 2400 "$OUT/fa-identity" "$dev" "$group" \
        > "$OUT/id-$dev-$group-$tag.txt" 2> "$OUT/id-$dev-$group-$tag.err"
}
same() { cmp -s "$OUT/id-$1-$2-$3.txt" "$OUT/id-$1-$2-$4.txt"; }
build_tools() {
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -Iggml/include codex/fa-q8-identity.cpp -L build-cuda/bin -lggml -lggml-base -o "$OUT/fa-identity"
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -Iggml/include codex/fa-q8-bench.cpp -L build-cuda/bin -lggml -lggml-base -o "$OUT/fa-bench"
}
identity() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev group f
    for dev in CUDA0 CUDA1; do
        for group in decode verify other; do
            runid "$dev" "$group" new "$ROOT/build-cuda/bin"
            runid "$dev" "$group" ref "$REF_LIB"
            f=$OUT/id-$dev-$group-new.txt
            [[ $(wc -l < "$f") -ge 10 && $(grep -c UNSUPPORTED "$f") == 0 && $(grep -c 'nonfinite=[1-9]' "$f") == 0 ]] || { echo "FAIL vacuous or non-finite $f"; return 1; }
            echo "$dev $group cases=$(wc -l < "$f") new==ref:$(same "$dev" "$group" new ref && echo yes || echo no)"
            same "$dev" "$group" new ref || { echo "FAIL $dev $group bytes differ from the base build"; return 1; }
        done
        [[ $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-new.err") -gt 0 && $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-ref.err") -gt 0 ]] || { echo "FAIL $dev decode not engaged in the driver"; return 1; }
        # verify under cap 0 (convert + f16) against the in-kernel default: reported, not a gate here (the prefill group is the gate)
        runid "$dev" verify cap0 "$ROOT/build-cuda/bin" GGML_CUDA_FA_Q8_QTILES=0
        echo "$dev verify in-kernel vs convert differing cases (informational): $(diff "$OUT/id-$dev-verify-new.txt" "$OUT/id-$dev-verify-cap0.txt" | grep -c '^>')"
        runid "$dev" prefill new "$ROOT/build-cuda/bin"
        runid "$dev" prefill ref "$REF_LIB"
        python3 codex/fa_q8_prefill_check.py "$OUT/id-$dev-prefill-new.txt" "$OUT/id-$dev-prefill-ref.txt" | tee "$OUT/prefill-check-$dev.txt"
        grep -aq 'prefill check PASS' "$OUT/prefill-check-$dev.txt"
    done
    # the comparison must see a single flipped digest
    f=$OUT/id-CUDA0-decode-new.txt
    sed '1s/ [0-9a-f]\{16\} / 0000000000000000 /' "$f" > "$OUT/mutated.txt"
    if cmp -s "$f" "$OUT/mutated.txt"; then
        echo 'FAIL digest mutation not seen'; return 1
    fi
}
# Informational: ik (in-kernel) against cv (convert + f16), us per call incl. conversion, ABBA per shape, both GPUs.
bench() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev
    for dev in CUDA0 CUDA1; do
        env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$ROOT/build-cuda/bin" timeout 2400 bash "$CCX" "$OUT/fa-bench" "$dev" > "$OUT/bench-$dev.txt" 2> "$OUT/bench-$dev.err"
    done
    python3 codex/fa_q8_bench_summary.py "$OUT/bench-summary.json" "$OUT/bench-CUDA0.txt" "$OUT/bench-CUDA1.txt"
}
# One boot per setting of the served J5q np2 shape; the compute buffer lines of the main and the draft context are recorded.
# r_* are the base build (cap knob absent) and the new build at -ub 128 (served) and -ub 512; r_q8 is the draft q8_0 KV shape.
reserve_boot() { # TAG EXE LIBDIR UB [ARGS...]
    local tag=$1 exe=$2 lib=$3 ub=$4 rc=0
    shift 4
    env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 600 bash "$CCX" python3 codex/fa_q8_reserve.py "$OUT/reserve-$tag.log" "$exe" "$lib" "$ub" 18911 "$@" || rc=$?
    echo "reserve $tag rc=$rc"
    return "$rc"
}
reserve_max() { sort -g "$OUT/reserve-$1.log.sizes" | tail -n 1 | cut -f1; }
reserve() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local new=$ROOT/build-cuda/bin
    reserve_boot base128 "$REF_LIB/round-trace" "$REF_LIB" 128
    reserve_boot new128 "$new/round-trace" "$new" 128
    reserve_boot base512 "$REF_LIB/round-trace" "$REF_LIB" 512
    reserve_boot new512 "$new/round-trace" "$new" 512
    reserve_boot new512all "$new/round-trace" "$new" 512 GGML_CUDA_FA_Q8_QTILES=1000000
    reserve_boot new128all "$new/round-trace" "$new" 128 GGML_CUDA_FA_Q8_QTILES=1000000
    # the K1 shape (draft KV q8_0) OOMs at boot on the base build; the new build may fit. Informational.
    reserve_boot new128q8 "$new/round-trace" "$new" 128 --draft-q8 || echo 'new128q8 did not boot (informational)'
    local t
    for t in base128 new128 base512 new512 new512all new128all; do
        echo "max compute buffer MiB $t: $(reserve_max $t)"
        cut -f2 "$OUT/reserve-$t.log.sizes"
    done
    # the new build must never reserve more than the base at the same -ub
    awk -v a="$(reserve_max base128)" -v b="$(reserve_max new128)" 'BEGIN { exit !(b <= a + 1) }'
    awk -v a="$(reserve_max base512)" -v b="$(reserve_max new512)" 'BEGIN { exit !(b <= a + 1) }'
    awk -v a="$(reserve_max base512)" -v b="$(reserve_max new512all)" 'BEGIN { exit !(b <= a + 1) }'
}
freeze() {
    nice -n 19 "$W/bin/wt" freeze "$ROOT" fa-q8-inkernel --build build-cuda --desc 'stack10-cand2 + in-kernel q8_0 K/V dequantization in the MMA flash-attention kernel at prefill batch sizes; CUDA gates passed'
    [[ $(sed -n '1p' "$W/frozen/fa-q8-inkernel/SOURCE") == "$SHA" ]]
}
items() { nice -n 19 python3 codex/render_fa_q8_items.py --out "$ROOT/codex/items/rendered"; }
# Falsifies the exactness claim (in-kernel bytes == convert + f16 bytes) without invalidating the artefacts above: it runs last.
strict_bytes() {
    local dev
    for dev in CUDA0 CUDA1; do
        python3 codex/fa_q8_prefill_check.py --strict-bytes "$OUT/id-$dev-prefill-new.txt" "$OUT/id-$dev-prefill-ref.txt"
    done
}
stage 40 cuda-build build
FAIL_RC=1 stage 41 sass-gate sass
FAIL_RC=8
stage 42 cuda-stamps stamps
stage 43 cuda-fa-ops ops
stage 44 build-tools build_tools
stage 45 fa-byte-identity identity
stage 46 fa-bench bench
stage 47 reserve-canary reserve
stage 48 freeze-fa-q8-inkernel freeze
stage 49 render-and-check-items items
stage 50 prefill-bytes-strict strict_bytes
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True,items='codex/items/rendered; checked, NOT submitted'),indent=2)+'\n')
PY_RESULT
echo 'bw-fa-q8-inkernel: PASS; frozen and item checked, NOT submitted'
