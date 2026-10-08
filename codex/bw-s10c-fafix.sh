#!/usr/bin/env bash
# CUDA window for the pkg54 verify-FA fix (fixup of a1fff47fc). Only ggml/src/ggml-cuda/fattn-mma-f16.cuh differs from the
# Mac-gated stack10-cand source 0cfc0723e, so no CPU gate is repeated; preflight enforces that.
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
BASE=0cfc0723e7e991600ca01004efa7cd377400c37b
S9_LIB=$W/frozen/stack9/build/bin
CAND_LIB=$W/frozen/stack10-cand/build/bin
preflight() {
    [[ $(git branch --show-current) == topic/s10c-fafix ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) && ! -e $W/frozen/stack10-cand2 ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    local changed f
    mapfile -t changed < <(git diff --name-only "$BASE" HEAD)
    for f in "${changed[@]}"; do
        case $f in codex/*|ggml/src/ggml-cuda/fattn-mma-f16.cuh) ;; *) return 1 ;; esac
    done
    for p in "$CCCL" "$W/bin/buildgate" "$W/bin/wt" "$W/frozen/stack9/SOURCE" "$W/frozen/stack10-cand/SOURCE" \
             "$S9_LIB/libggml-cuda.so" "$CAND_LIB/libggml-cuda.so" /usr/local/cuda-13.2/bin/nvcc /usr/local/cuda-13.2/bin/cuobjdump \
             /usr/bin/g++ codex/sass_gate.py codex/fa-identity.cpp codex/render_s10c2_items.py codex/items/s10c2-np2-real-j5q.toml.in; do
        [[ -e $p ]] || return 1
    done
    [[ $(sed -n '1p' "$W/frozen/stack9/SOURCE") == cd776e104dbd516c7b74506992d2e8efd7600155 ]] || return 1
    [[ $(sed -n '1p' "$W/frozen/stack10-cand/SOURCE") == "$BASE" ]] || return 1
    python3 - <<'PY_GATE' || return 1
from pathlib import Path
for tested in Path('codex/parser-harness/harness').glob('*.py'):
    assert tested.read_bytes() == (Path.home()/'repos/qwen38-eval/harness'/tested.name).read_bytes(), 'parent parser changed since Mac replay'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-s10c-fafix: FAIL preflight'; exit 8; }
    echo 'bw-s10c-fafix: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-s10c-fafix && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/s10c-fafix-a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-s10c-fafix: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    if [[ $rc != 0 ]]; then
        echo "bw-s10c-fafix: FAIL $name"
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
# Measured stack10-cand regression: <256,256,8,8,Q8_0> 10056 -> 10536 instructions, STL 92 -> 151, <..4,8..> STACK 40 -> 48.
# The control feeds stack10-cand through the same gate and requires it to fail.
sass() {
    python3 codex/sass_gate.py "$OUT/sass" "$S9_LIB/libggml-cuda.so" build-cuda/bin/libggml-cuda.so --control "$CAND_LIB/libggml-cuda.so"
}
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
all_passed() {
    local n m
    read -r n m < <(sed -n 's#.*[^0-9]\([0-9][0-9]*\)/\([0-9][0-9]*\) tests passed.*#\1 \2#p' "$1" | tail -n 1) || true
    echo "$1: passed=${n:-none} total=${m:-none}"
    [[ -n ${n:-} && ${n:-0} == "${m:-x}" && $n -gt 0 ]]
}
# Every FLASH_ATTN_EXT case must pass. A failing case is tolerated only if the stack10-cand build fails the same case on this
# device (a pre-existing failure, listed in the log); anything else, or a crash or timeout without a failing case, fails.
fa_full() {
    local dev=$1 base=$W/wt/stack10-cand/build-cuda/bin/test-backend-ops
    if env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 2700 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -j 1 > "$OUT/fa-$dev.log" 2>&1; then
        all_passed "$OUT/fa-$dev.log"
        return 0
    fi
    echo "fa-$dev: failing cases, comparing with the stack10-cand build"
    grep -a 'FAIL' "$OUT/fa-$dev.log" > "$OUT/fa-$dev.fail.raw" || { echo "FAIL fa-$dev: no failing case line (crash or timeout)"; return 1; }
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/fa-$dev.fail.raw" | LC_ALL=C sort -u > "$OUT/fa-$dev.fail"
    [[ -x $base ]] || { echo "FAIL fa-$dev: no stack10-cand baseline binary"; return 1; }
    env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 2700 nice -n 19 "$base" test -b "$dev" -o FLASH_ATTN_EXT -j 1 > "$OUT/fa-base-$dev.log" 2>&1 || true
    grep -a 'FAIL' "$OUT/fa-base-$dev.log" > "$OUT/fa-base-$dev.fail.raw" || true
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/fa-base-$dev.fail.raw" | LC_ALL=C sort -u > "$OUT/fa-base-$dev.fail"
    LC_ALL=C comm -23 "$OUT/fa-$dev.fail" "$OUT/fa-base-$dev.fail" > "$OUT/fa-$dev.new-failures"
    cat "$OUT/fa-$dev.new-failures"
    [[ ! -s $OUT/fa-$dev.new-failures ]]
}
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev
    for dev in CUDA0 CUDA1; do
        fa_full "$dev"
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p 'hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0' -j 1 > "$OUT/q8-$dev.log" 2>&1
        grep -a 'fattn_q8_decode:' "$OUT/q8-$dev.log" > /dev/null
        all_passed "$OUT/q8-$dev.log"
        env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_Q8_DECODE=0 timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p 'hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0' -j 1 > "$OUT/q8-control-$dev.log" 2>&1
        if grep -aq 'fattn_q8_decode:' "$OUT/q8-control-$dev.log"; then
            echo 'FAIL grouped q8 off control engaged'; return 1
        fi
    done
}
# fa-identity prints one FNV64 digest of the raw f32 output per case, with inputs from a fixed generator, through the public
# ggml API: the same binary runs against the new, the stack10-cand and the stack9 libraries (LD_LIBRARY_PATH).
#   decode : new == stack10-cand            (the single-query path must keep its bytes)
#   verify : new == stack9                  (multi-query q8_0 verify must have its stack9 bytes); stack10-cand is informational
#   other  : new == stack9 and new == cand  (f16 K/V and q8_0 with 2 or 9+ queries are untouched)
runid() { # DEVICE GROUP TAG LIBDIR [ENV...]
    local dev=$1 group=$2 tag=$3 lib=$4
    shift 4
    env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" "$@" timeout 900 "$OUT/fa-identity" "$dev" "$group" \
        > "$OUT/id-$dev-$group-$tag.txt" 2> "$OUT/id-$dev-$group-$tag.err"
}
same() { cmp -s "$OUT/id-$1-$2-$3.txt" "$OUT/id-$1-$2-$4.txt"; }
identity() {
    [[ $VISIBLE == GPU-* ]] || return 1
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -Iggml/include codex/fa-identity.cpp -L build-cuda/bin -lggml -lggml-base -o "$OUT/fa-identity"
    local dev group f
    for dev in CUDA0 CUDA1; do
        for group in decode verify other; do
            runid "$dev" "$group" new "$ROOT/build-cuda/bin"
            runid "$dev" "$group" cand "$CAND_LIB"
            runid "$dev" "$group" s9 "$S9_LIB"
            f=$OUT/id-$dev-$group-new.txt
            [[ $(wc -l < "$f") -ge 10 && $(grep -c UNSUPPORTED "$f") == 0 && $(grep -c 'nonfinite=[1-9]' "$f") == 0 ]] || { echo "FAIL vacuous or non-finite $f"; return 1; }
            echo "$dev $group cases=$(wc -l < "$f") new==cand:$(same "$dev" "$group" new cand && echo yes || echo no) new==s9:$(same "$dev" "$group" new s9 && echo yes || echo no) cand==s9:$(same "$dev" "$group" cand s9 && echo yes || echo no)"
        done
        same "$dev" decode new cand || { echo "FAIL $dev decode bytes differ from stack10-cand"; return 1; }
        same "$dev" verify new s9 || { echo "FAIL $dev verify bytes differ from stack9"; return 1; }
        same "$dev" other new s9 && same "$dev" other new cand || { echo "FAIL $dev untouched paths differ"; return 1; }
        # engagement: the driver's decode runs use the grouped kernels in new and cand, and not with the switch off
        [[ $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-new.err") -gt 0 && $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-cand.err") -gt 0 ]] || { echo "FAIL $dev decode not engaged in the driver"; return 1; }
        runid "$dev" decode q8off "$ROOT/build-cuda/bin" GGML_CUDA_Q8_DECODE=0
        [[ $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-q8off.err") == 0 ]] || { echo "FAIL $dev switch-off control engaged"; return 1; }
        echo "$dev decode on-vs-off differing cases (informational): $(diff "$OUT/id-$dev-decode-new.txt" "$OUT/id-$dev-decode-q8off.txt" | grep -c '^>')"
        runid "$dev" verify f16copy "$ROOT/build-cuda/bin" GGML_CUDA_FA_MMA_Q8_0=0
        echo "$dev verify q8-in-kernel vs f16-copy differing cases (informational): $(diff "$OUT/id-$dev-verify-new.txt" "$OUT/id-$dev-verify-f16copy.txt" | grep -c '^>')"
    done
    # the comparison must see a single flipped digest
    f=$OUT/id-CUDA0-decode-new.txt
    sed '1s/ [0-9a-f]\{16\} / 0000000000000000 /' "$f" > "$OUT/mutated.txt"
    if cmp -s "$f" "$OUT/mutated.txt"; then
        echo 'FAIL digest mutation not seen'; return 1
    fi
}
freeze() {
    nice -n 19 "$W/bin/wt" freeze "$ROOT" stack10-cand2 --build build-cuda --desc 'stack10-cand + multi-query q8_0 FA verify instances restored to stack9 SASS; CUDA gates passed'
    [[ $(sed -n '1p' "$W/frozen/stack10-cand2/SOURCE") == "$SHA" ]]
}
items() { nice -n 19 python3 codex/render_s10c2_items.py --out "$ROOT/codex/items/rendered"; }
stage 40 cuda-build build
FAIL_RC=1 stage 41 sass-gate sass
FAIL_RC=8
stage 42 cuda-stamps stamps
stage 43 cuda-fa-ops ops
stage 44 fa-byte-identity identity
stage 45 freeze-stack10-cand2 freeze
stage 46 render-and-check-items items
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True,items='codex/items/rendered; checked, NOT submitted'),indent=2)+'\n')
PY_RESULT
echo 'bw-s10c-fafix: PASS; frozen and item checked, NOT submitted'
