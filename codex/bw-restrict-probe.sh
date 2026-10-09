#!/usr/bin/env bash
# Restrict-probe window: does upstream #24030 (GGML_CUDA_RESTRICT empty on Hopper+ with PDL) cost kernel time outside MMQ?
# Two CUDA builds of this tree (stock; -DGGML_CUDA_FORCE_RESTRICT keeps __restrict__ with PDL on), SASS load census of the touched
# objects (must-differ control: mmvq.cu.o), test-backend-ops perf at served decode shapes, arms interleaved stock forced stock forced
# stock. Perf-only: the forced build reintroduces the #24030 hazard; its outputs are never checked for identity and it is never served.
# rc: 0 pass | 5 negative control void | 6 integrity | 8 deterministic failure (build, filter count, run)
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export BUILDGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BASE=45d90d3e4ec1b713bef3804bac5bfc93cbff94c1
SCR=/mnt/scratch/restrict-probe
BIN_STOCK=$SCR/stock/build/bin/test-backend-ops
BIN_FORCED=$SCR/forced/build/bin/test-backend-ops
OBJ=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
CUOBJDUMP=/usr/local/cuda-13.2/bin/cuobjdump
FORCE=-DGGML_CUDA_FORCE_RESTRICT

preflight() {
    [[ $(git branch --show-current) == topic/restrict-probe ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    [[ $(grep -c -- "GGML_CUDA_FORCE_RESTRICT" ggml/src/ggml-cuda/common.cuh) -ge 1 ]] || return 1
    local p
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc "$CUOBJDUMP" codex/restrict_sass.py codex/restrict_summary.py \
             /mnt/scratch; do
        [[ -e $p ]] || return 1
    done
    [[ -w /mnt/scratch ]] || return 1
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-restrict-probe: FAIL preflight'; exit 8; }
    echo 'bw-restrict-probe: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-restrict-probe && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/restrict-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-restrict-probe: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] && return 0
    [[ $rc == 5 || $rc == 6 ]] && exit "$rc"
    echo "bw-restrict-probe: FAIL $name"
    exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build_arm() { # NAME [extra cmake args]: same source, same recipe; only CMAKE_CUDA_FLAGS differs
    local name=$1
    shift
    nice -n 19 "$W/bin/buildgate" cmake -S "$ROOT" -B "$SCR/$name/build" "${flags[@]}" "$@"
    nice -n 19 "$W/bin/buildgate" cmake --build "$SCR/$name/build" --target test-backend-ops -j4
}
builds() {
    mkdir -p "$SCR"
    build_arm stock
    build_arm forced "-DCMAKE_CUDA_FLAGS=$FORCE"
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
stamps() {
    local arm n f
    for arm in stock forced; do
        n=$(strings "$SCR/$arm/build/bin/libggml-cuda.so" | grep -c DeviceTopK || true)
        echo "DeviceTopK_count_M=$n $arm"
        [[ $n == 25 ]]
    done
    f=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make
    [[ $(grep -c -- "$FORCE" "$SCR/forced/build/$f" || true) -ge 1 ]]
    [[ $(grep -c -- "$FORCE" "$SCR/stock/build/$f" || true) == 0 ]]
    [[ $(grep -c '120a' "$SCR/stock/build/$f" || true) -ge 1 ]]
    # same recipe: identical CUDA_FLAGS apart from the switch
    [[ $(grep '^CUDA_FLAGS' "$SCR/forced/build/$f" | sed "s/$FORCE//; s/  */ /g") == $(grep '^CUDA_FLAGS' "$SCR/stock/build/$f" | sed 's/  */ /g') ]]
    strings "$SCR/forced/build/bin/libggml-base.so" | grep -c "$(git rev-parse --short HEAD)" > /dev/null
}
sass() {
    (ulimit -v 6000000; nice -n 19 python3 codex/restrict_sass.py "$SCR/stock/build/$OBJ" "$SCR/forced/build/$OBJ" "$CUOBJDUMP" \
        > "$OUT/sass-counts.tsv" 2> "$OUT/sass-verdict.txt") || { local rc=$?; cat "$OUT/sass-verdict.txt"; return $rc; }
    cat "$OUT/sass-verdict.txt"
}

# Served decode shapes, one GPU of the two-way split (model metadata: qwen35, n_embd 5120, ffn 17408, 24 Q / 4 KV heads, key/value
# length 256, ssm group 16 x state 128, time_step_rank 48, conv kernel 4, vocab shard 124160). Cases are in tests/test-backend-ops.cpp
# ("local: served decode shapes"); each filter's row count must equal the case count (rc 8 otherwise).
SETS=(mm fa conv gdn gr sr scan)
declare -A OPNAME=([mm]=MUL_MAT [fa]=FLASH_ATTN_EXT [conv]=SSM_CONV [gdn]=GATED_DELTA_NET [gr]=GET_ROWS [sr]=SET_ROWS [scan]=SSM_SCAN)
declare -A PARAMS=(
    [mm]='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(8704|5120|6144|3072|124160),n=[1-8],k=(5120|8704|3072),bs=\[1,1\],nr=\[1,1\]'
    [fa]='hsk=256,hsv=256,nh=2,nr23=\[6,1\],kv=(8192|32768|90112),nb=(1|6|12),mask=1,.*type_K=f16,type_V=f16,permute=\[0,2,1,3\]'
    [conv]='ne_b=\[4,5120,'
    [gdn]='head_count=8,head_size=128,'
    [gr]='type=f32,n=(393216,m=8,r=[12]|5120,m=6,r=6),'
    [sr]='ne=\[512,90112,1,1\]'
    [scan]='n_seq_tokens=1,n_seqs=1,')
declare -A EXPECT=([mm]=242 [fa]=9 [conv]=3 [gdn]=3 [gr]=3 [sr]=3 [scan]=2)
declare -A TMO=([mm]=1800 [fa]=600 [conv]=600 [gdn]=600 [gr]=600 [sr]=600 [scan]=600)
RC=0
run_set() { # TAG DEV BIN SET
    local tag=$1 dev=$2 bin=$3 set=$4 n ok
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout "${TMO[$set]}" nice -n 19 "$bin" perf -b "$dev" -o "${OPNAME[$set]}" \
        -p "${PARAMS[$set]}" --output sql > "$OUT/$tag-$set.out" 2> "$OUT/$tag-$set.err" || RC=$?
    printf '%s\t%s\t%s\n' "$tag" "$set" "$RC" >> "$OUT/runs.tsv"
    [[ $RC == 0 ]] || { echo "FAIL $tag-$set rc=$RC"; return 8; }
    n=$(grep -ac '^INSERT INTO test_backend_ops' "$OUT/$tag-$set.out" || true)
    ok=$(grep -ac "^INSERT INTO test_backend_ops.*'perf', '1', '1'" "$OUT/$tag-$set.out" || true)
    [[ $n == "${EXPECT[$set]}" && $ok == "$n" ]] || { echo "FAIL $tag-$set matched $n (supported+passed $ok), expected ${EXPECT[$set]}"; return 8; }
}
clocks() {
    nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv,noheader 2>/dev/null \
        | sed "s/^/$1,/" >> "$OUT/clocks.csv" || true
}
perf() {
    [[ $VISIBLE == GPU-* ]] || return 6
    local dev arm i=0 set
    for dev in CUDA0 CUDA1; do
        for arm in stock forced stock forced stock; do
            i=$((i+1))
            clocks "$dev-$arm-$i-pre"
            for set in "${SETS[@]}"; do
                # CUDA1 covers the MMVQ/MMQ matrices only
                [[ $dev == CUDA1 && $set != mm ]] && continue
                case $arm in
                    stock) run_set "perf-$dev-stock-$i" "$dev" "$BIN_STOCK" "$set" ;;
                    forced) run_set "perf-$dev-forced-$i" "$dev" "$BIN_FORCED" "$set" ;;
                esac
            done
            clocks "$dev-$arm-$i-post"
        done
    done
    local rc=0
    python3 codex/restrict_summary.py "$OUT" > "$OUT/perf.tsv" 2> "$OUT/perf-verdict.txt" || rc=$?
    cat "$OUT/perf-verdict.txt"
    return $rc
}
stage 40 cuda-builds builds
stage 42 cuda-stamps stamps
stage 43 sass-census sass
stage 46 perf-arms perf
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, perf='perf.tsv',
    served='served.tsv', sass='sass-counts.tsv', perf_verdict=(p/'perf-verdict.txt').read_text().splitlines(),
    sass_verdict=(p/'sass-verdict.txt').read_text().splitlines()), indent=2) + '\n')
PY_RESULT
echo 'bw-restrict-probe: PASS'
