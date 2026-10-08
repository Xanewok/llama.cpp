#!/usr/bin/env bash
# Build window with GPU checks (gpuq filler; submits nothing). Usage: bw-s9-gdn4.sh --preflight | OUT_DIR
# stack9 cd776e104 + #30087 (GDN four state columns per warp), HEAD of topic/s9-gdn4.
#   40 server build (stack9 recipe: sm_120a, CUDA 13.2, CCCL v3.4.3, NCCL off, tests off)   41 stamps and kernel symbols
#   42 test-backend-ops variants from one test build dir (base kernel, candidate, two kernel mutants, old perf harness)
#   43 correctness on CUDA0 and CUDA1 against the CPU backend; mutants must fail; joint-path engagement proof
#   44 cuobjdump resource usage of every gated_delta_net_cuda instance: no stack frame, no local memory (spills)
#   45 kernel perf base vs candidate at the served shapes (ABBA, both GPUs); not a gate, read perf.tsv
#   46 bin/wt freeze as frozen/stack9-gdn4   47 render the np1 depth and np2 real timing items against it; gpuq submit --check only
# rc: 0 pass | 3 Xid | 4 GPU missing or busy | 5 a mutant escaped (control inconclusive) | 6 integrity | 7 budget | 8 deterministic failure
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
BUILDGATE=$W/bin/buildgate
CCX01=$W/work/aa-load/ccx01.sh
NAME=stack9-gdn4
BASE_SHA=cd776e104dbd516c7b74506992d2e8efd7600155
F=ggml/src/ggml-cuda/gated_delta_net.cu
T=tests/test-backend-ops.cpp
H=codex/s9gdn4
EVIDENCE=$H/mac-evidence/summary.json
VAR=$W/work/s9-gdn4-variants
GPUS=(GPU-ee9a3956-297c-604b-b4c1-f07930d27e46 GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f)
BUDGET_S=${BUDGET_S:-8000}
# Mangled-name fingerprints of gated_delta_net_cuda<S_v, KDA, keep_rs, NS[, cols_per_warp = 4]>
KERNEL4='gated_delta_net_cudaILi128ELb[01]ELb[01]ELi[124]ELi4E'
KERNEL1='gated_delta_net_cudaILi128ELb[01]ELb[01]ELi[124]EEv'
START=$SECONDS
CUTOFF=$(date '+%Y-%m-%d %H:%M:%S')

die() { local rc=$1; shift; echo "bw-s9-gdn4: FAIL rc=$rc $*"; exit "$rc"; }
preflight() {
    [[ $(git branch --show-current) == topic/s9-gdn4 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor "$BASE_SHA" HEAD || return 1
    [[ $(head -1 "$W/frozen/stack9/SOURCE" | cut -d' ' -f1) == "$BASE_SHA" ]] || return 1
    [[ ! -e $W/frozen/$NAME && ! -e $VAR ]] || return 1
    [[ -z $(ls codex/items/rendered/s9gdn4-*.gpuq.toml 2> /dev/null || true) ]] || return 1
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf flock taskset cuobjdump c++filt nvidia-smi journalctl; do
        command -v "$tool" > /dev/null || return 1
    done
    for p in "$CCCL" "$BUILDGATE" "$CCX01" "$W/bin/wt" /usr/local/cuda-13.2/bin/nvcc "$W/frozen/stack9/build/bin/libggml-cuda.so" "$EVIDENCE"; do
        [[ -e $p ]] || return 1
    done
    # only the kernel and the test harness differ from stack9 (outside codex/)
    [[ $(git diff --name-only "$BASE_SHA" HEAD -- . ':!codex') == $'ggml/src/ggml-cuda/gated_delta_net.cu\ntests/test-backend-ops.cpp' ]] || return 1
    python3 - "$EVIDENCE" <<'PY' || return 1
import hashlib, json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['result'] == 'PASS' and {20, 21, 22} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
h = hashlib.sha256()
for name in r['source_files']:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gates'
assert r['gdn_cases'] >= 70
PY
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-s9-gdn4: PASS preflight'; exit 0; fi
    echo 'bw-s9-gdn4: FAIL preflight'; exit 8
fi
[[ ${GPUQ_ITEM:-} == bw-s9-gdn4 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1")
case $OUT in "$ROOT"/codex/window/s9-gdn4-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || die 8 preflight
[[ $VISIBLE == *"${GPUS[0]}"* && $VISIBLE == *"${GPUS[1]}"* ]] || die 4 "both GPUs must be visible to the item: $VISIBLE"
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
MIN_CASES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gdn_cases"])' "$EVIDENCE")
trap 'git checkout -q -- "$F" "$T"' EXIT

stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-s9-gdn4: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] && return 0
    case $rc in 3 | 4 | 5 | 6 | 7) exit "$rc" ;; *) exit 8 ;; esac
}
guard() {
    if journalctl -k -b --since "$CUTOFF" --no-pager 2> /dev/null | grep -Ec '\bXid\b|fallen off the bus' > /dev/null; then
        die 3 "Xid since $CUTOFF"
    fi
    (( SECONDS - START < BUDGET_S )) || die 7 "budget expired"
}
check_gpu() { # INDEX
    local u=${GPUS[$1]} name mem
    name=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v u="$u" '$1 == u {print $2}')
    [[ -n $name ]] || die 4 "GPU $u missing"
    [[ $name == *"RTX 5060 Ti"* ]] || die 6 "GPU $u is not an RTX 5060 Ti: $name"
    for _ in $(seq 45); do
        mem=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u="$u" '$1 == u {print $2}')
        (( mem <= 512 )) && return 0
        sleep 1
    done
    die 4 "GPU $u memory did not release"
}

server_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
test_flags=("${server_flags[@]/-DLLAMA_BUILD_TESTS=OFF/-DLLAMA_BUILD_TESTS=ON}")

has_symbols() { # LIB PATTERN WANT(0|1)
    local n
    n=$(strings -a "$1" | grep -Ec "$2" || true)
    if [[ $3 == 1 ]]; then (( n > 0 )); else (( n == 0 )); fi
}
is_elf() { [[ -f $1 && ! -L $1 && $(head -c4 "$1" | tail -c3) == ELF ]]; }

build_server() {
    nice -n 19 "$BUILDGATE" cmake -B build-cuda "${server_flags[@]}"
    nice -n 19 "$BUILDGATE" cmake --build build-cuda --target llama-server -j4
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null
    has_symbols build-cuda/bin/libggml-cuda.so "$KERNEL4" 1
    has_symbols build-cuda/bin/libggml-cuda.so "$KERNEL1" 0
    has_symbols "$W/frozen/stack9/build/bin/libggml-cuda.so" "$KERNEL4" 0
    has_symbols "$W/frozen/stack9/build/bin/libggml-cuda.so" "$KERNEL1" 1
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}

build_tests() { nice -n 19 "$BUILDGATE" cmake --build build-test --target test-backend-ops -j4; }
# VARIANT: test-backend-ops and libs of build-test, relocatable (rpath $ORIGIN, SHA256SUMS)
stash() {
    local d=$VAR/$1/bin f
    mkdir -p "$d"
    cp -p build-test/bin/test-backend-ops "$d/"
    for f in build-test/bin/*.so*; do cp -a "$f" "$d/"; done
    for f in "$d"/*; do
        if is_elf "$f"; then patchelf --set-rpath '$ORIGIN' "$f"; fi
    done
    (cd "$d" && find . -maxdepth 1 \( -type f -o -type l \) | sed 's#^\./##' | grep -v '^SHA256SUMS$' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS)
    [[ -z $(ldd "$d/test-backend-ops" | grep 'not found' || true) ]]
    [[ -z $(ldd "$d/test-backend-ops" | awk '/libggml/ {print $3}' | grep -v "^$(cd "$d" && pwd -P)/" || true) ]]
    echo "variant $1: libggml-cuda $(sha256sum "$d/libggml-cuda.so.0.25.3" | cut -c1-16)"
}
# VARIANT KIND: KIND is the source edit applied before the rebuild; the EXIT trap and the next stage restore both files
variant() {
    local v=$1 kind=$2
    git checkout -q -- "$F" "$T"
    case $kind in
        base) git show "$BASE_SHA:$F" > "$F" ;;
        cand) ;;
        mutant-red)
            sed -i 's#warp_reduce_sum<lanes_per_col>(kv_shard)#warp_reduce_sum<warp_size>(kv_shard)#' "$F"
            [[ $(git diff --numstat -- "$F") == $'2\t2\t'"$F" ]] ;;
        mutant-attn)
            sed -i 's#if (lane_in_col == 0) {#if (lane == 0) {#' "$F"
            [[ $(git diff --numstat -- "$F") == $'2\t2\t'"$F" ]] ;;
        oldh)
            # the old perf harness: the fusion cases repeat only their output node (ggml_sum)
            sed -i 's#return { gdn_node, cpy_node };#return { out };#' "$T"
            [[ $(git diff --numstat -- "$T") == $'1\t1\t'"$T" ]] ;;
        *) return 1 ;;
    esac
    git diff -U0 -- "$F" "$T" > "$OUT/variant-$v.diff"
    build_tests
    case $kind in
        base) has_symbols build-test/bin/libggml-cuda.so "$KERNEL4" 0; has_symbols build-test/bin/libggml-cuda.so "$KERNEL1" 1 ;;
        *) has_symbols build-test/bin/libggml-cuda.so "$KERNEL4" 1 ;;
    esac
    stash "$v"
    git checkout -q -- "$F" "$T"
}
variants() {
    "$W/bin/wt" work s9-gdn4-variants --purpose "test-backend-ops variants for bw-s9-gdn4: base kernel, candidate, two kernel mutants, old perf harness" > /dev/null
    nice -n 19 "$BUILDGATE" cmake -B build-test "${test_flags[@]}"
    for spec in base:base cand:cand mutant-red:mutant-red mutant-attn:mutant-attn oldh:oldh; do
        variant "${spec%%:*}" "${spec##*:}"
    done
    [[ -z $(git status --porcelain --untracked-files=no) ]]
    # every variant is a different binary set
    [[ $(sha256sum "$VAR"/*/bin/libggml-cuda.so.0.25.3 | awk '{print $1}' | sort | uniq | wc -l) == 4 ]]  # oldh shares the candidate kernel
    [[ $(sha256sum "$VAR"/cand/bin/libggml-cuda.so.0.25.3 | cut -d' ' -f1) == $(sha256sum "$VAR"/oldh/bin/libggml-cuda.so.0.25.3 | cut -d' ' -f1) ]]
    [[ $(sha256sum "$VAR"/cand/bin/test-backend-ops | cut -d' ' -f1) != $(sha256sum "$VAR"/oldh/bin/test-backend-ops | cut -d' ' -f1) ]]
}

RC=0
WRAP=()
# tbo TAG VARIANT GPU_INDEX [ENV=VAL...] -- ARGS: one test-backend-ops run on one GPU as CUDA0; RC holds its status
tbo() {
    local tag=$1 v=$2 gpu=${GPUS[$3]} extra=()
    shift 3
    while [[ $1 != -- ]]; do extra+=("$1"); shift; done
    shift
    guard
    RC=0
    "${WRAP[@]}" env CUDA_VISIBLE_DEVICES="$gpu" "${extra[@]}" timeout 900 nice -n 5 "$VAR/$v/bin/test-backend-ops" -b CUDA0 "$@" > "$OUT/$tag.log" 2>&1 || RC=$?
    (( RC != 124 )) || die 7 "$tag hung (900 s)"
    printf '%s\t%s\n' "$tag" "$RC" >> "$OUT/runs.tsv"
}
passed() { [[ $RC == 0 ]] && python3 -P "$H/check_ops.py" passed "$OUT/$1.log" "$MIN_CASES" > "$OUT/$1.check" 2>&1; }
OPS='GATED_DELTA_NET.*'
correctness() {
    local g
    for g in 0 1; do
        check_gpu "$g"
        tbo "cand-gpu$g" cand "$g" -- test -o "$OPS"
        passed "cand-gpu$g" || die 8 "candidate fails on GPU $g (see $OUT/cand-gpu$g.log, $(cat "$OUT/cand-gpu$g.check" 2> /dev/null || true))"
    done
    # the new cases and the attention-score comparison also pass on the stack9 kernel: they are not a gate that only the candidate meets
    tbo base-gpu0 base 0 -- test -o "$OPS"
    passed base-gpu0 || die 8 "stack9 kernel fails the same cases (see $OUT/base-gpu0.log): the case list is not a valid gate"
    cat "$OUT"/cand-gpu0.check "$OUT"/cand-gpu1.check "$OUT"/base-gpu0.check
    # negative controls: a column-mixing reduction and a missing attention-score write must fail the same cases
    local m
    for m in mutant-red mutant-attn; do
        tbo "$m-gpu0" "$m" 0 -- test -o "$OPS"
        if passed "$m-gpu0"; then die 5 "$m passed all cases; the tolerance is not a gate for this lever"; fi
        (( RC != 0 )) || die 5 "$m: no failing case, rc 0"
        python3 -P "$H/check_ops.py" failed "$OUT/$m-gpu0.log" 'GATED_DELTA_NET_INDEXED_CACHE_FUSION' || die 5 "$m not caught by the indexed fused cases (the joint path)"
        python3 -P "$H/check_ops.py" failed "$OUT/$m-gpu0.log" 'GATED_DELTA_NET\(' || die 5 "$m not caught by the plain cases"
    done
    # engagement: the stack9 racy-layout mutant logs only when the fused indexed multi-sequence path ran, i.e. the joint block layout
    tbo joint-engaged cand 0 GGML_CUDA_GDN_JOINT_MUTANT=1 -- test -o 'GATED_DELTA_NET_INDEXED_CACHE_FUSION' -p 'head_count=24,'
    grep -c 'gdn_joint_mutant: engaged' "$OUT/joint-engaged.log" > /dev/null || die 8 "no gdn_joint_mutant line: the joint path was not exercised"
    echo "joint-layout mutant rc=$RC (informational: a race escaping on one run is luck, not a verdict)"
}

resources() {
    python3 -P "$H/resource_usage.py" build-cuda/bin/libggml-cuda.so "$W/frozen/stack9/build/bin/libggml-cuda.so" > "$OUT/resource.tsv"
    cat "$OUT/resource.tsv"
}

perf_runs() {
    local g r arm order
    # timing runs keep the harness on CCX0+1 like the queue's measure items
    WRAP=(bash "$CCX01")
    for g in 0 1; do
        check_gpu "$g"
        for r in 1 2 3 4; do
            if (( r == 1 || r == 4 )); then order=(base cand); else order=(cand base); fi
            for arm in "${order[@]}"; do
                tbo "perfA-gpu$g-$arm-$r" "$arm" "$g" -- perf -o "$OPS" -p 'head_count=24,'
                [[ $RC == 0 ]] || die 8 "perf A $arm gpu$g round $r"
                tbo "perfB-gpu$g-$arm-$r" "$arm" "$g" -- perf -o GATED_DELTA_NET -p 'head_count=32,'
                [[ $RC == 0 ]] || die 8 "perf B $arm gpu$g round $r"
            done
        done
    done
    # control: the old harness times ggml_sum in the fusion rows, so its rows must sit well below the new ones
    check_gpu 0
    for r in 1 2; do
        tbo "perfC-gpu0-oldh-$r" oldh 0 -- perf -o 'GATED_DELTA_NET_CACHE_FUSION,GATED_DELTA_NET_INDEXED_CACHE_FUSION' -p 'head_count=24,'
        [[ $RC == 0 ]] || die 8 "perf C oldh round $r"
    done
    python3 -P "$H/perf_summary.py" "$OUT" > "$OUT/perf.tsv"
    cat "$OUT/perf.tsv"
}
freeze() {
    git checkout -q -- "$F" "$T"
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null
    nice -n 19 "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda --desc 'stack9 cd776e104 + #30087 (GDN four state columns per warp, joint multi-sequence layout kept); CUDA 13.2 120a CCCL343 NCCL OFF; round-trace = cp of llama-server'
    [[ $(head -1 "$W/frozen/$NAME/SOURCE") == "$SHA" ]]
    {
        printf 'commit %s (base stack9 %s)\nconfigure: cmake -B build-cuda ' "$SHA" "$BASE_SHA"
        printf '%q ' "${server_flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\nvariants: %s/{base,cand,mutant-red,mutant-attn,oldh}/bin from build-test (tests on)\n' "$VAR"
        printf 'CUDA_VISIBLE_DEVICES="" for the build; test-backend-ops ran on both RTX 5060 Ti; no model run\n'
    } > "$W/frozen/$NAME/RECIPE.txt"
}

items() {
    mkdir -p codex/items/rendered
    nice -n 19 python3 -P "$H/render_items.py" --out "$ROOT/codex/items/rendered"
}

stage 40 cuda-build build_server
stage 41 cuda-stamps stamps
stage 42 test-variants variants
stage 43 cuda-correctness correctness
stage 44 resource-usage resources
stage 45 kernel-perf perf_runs
stage 46 freeze-stack9-gdn4 freeze
stage 47 render-and-check-items items
python3 - "$OUT" "$SHA" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p / 'summary.json').write_text(json.dumps({'commit': sys.argv[2], 'result': 'PASS', 'GPU_execution': True, 'frozen': 'frozen/stack9-gdn4',
                                           'perf': 'perf.tsv (not a gate)', 'resource': 'resource.tsv', 'items': 'codex/items/rendered/s9gdn4-*.gpuq.toml; checked, NOT submitted'}, indent=2) + '\n')
PY
echo 'bw-s9-gdn4: PASS; frozen/stack9-gdn4 built, items checked, NOTHING submitted'
