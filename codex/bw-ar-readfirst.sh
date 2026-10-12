#!/usr/bin/env bash
# ar-readfirst window (gpuq build window, never submits). AllReduce chunked kernel with the loads of round r-1 issued before the
# stores of round r, on master abee0c847. Adapted from wt/ar-duplex codex/bw-ar-duplex.sh. Stages:
#   40 CPU protocol model: master, the general read-first commit, the local commit in read-first and store-first order (+ mutants)
#   41 CUDA build of this tree (vanilla recipe = frozen/pkg89-base's), llama-server + test-llama-archs, on /mnt/scratch
#   42 stamps (base arm = frozen/pkg89-base libs: master abee0c847, same recipe; checksums verified)
#   43 driver build + CPU selftest
#   44 byte identity vs master on both GPUs (arms rf sf one rv2 x default/F32 wire/copy engine off), stalled rank 1, no-wait mutants
#   45 compute-sanitizer synccheck/racecheck, base vs rf (same rc)
#   46 TP identity: test-llama-archs Meta(CUDA0,CUDA1) -a qwen35 and llama, this build vs the same binary on master ggml libs:
#      rc 0 and identical Meta rows
#   47 per-call perf, ABBA (base rf sf one rv2), 240 calls per cell and batch
#   48 GATE: rf vs base >= 10 % lower at rows6 and rows12 and no decode cell > +2 % (both batches). FAIL ends the window here with
#      rc 0 and summary result GATE_FAIL: no freeze, no items.
#   49 freeze frozen/ar-readfirst, 50 render + `gpuq submit --check` arp-np1-identity-j5q and arp-np2-real-j5q (never submitted)
# test-backend-ops has no case that reaches the AllReduce (no comm/allreduce op in tests/test-backend-ops.cpp), so it is not run.
# rc: 0 pass or GATE_FAIL | 5 negative control void | 6 integrity | 8 a correctness gate fails
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10 GGML_CCACHE=ON
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda-13.2
export TMPDIR=/mnt/scratch/tmp-build
ID=bw-ar-readfirst
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BRANCH=topic/ar-readfirst
BASE=abee0c8476b585b0d081d580b5ce21d0addc22c4
DUP1=10c3c0edf   # cherry-pick of ar-duplex f9d0748b6 (store r / load r-1 rounds)
DUP2=d07d13f5b   # cherry-pick of ar-duplex af3762817 (thread-0 release after the barrier, 64-bit tokens)
RF=c04a38e96     # general: loads of round r-1 before the stores of round r
LOCAL=a6f51a7d4  # local knobs: ROUND_VECS, STORE_FIRST, MUTANT, STALL_US
SCR=/mnt/scratch/work/ar-readfirst-window
BUILD=$SCR/build-cuda
NEWLIB=$BUILD/bin
BASE_ROOT=$W/frozen/pkg89-base
BASELIB=$BASE_ROOT/build/bin
RECIPE=$W/frozen/vanilla-08246a28f/RECIPE.txt
MAC_GATE=$W/work/build-window-1007/ar-readfirst-mac-a2
DRV=$SCR/ar-duplex-driver
SAN=/usr/local/cuda-13.2/bin/compute-sanitizer
FROZEN=ar-readfirst
LIGHT=(nice -n 19 taskset -c 6-11,18-23)
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
targets=(llama-server test-llama-archs)
configure_line() { printf 'configure: cmake -S . -B build-cuda '; printf '%q ' "${flags[@]}"; }

preflight() {
    [[ $(git branch --show-current) == "$BRANCH" ]] || return 6
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 6
    [[ $(git rev-parse "$DUP1^") == "$BASE" && $(git rev-parse "$DUP2^") == $(git rev-parse "$DUP1") ]] || return 6
    [[ $(git rev-parse "$RF^") == $(git rev-parse "$DUP2") && $(git rev-parse "$LOCAL^") == $(git rev-parse "$RF") ]] || return 6
    git merge-base --is-ancestor "$LOCAL" HEAD || return 6
    [[ $(git diff --name-only "$BASE" HEAD -- . ':!codex' | tr '\n' ' ') == 'ggml/src/ggml-cuda/allreduce.cu ' ]] || return 6
    [[ -z $(git diff --name-only "$LOCAL" HEAD -- . ':!codex') ]] || return 6
    [[ ! -e $W/frozen/$FROZEN ]] || return 6
    if git show-ref --verify --quiet "refs/tags/frozen/$FROZEN"; then return 6; fi
    [[ $(sed -n '1p' "$BASE_ROOT/SOURCE") == "$BASE" && -f $BASE_ROOT/STAMPS.txt && -x $BASE_ROOT/build/bin/round-trace ]] || return 6
    [[ $(configure_line) == "$(grep -a '^configure: ' "$RECIPE")" ]] || return 6
    [[ $(configure_line) == "$(grep -a '^configure: ' "$BASE_ROOT/RECIPE.txt")" ]] || return 6
    local tool p
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf taskset g++ tar ldd sha256sum; do
        command -v "$tool" > /dev/null || return 6
    done
    for p in "$CCCL" "$W/bin/buildgate" "$W/bin/wt" "$W/bin/cpugate" /usr/local/cuda-13.2/bin/nvcc "$SAN" "$RECIPE" \
             codex/ar-duplex-driver.cpp codex/ar-cpumodel.cpp codex/ar-cpumodel.sh codex/ar_summary.py codex/render_arp.py \
             codex/postrun_proofs.sh; do
        [[ -e $p ]] || return 6
    done
    [[ -d $TMPDIR && -w $TMPDIR && -w /mnt/scratch ]] || return 6
    # the Mac CPU gate passed on a commit with this exact kernel source and these model files
    local mac_sha
    mac_sha=$(sed -n 's/^HEAD=//p' "$MAC_GATE/mac-run.txt")
    [[ $(sed -n 's/^EXIT_CODE=//p' "$MAC_GATE/mac-run.txt") == 0 ]] || return 6
    grep -aq '^mac-gate: PASS' "$MAC_GATE/payload.log" || return 6
    [[ -z $(git diff --name-only "$mac_sha" HEAD -- ggml codex/ar-cpumodel.cpp codex/ar-cpumodel.sh codex/ar_summary.py) ]] || return 6
    for p in codex/bw-ar-readfirst.sh codex/ar-cpumodel.sh codex/postrun_proofs.sh; do bash -n "$p" || return 6; done
    python3 -c "
import sys
for f in sys.argv[1:]: compile(open(f).read(), f, 'exec')" codex/ar_summary.py codex/render_arp.py || return 6
}
selftest() {
    "${LIGHT[@]}" python3 codex/ar_summary.py --selftest > /dev/null || return 8
    bash codex/postrun_proofs.sh --selftest > /dev/null || return 8
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo "$ID: FAIL preflight"; exit 6; }
    echo "$ID: PASS preflight"; exit 0
fi
if [[ ${1:-} == --selftest ]]; then
    selftest || { echo "$ID: FAIL selftest"; exit 8; }
    echo "$ID: PASS selftest (perf gate good/bad/incomplete, postrun proofs)"; exit 0
fi
[[ ${GPUQ_ITEM:-} == "$ID" && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT" "$SCR"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || { echo "$ID: FAIL preflight"; exit 6; }
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
stage() {  # CODE NAME CMD... ; a failure ends the window
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    (set -e; "$@") > "$OUT/$name.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "$ID: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] && return 0
    echo "$ID: FAIL $name"
    case $rc in 5|6) exit "$rc" ;; *) exit 8 ;; esac
}
summary() {  # RESULT [ITEMS...]
    python3 - "$OUT" "$SHA" "$@" <<'PY_RESULT'
import json, sys
from pathlib import Path
out, sha, result, *items = sys.argv[1:]
lines = (Path(out) / 'perf.tsv').read_text().splitlines() if (Path(out) / 'perf.tsv').exists() else []
keep = [l for l in lines if l.startswith(('gate\t', 'falsifier\t', 'mechanism\t'))]
(Path(out) / 'summary.json').write_text(json.dumps(dict(result=result, commit=sha, GPU_execution=True, perf='perf.tsv',
    perf_gates=keep, frozen=[] if result != 'PASS' else ['ar-readfirst'], items_checked_not_submitted=items), indent=2) + '\n')
PY_RESULT
}

cpu_model() {
    local rev mode order tag
    # master, the general read-first commit, the local commit in both orders: the model passes, and with the wait compiled out it fails
    for tag in base:"$BASE":rf rf:"$RF":rf local-rf:"$LOCAL":rf local-sf:"$LOCAL":sf; do
        IFS=: read -r name rev order <<< "$tag"
        for mode in quick mutated; do
            AR_MODEL_OUT=$(mktemp -d "$SCR/model-XXXXXX") bash codex/ar-cpumodel.sh "$rev" "$mode" "$order" > "$OUT/cpumodel-$name-$mode.out" 2>&1 \
                || { echo "FAIL cpumodel $name $mode"; return 8; }
            grep -aq '^cpumodel' "$OUT/cpumodel-$name-$mode.out" || { echo "FAIL cpumodel $name $mode printed no verdict"; return 6; }
        done
    done
}
build() {
    [[ ! -e $BUILD ]] || return 6   # a stale build dir from an earlier attempt would mix configurations
    "${LIGHT[@]}" "$W/bin/buildgate" cmake -S "$ROOT" -B "$BUILD" "${flags[@]}"
    "${LIGHT[@]}" "$W/bin/buildgate" cmake --build "$BUILD" -j4 --target "${targets[@]}"
    cp "$BUILD/bin/llama-server" "$BUILD/bin/round-trace"
}
stamps() {
    (cd "$BASELIB" && sha256sum -c --quiet SHA256SUMS) || return 6
    [[ $(grep -ac 'GGML_CUDA_AR_STORE_FIRST' "$NEWLIB/libggml-cuda.so") -ge 1 ]] || return 6
    [[ $(grep -ac 'GGML_CUDA_AR_STORE_FIRST' "$BASELIB/libggml-cuda.so") == 0 ]] || return 6
    [[ $(grep -ac 'GGML_CUDA_AR_COPY_THRESHOLD' "$BASELIB/libggml-cuda.so") -ge 1 ]] || return 6
    strings "$NEWLIB/libggml-base.so" | grep -x "$SHORT" > /dev/null || return 6
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    grep -ax 'GGML_CUDA_NCCL:BOOL=OFF' "$BUILD/CMakeCache.txt" || return 6
    grep -aEx 'CMAKE_CUDA_ARCHITECTURES:(STRING|UNINITIALIZED)=120a' "$BUILD/CMakeCache.txt" || return 6
    cp "$BUILD/CMakeCache.txt" "$OUT/CMakeCache-rf.txt"
    printf 'source_M=%s\nbase_D=%s (upstream master)\nCUDA_D=13.2\nCCCL_D=3.4.3\narch_D=120a\nNCCL_D=OFF\n' "$SHA" "$BASE" > "$OUT/STAMPS-rf.txt"
    # SASS of the read-first kernel, for review: in each instantiation the peer loads (LDG.E.128) precede the host stores
    # (STG.E.128) of the same loop iteration. Informational, not a gate.
    cuobjdump -sass -fun ggml_cuda_ar_kernel "$NEWLIB/libggml-cuda.so" > "$OUT/sass-ar-kernel.txt" 2>&1 || true
}
driver() {
    "${LIGHT[@]}" g++ -std=c++17 -O2 -Wall -Wextra -I ggml/include codex/ar-duplex-driver.cpp -o "$DRV" -L "$NEWLIB" -lggml-base -lggml-cuda -lggml
    env LD_LIBRARY_PATH="$NEWLIB" CUDA_VISIBLE_DEVICES='' "$DRV" selftest
}

RC=0
drv() { # TAG LIBDIR MODE-ARGS... -- [ENV=VAL ...]: stdout -> $OUT/TAG.out, stderr -> $OUT/TAG.err, status in RC
    local tag=$1 lib=$2
    shift 2
    local -a args=() envs=()
    while [[ $# -gt 0 && $1 != -- ]]; do args+=("$1"); shift; done
    [[ ${1:-} == -- ]] && shift
    envs=("$@")
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" "${envs[@]}" timeout 900 nice -n 19 "$DRV" "${args[@]}" > "$OUT/$tag.out" 2> "$OUT/$tag.err" || RC=$?
    printf '%s\t%s\n' "$tag" "$RC" >> "$OUT/runs.tsv"
}
need_ok() { # TAG: the run passed and took the internal AllReduce path
    [[ $RC == 0 ]] || { echo "FAIL $1 rc=$RC"; return 8; }
    grep -aq 'initialized AllReduce pipeline: 2 GPUs' "$OUT/$1.err" || { echo "FAIL $1 did not initialise the internal AllReduce"; return 6; }
}
arm_env() { # ARM -> knob env of that arm (base has none)
    case $1 in
        base|rf) echo LC_ALL=C ;;
        sf)      echo GGML_CUDA_AR_STORE_FIRST=1 ;;
        one)     echo GGML_CUDA_AR_ROUND_VECS=0 ;;
        rv2)     echo GGML_CUDA_AR_ROUND_VECS=2 ;;
    esac
}
identity() {
    [[ $VISIBLE == GPU-*,GPU-* ]] || return 6
    local variant venv tag
    # variant = extra environment of the whole pass: default, pure F32 wire, copy engine off (sizes above 1 MB take the kernel path)
    for variant in default f32wire nocopy; do
        case $variant in
            default) venv=(LC_ALL=C) ;;
            f32wire) venv=(GGML_CUDA_AR_BF16_THRESHOLD=0) ;;
            nocopy)  venv=(GGML_CUDA_AR_COPY_THRESHOLD=0) ;;
        esac
        drv "base-$variant" "$BASELIB" identity -- "${venv[@]}"
        need_ok "base-$variant" || return $?
        [[ $(grep -ac 'ref=BAD' "$OUT/base-$variant.out") == 0 ]] || { echo "FAIL base-$variant disagrees with the CPU reference: driver defect"; return 6; }
        [[ $(grep -ac '^id ' "$OUT/base-$variant.out") -ge 30 ]] || { echo "FAIL base-$variant ran too few cases"; return 6; }
        for tag in rf sf one rv2; do
            drv "$tag-$variant" "$NEWLIB" identity -- "${venv[@]}" "$(arm_env "$tag")"
            need_ok "$tag-$variant" || return $?
            cmp "$OUT/base-$variant.out" "$OUT/$tag-$variant.out" || { echo "FAIL $tag-$variant output hashes differ from master"; return 8; }
        done
    done
    # slow rank 1 with the correct wait: still identical (the wait orders the exchange)
    for tag in rf sf; do
        drv "stall-$tag" "$NEWLIB" identity -- GGML_CUDA_AR_STALL_US=100 "$(arm_env "$tag")"
        need_ok "stall-$tag" || return $?
        cmp "$OUT/base-default.out" "$OUT/stall-$tag.out" || { echo "FAIL stalled rank 1 changes the result ($tag)"; return 8; }
    done
    # Negative controls: the same slow rank 1 without the wait must fail the reference check and differ from master.
    for tag in rf sf one; do
        drv "mutant-$tag" "$NEWLIB" identity -- GGML_CUDA_AR_MUTANT=1 GGML_CUDA_AR_STALL_US=100 "$(arm_env "$tag")"
        [[ $RC == 1 ]] || { echo "negative control void: no-wait mutant ($tag) passed or crashed (rc=$RC)"; return 5; }
        grep -aq 'ar-duplex: FAIL' "$OUT/mutant-$tag.err" || { echo "negative control void: no FAIL line ($tag)"; return 5; }
        ! cmp -s "$OUT/base-default.out" "$OUT/mutant-$tag.out" || { echo "negative control void: mutant hashes equal master ($tag)"; return 5; }
    done
    # Informational: the unstalled no-wait mutant is a race that may or may not fire.
    drv mutant-raw-rf "$NEWLIB" identity -- GGML_CUDA_AR_MUTANT=1
}
sanitizers() {
    [[ $VISIBLE == GPU-*,GPU-* ]] || return 6
    local tool arm b n lib
    # Two kernels spin on each other: if compute-sanitizer serializes launches both arms time out and the stage records a void, never a pass.
    for tool in synccheck racecheck; do
        for arm in base rf; do
            RC=0
            lib=$NEWLIB; [[ $arm == base ]] && lib=$BASELIB
            env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" timeout 900 nice -n 19 "$SAN" --tool "$tool" --error-exitcode 99 "$DRV" identity small \
                > "$OUT/san-$tool-$arm.out" 2> "$OUT/san-$tool-$arm.err" || RC=$?
            printf '%s\t%s\t%s\n' "$tool" "$arm" "$RC" >> "$OUT/sanitizers.tsv"
        done
        b=$(awk -v t="$tool" '$1 == t && $2 == "base" {print $3}' "$OUT/sanitizers.tsv")
        n=$(awk -v t="$tool" '$1 == t && $2 == "rf" {print $3}' "$OUT/sanitizers.tsv")
        echo "$tool: base rc=$b rf rc=$n"
        [[ $n == "$b" ]] || { echo "FAIL $tool: rf rc $n differs from base rc $b"; return 8; }
        [[ $n == 0 ]] || echo "NOTE $tool: base and rf both rc=$n (124 = timeout: the sanitizer serializes the two spinning kernels; void, not a pass)"
    done
}
meta_rows() {  # LOG -> the Meta rows of the result table, colour codes stripped
    sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -a '^|.*|.*Meta.*|' || true
}
tp_identity() {
    [[ $VISIBLE == GPU-*,GPU-* ]] || return 6
    local arch tag lib rc stray
    # the base run must resolve every ggml/llama library from the master build
    stray=$(env LD_LIBRARY_PATH="$BASELIB" ldd "$NEWLIB/test-llama-archs" | awk '/libggml|libllama/ {print $3}' | grep -v "^$BASELIB/" || true)
    [[ -z $stray ]] || { echo "base libraries resolve outside $BASELIB: $stray"; return 6; }
    for arch in qwen35 llama; do
        for tag in rf base; do
            lib=$NEWLIB; [[ $tag == base ]] && lib=$BASELIB
            rc=0
            env CUDA_VISIBLE_DEVICES="$VISIBLE" CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH="$lib" timeout 900 nice -n 19 \
                "$NEWLIB/test-llama-archs" -a "^$arch\$" -s 42 > "$OUT/tp-$arch-$tag.log" 2>&1 || rc=$?
            echo "test-llama-archs $arch $tag rc=$rc"
            [[ $rc == 0 ]] || { echo "FAIL test-llama-archs $arch $tag rc=$rc"; return 8; }
            meta_rows "$OUT/tp-$arch-$tag.log" > "$OUT/tp-$arch-$tag.meta"
            [[ $(grep -c 'OK' "$OUT/tp-$arch-$tag.meta") -ge 1 ]] || { echo "FAIL no passing Meta row ($arch $tag)"; return 6; }
        done
        cmp "$OUT/tp-$arch-base.meta" "$OUT/tp-$arch-rf.meta" || { echo "FAIL Meta rows differ from master ($arch)"; return 8; }
        cat "$OUT/tp-$arch-rf.meta"
    done
}
perf() {
    [[ $VISIBLE == GPU-*,GPU-* ]] || return 6
    local i=0 arm lib
    # ABBA blocks; every arm is a fresh process (the knobs are read at pipeline init), 240 calls per cell and batch.
    for arm in base rf rf base one sf sf one base rf rf base rv2 one sf rv2; do
        i=$((i+1))
        lib=$NEWLIB; [[ $arm == base ]] && lib=$BASELIB
        drv "perf-$i-$arm" "$lib" perf "$arm" 240 -- "$(arm_env "$arm")"
        [[ $RC == 0 ]] || { echo "FAIL perf $arm rc=$RC"; return 8; }
    done
    python3 codex/ar_summary.py "$OUT" > "$OUT/perf.tsv"
    grep -a -E '^(gate|falsifier|mechanism)' "$OUT/perf.tsv"
}
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    "${LIGHT[@]}" "$W/bin/wt" freeze "$ROOT" "$FROZEN" --build "$BUILD" --desc \
        "master abee0c847 + AllReduce chunked kernel read-first pipeline ($RF) + local knobs ($LOCAL, defaults = general path); CUDA 13.2 120a CCCL343 NCCL OFF; vanilla recipe; rf arm of the arp-* A/B items" || return 6
    [[ $(sed -n '1p' "$W/frozen/$FROZEN/SOURCE") == "$SHA" ]] || return 6
    {
        printf 'commit %s\n' "$SHA"
        configure_line
        printf '\nbuild dir: %s\nbuild: bin/buildgate cmake --build build-cuda -j4 --target %s\n' "$BUILD" "${targets[*]}"
        printf 'base: %s (upstream master)\nCPU gate: %s\nGPU gate: window %s, logs in %s\ntag: refs/tags/frozen/%s\n' \
            "$BASE" "$MAC_GATE/stages.tsv" "$ID" "$OUT" "$FROZEN"
    } > "$W/frozen/$FROZEN/RECIPE.txt" || return 6
    cp "$OUT/STAMPS-rf.txt" "$W/frozen/$FROZEN/STAMPS.txt" || return 6
    git tag "frozen/$FROZEN" "$SHA" || return 6
}
items() { "${LIGHT[@]}" python3 codex/render_arp.py --perf "$OUT/perf.tsv" --window "$OUT"; }

stage 40 cpu-model cpu_model
stage 41 cuda-build build
stage 42 stamps stamps
stage 43 driver driver
stage 44 identity-mutants identity
stage 45 sanitizers sanitizers
stage 46 tp-identity tp_identity
stage 47 perf-arms perf
GATE_RC=0
python3 codex/ar_summary.py --gate "$OUT/perf.tsv" || GATE_RC=$?
printf '48\tgate\t%s\t0\n' "$GATE_RC" >> "$OUT/stages.tsv"
case $GATE_RC in
    0) echo "$ID: gate PASS" ;;
    1) summary GATE_FAIL; echo "$ID: GATE_FAIL (micro-benchmark gate not met: no freeze, no items; see perf.tsv)"; exit 0 ;;
    *) echo "$ID: FAIL gate table incomplete"; exit 6 ;;
esac
stage 49 freeze freeze
stage 50 render-and-check items
summary PASS codex/items/rendered/arp-np1-identity-j5q.gpuq.toml codex/items/rendered/arp-np2-real-j5q.gpuq.toml
echo "$ID: PASS"
