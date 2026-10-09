#!/usr/bin/env bash
# ar-duplex window: CPU protocol model, CUDA builds of this tree and of the base (stack10 HEAD, allreduce.cu only differs), byte identity of
# the chunked AllReduce kernel against the base on both RTX 5060 Ti (incl. the wait-removal mutant), compute-sanitizer, per-call perf arms
# (ABBA), freeze frozen/ar-duplex, render one np2 real J5q item (gpuq submit --check only). No server, no model.
# rc: 0 pass | 5 negative control void | 6 integrity | 8 a correctness gate fails
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BASE=f163659410ac18610bfa09b506ddbabd40b9e17d
GENERAL=f9d0748b6
SCR=/mnt/scratch/ar-duplex
NEWLIB=$ROOT/build-cuda/bin
BASELIB=$SCR/base/build/bin
DRV=$SCR/ar-duplex-driver
SAN=/usr/local/cuda-13.2/bin/compute-sanitizer

preflight() {
    [[ $(git branch --show-current) == topic/ar-duplex ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    [[ $(git rev-parse "$GENERAL^") == "$BASE" ]] || return 1
    [[ $(git diff --name-only "$BASE" HEAD -- . ':!codex' | tr '\n' ' ') == 'ggml/src/ggml-cuda/allreduce.cu ' ]] || return 1
    [[ ! -e $W/frozen/ar-duplex ]] || return 1
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf taskset g++ tar; do
        command -v "$tool" > /dev/null || return 1
    done
    for p in "$CCCL" "$W/bin/buildgate" "$W/bin/wt" /usr/local/cuda-13.2/bin/nvcc "$SAN" "$EVAL/harness/mkitem.py" /mnt/scratch \
             codex/ar-duplex-driver.cpp codex/ar-duplex-cpumodel.cpp codex/ar-duplex-cpumodel.sh codex/ar_duplex_summary.py \
             codex/render_ardup_items.py codex/items/ardup-np2-real-j5q.toml.in codex/gpu_ab.sh; do
        [[ -e $p ]] || return 1
    done
    [[ -w /mnt/scratch ]] || return 1
    for script in codex/bw-ar-duplex.sh codex/ar-duplex-cpumodel.sh codex/gpu_ab.sh; do bash -n "$script" || return 1; done
    python3 - <<'PY_GATE' || return 1
import tomllib
from pathlib import Path
for name in ('codex/ar_duplex_summary.py', 'codex/render_ardup_items.py'):
    compile(Path(name).read_text(), name, 'exec')
tomllib.loads(Path('codex/items/ardup-np2-real-j5q.toml.in').read_text().split('[model]')[0].replace('@', ''))
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-ar-duplex: FAIL preflight'; exit 8; }
    echo 'bw-ar-duplex: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-ar-duplex && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/ardup-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT" "$SCR"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-ar-duplex: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] && return 0
    [[ $rc == 5 || $rc == 6 ]] && exit "$rc"
    echo "bw-ar-duplex: FAIL $name"
    exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)

cpu_model() {
    local rev mode
    # base and the two commits of this tree: the protocol model passes, and with the wait compiled out it must fail (negative control)
    for rev in "$BASE" "$GENERAL" "$SHA"; do
        for mode in quick mutated; do
            AR_MODEL_OUT=$(mktemp -d "$SCR/model-XXXXXX") bash codex/ar-duplex-cpumodel.sh "$rev" "$mode" > "$OUT/cpumodel-$rev-$mode.out" 2>&1 \
                || { echo "FAIL cpumodel $rev $mode"; return 8; }
            grep -aq '^cpumodel' "$OUT/cpumodel-$rev-$mode.out" || { echo "FAIL cpumodel $rev $mode printed no verdict"; return 6; }
        done
    done
}
build() {
    nice -n 19 "$W/bin/buildgate" cmake -S . -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target llama-server -j4
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
build_base() {
    rm -rf "$SCR/base"
    mkdir -p "$SCR/base/src"
    git archive "$BASE" | tar -x -C "$SCR/base/src"
    nice -n 19 "$W/bin/buildgate" cmake -S "$SCR/base/src" -B "$SCR/base/build" "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build "$SCR/base/build" --target ggml ggml-base ggml-cpu ggml-cuda -j4
}
stamps() {
    [[ $(grep -ac 'GGML_CUDA_AR_ROUND_VECS' "$NEWLIB/libggml-cuda.so") -ge 1 ]]
    [[ $(grep -ac 'GGML_CUDA_AR_ROUND_VECS' "$BASELIB/libggml-cuda.so") == 0 ]]
    [[ $(grep -ac 'GGML_CUDA_AR_COPY_THRESHOLD' "$BASELIB/libggml-cuda.so") -ge 1 ]]
    strings "$NEWLIB/libggml-base.so" | grep -x "$SHORT" > /dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    cmp -s "$NEWLIB/libggml-cuda.so" "$BASELIB/libggml-cuda.so" && { echo 'new and base libggml-cuda.so are identical'; return 6; }
    return 0
}
driver() {
    nice -n 19 g++ -std=c++17 -O2 -Wall -Wextra -I ggml/include codex/ar-duplex-driver.cpp -o "$DRV" -L "$NEWLIB" -lggml-base -lggml-cuda -lggml
    # CPU only: the reference checker flags a flipped bit
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
need_ok() { # TAG: the run passed and, on a new-code arm, took the internal AllReduce path
    [[ $RC == 0 ]] || { echo "FAIL $1 rc=$RC"; return 8; }
    grep -aq 'initialized AllReduce pipeline: 2 GPUs' "$OUT/$1.err" || { echo "FAIL $1 did not initialise the internal AllReduce"; return 6; }
}
identity() {
    [[ $VISIBLE == GPU-* ]] || return 6
    local variant venv tag
    # variant = extra environment of the whole pass: default, pure F32 wire, copy engine off (sizes above 1 MB take the chunked kernel path)
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
        # gen = the general commit's kernel (256 threads, one vector per thread per round); one = single round (pipelining off);
        # loc = the local default (auto threads); the rest is the block size / round size sweep
        for tag in gen one loc t64 t128 rv2; do
            case $tag in
                gen)  drv "$tag-$variant" "$NEWLIB" identity -- "${venv[@]}" GGML_CUDA_AR_THREADS=256 GGML_CUDA_AR_ROUND_VECS=1 ;;
                one)  drv "$tag-$variant" "$NEWLIB" identity -- "${venv[@]}" GGML_CUDA_AR_ROUND_VECS=0 ;;
                loc)  drv "$tag-$variant" "$NEWLIB" identity -- "${venv[@]}" ;;
                t64)  drv "$tag-$variant" "$NEWLIB" identity -- "${venv[@]}" GGML_CUDA_AR_THREADS=64 ;;
                t128) drv "$tag-$variant" "$NEWLIB" identity -- "${venv[@]}" GGML_CUDA_AR_THREADS=128 ;;
                rv2)  drv "$tag-$variant" "$NEWLIB" identity -- "${venv[@]}" GGML_CUDA_AR_THREADS=256 GGML_CUDA_AR_ROUND_VECS=2 ;;
            esac
            need_ok "$tag-$variant" || return $?
            cmp "$OUT/base-$variant.out" "$OUT/$tag-$variant.out" || { echo "FAIL $tag-$variant output hashes differ from the base"; return 8; }
        done
    done
    # slow rank 1 with the correct wait: still identical (the wait orders the exchange)
    drv stall-gen "$NEWLIB" identity -- GGML_CUDA_AR_STALL_US=100 GGML_CUDA_AR_THREADS=256 GGML_CUDA_AR_ROUND_VECS=1
    need_ok stall-gen || return $?
    cmp "$OUT/base-default.out" "$OUT/stall-gen.out" || { echo "FAIL stalled rank 1 changes the result (gen)"; return 8; }
    drv stall-loc "$NEWLIB" identity -- GGML_CUDA_AR_STALL_US=100
    need_ok stall-loc || return $?
    cmp "$OUT/base-default.out" "$OUT/stall-loc.out" || { echo "FAIL stalled rank 1 changes the result (loc)"; return 8; }
    # Negative controls: the same slow rank 1 without the wait must fail the reference check and differ from the base hashes.
    for tag in gen loc one; do
        case $tag in
            gen) drv "mutant-$tag" "$NEWLIB" identity -- GGML_CUDA_AR_MUTANT=1 GGML_CUDA_AR_STALL_US=100 GGML_CUDA_AR_THREADS=256 GGML_CUDA_AR_ROUND_VECS=1 ;;
            one) drv "mutant-$tag" "$NEWLIB" identity -- GGML_CUDA_AR_MUTANT=1 GGML_CUDA_AR_STALL_US=100 GGML_CUDA_AR_ROUND_VECS=0 ;;
            *)   drv "mutant-$tag" "$NEWLIB" identity -- GGML_CUDA_AR_MUTANT=1 GGML_CUDA_AR_STALL_US=100 ;;
        esac
        [[ $RC == 1 ]] || { echo "negative control void: no-wait mutant ($tag) passed or crashed (rc=$RC)"; return 5; }
        grep -aq 'ar-duplex: FAIL' "$OUT/mutant-$tag.err" || { echo "negative control void: no FAIL line ($tag)"; return 5; }
        ! cmp -s "$OUT/base-default.out" "$OUT/mutant-$tag.out" || { echo "negative control void: mutant hashes equal the base ($tag)"; return 5; }
    done
    # Informational: the unstalled no-wait mutant is a race that may or may not fire.
    drv mutant-raw-loc "$NEWLIB" identity -- GGML_CUDA_AR_MUTANT=1
}
sanitizers() {
    [[ $VISIBLE == GPU-* ]] || return 6
    local tool arm b n
    # Two kernels spin on each other: if compute-sanitizer serializes launches both arms time out and the stage records a void, never a pass.
    for tool in synccheck racecheck; do
        for arm in base loc; do
            RC=0
            local lib=$NEWLIB; [[ $arm == base ]] && lib=$BASELIB
            env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" timeout 900 nice -n 19 "$SAN" --tool "$tool" --error-exitcode 99 "$DRV" identity small \
                > "$OUT/san-$tool-$arm.out" 2> "$OUT/san-$tool-$arm.err" || RC=$?
            printf '%s\t%s\t%s\n' "$tool" "$arm" "$RC" >> "$OUT/sanitizers.tsv"
        done
        b=$(awk -v t="$tool" '$1 == t && $2 == "base" {print $3}' "$OUT/sanitizers.tsv")
        n=$(awk -v t="$tool" '$1 == t && $2 == "loc" {print $3}' "$OUT/sanitizers.tsv")
        echo "$tool: base rc=$b new rc=$n"
        [[ $n == "$b" ]] || { echo "FAIL $tool: new rc $n differs from base rc $b"; return 8; }
        [[ $n == 0 ]] || echo "NOTE $tool: base and new both rc=$n (124 = timeout: the sanitizer serializes the two spinning kernels; void, not a pass)"
    done
}
perf() {
    [[ $VISIBLE == GPU-* ]] || return 6
    local i=0 arm
    # ABBA blocks; every arm is a fresh process (the knobs are read at pipeline init), 240 calls per cell and batch.
    for arm in base loc loc base gen one one gen t64 t128 t128 t64 rv2 base loc loc base; do
        i=$((i+1))
        local lib=$NEWLIB; [[ $arm == base ]] && lib=$BASELIB
        case $arm in
            base|loc) drv "perf-$i-$arm" "$lib" perf "$arm" 240 -- ;;
            gen)  drv "perf-$i-$arm" "$lib" perf "$arm" 240 -- GGML_CUDA_AR_THREADS=256 GGML_CUDA_AR_ROUND_VECS=1 ;;
            one)  drv "perf-$i-$arm" "$lib" perf "$arm" 240 -- GGML_CUDA_AR_ROUND_VECS=0 ;;
            t64)  drv "perf-$i-$arm" "$lib" perf "$arm" 240 -- GGML_CUDA_AR_THREADS=64 ;;
            t128) drv "perf-$i-$arm" "$lib" perf "$arm" 240 -- GGML_CUDA_AR_THREADS=128 ;;
            rv2)  drv "perf-$i-$arm" "$lib" perf "$arm" 240 -- GGML_CUDA_AR_THREADS=256 GGML_CUDA_AR_ROUND_VECS=2 ;;
        esac
        [[ $RC == 0 ]] || { echo "FAIL perf $arm rc=$RC"; return 8; }
    done
    python3 codex/ar_duplex_summary.py "$OUT" > "$OUT/perf.tsv"
}
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    nice -n 19 "$W/bin/wt" freeze "$ROOT" ar-duplex --build build-cuda --desc 'stack10 HEAD + chunked AllReduce kernel pipelined (store of round r with load of round r-1) + local knobs; CPU model, byte identity vs base, mutant, sanitizer gates passed; CUDA 13.2 120a CCCL343 NCCL OFF'
    [[ $(head -n 1 "$W/frozen/ar-duplex/SOURCE") == "$SHA" ]]
    {
        printf 'commit %s\nconfigure: cmake -S . -B build-cuda ' "$SHA"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
    } > "$W/frozen/ar-duplex/RECIPE.txt"
}
items() {
    nice -n 19 python3 codex/render_ardup_items.py --out "$ROOT/codex/items/rendered"
}
stage 40 cpu-model cpu_model
stage 41 cuda-build build
stage 42 baseline-build build_base
stage 43 stamps stamps
stage 44 driver driver
stage 45 identity-mutants identity
stage 46 sanitizers sanitizers
stage 47 perf-arms perf
stage 48 freeze freeze
stage 49 render-item items
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
slow = [l for l in (p/'perf.tsv').read_text().splitlines() if l.startswith(('slower_than_base', 'slope'))]
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, perf='perf.tsv', perf_gates=slow,
                                              item='codex/items/rendered/ardup-np2-real-j5q.gpuq.toml'), indent=2) + '\n')
PY_RESULT
echo 'bw-ar-duplex: PASS'
