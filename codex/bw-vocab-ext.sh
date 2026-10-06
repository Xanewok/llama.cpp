#!/usr/bin/env bash
# Build-window payload for the per-request draft vocabulary extension (never submits anything).
# Usage: bw-vocab-ext.sh [--preflight] [OUT_DIR]   (default OUT_DIR codex/runs/bw-vocab-ext)
#
# Stage 1  CPU build of this tree and of the stack6 base source (git archive ac7465885 into codex/base-src), 4B fixture
#          (Qwen3.5-4B UD-Q4_K_XL, one MTP layer), -c 4096:
#          a) test-draft-vocab-ext single and tp (Meta(CPU,CPU)): off = exactly the base ids; filler, placed (2 seqs,
#             quota, update), re-created context and backend top-k logits equal the full head; a second MTP context is
#             refused; then the same with LLAMA_DRAFT_VOCAB_EXT_MUTANT=row, which must fail on the logit check;
#          b) bw-ext-driver (server_context, copy task, 1/17 reduced head) meta 0/1 x greedy/prob np1 and prob np2:
#             ref/ref2 = stack6 base build, off = this tree without the extension, on = --spec-draft-vocab-ext 512;
#             bw-ext-compare.py judges base repeatability, off == stack6, off inert, engagement, and on accepting
#             out-of-head draft tokens (greedy np1).
# Stage 2  CUDA build (stack6 recipe) + CCCL check + round-trace copy + wt freeze draft-vocab-ext + RECIPE.txt.
# Stage 3  GPU item generation (gen_items_vocab_ext.py) + gpuq submit --check (no submit).
#
# Exit codes: 0 all pass | 2 extension off differs from the stack6 base | 3 test-draft-vocab-ext failed | 4 mutant escaped
# | 5 stack6 base not repeatable | 6 no engagement | 7 OOV gate (off not inert, or on accepted none) | 8 CPU build/run error or compare crash
# | 9 CUDA build/freeze/CCCL | 10 item generation or check. 2-8 stop before stage 2 (no freeze, no items).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
W=$HOME/repos/qwen38-eval-worktrees
cd "$(dirname "$(readlink -f "$0")")/.." || exit 8
ROOT=$PWD
BASE=ac7465885dc9f13de396ccd91b4be5eaffde3730
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=draft-vocab-ext
N_PREDICT=160
log() { echo "bw-vocab-ext: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "$W/bin/buildgate" "$W/bin/cpugate" "$W/bin/wt" "$MODEL" "$CCCL" "$W/frozen/stack6/build/bin/round-trace" \
             "$ROOT/codex/test-draft-vocab-ext.cpp" "$ROOT/codex/bw-ext-driver.cpp" "$ROOT/codex/bw-ext-compare.py" \
             "$ROOT/codex/gen_items_vocab_ext.py" "$W/wt/harness-real-prompt/harness/experiments/realprompt-depth-j5q.toml" \
             /home/xanewok/repos/qwen38-eval/recipes/_common/kld/tokens64k.bin /home/xanewok/repos/qwen38-eval/recipes/_common/kld/tokens8k.bin; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    command -v gpuq > /dev/null || die 8 "preflight: gpuq not on PATH"
    git cat-file -e "$BASE^{commit}" || die 8 "preflight: base commit $BASE missing"
    [[ ! -e $W/frozen/$NAME ]] || die 8 "preflight: $W/frozen/$NAME already exists"
    exit 0
fi
OUT=$(readlink -m "${1:-$ROOT/codex/runs/bw-vocab-ext}")
mkdir -p "$OUT"

[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window (wt freeze needs a clean tree)"
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=$W/bin/buildgate
CPUGATE=$W/bin/cpugate
cpu_flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)

# ---------------------------------------------------------------- stage 1: CPU builds
build_driver() { # $1 source root, $2 output, $3.. extra flags
    local src=$1 out=$2; shift 2
    "$BUILDGATE" g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED "$@" -I"$src/tools/server" -I"$src/common" \
        -I"$src/include" -I"$src/ggml/include" -I"$src/vendor" -I"$src/tools/mtmd" -I"$src" "$ROOT/codex/bw-ext-driver.cpp" \
        "$src/build-cpu/tools/server/libserver-context.a" "$src/build-cpu/common/libllama-common-base.a" \
        -L"$src/build-cpu/bin" -Wl,-rpath,"$src/build-cpu/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$out"
}
{
    "$BUILDGATE" cmake -B build-cpu "${cpu_flags[@]}" &&
    "$BUILDGATE" cmake --build build-cpu --target llama-server -j4 &&
    "$BUILDGATE" g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Isrc -Iggml/include \
        codex/test-draft-vocab-ext.cpp -Lbuild-cpu/bin -Wl,-rpath,"$ROOT/build-cpu/bin" -lllama -lggml -lggml-cpu -lggml-base \
        -o build-cpu/bin/test-draft-vocab-ext &&
    build_driver "$ROOT" build-cpu/bin/bw-ext-driver -DVOCAB_EXT &&
    if [[ $(cat codex/base-src/.base 2>/dev/null) != "$BASE" ]]; then
        rm -rf codex/base-src && mkdir -p codex/base-src && git archive "$BASE" | tar -x -C codex/base-src && echo "$BASE" > codex/base-src/.base
    fi &&
    (cd codex/base-src && "$BUILDGATE" cmake -B build-cpu "${cpu_flags[@]}" && "$BUILDGATE" cmake --build build-cpu --target llama-server -j4) &&
    build_driver "$ROOT/codex/base-src" codex/base-src/build-cpu/bin/bw-ext-driver
} > "$OUT/build-cpu.log" 2>&1 || die 8 "CPU build (see $OUT/build-cpu.log)"
log "CPU builds ok (this tree + stack6 base)"

# ---------------------------------------------------------------- stage 1a: llama-level test and mutant
for mode in single tp; do
    env -u LLAMA_DRAFT_VOCAB_EXT_MUTANT "$CPUGATE" build-cpu/bin/test-draft-vocab-ext "$MODEL" "$mode" \
        > "$OUT/test-$mode.log" 2>&1 || die 3 "test-draft-vocab-ext $mode (see $OUT/test-$mode.log)"
    log "test-draft-vocab-ext $mode PASS"
    LLAMA_DRAFT_VOCAB_EXT_MUTANT=row "$CPUGATE" build-cpu/bin/test-draft-vocab-ext "$MODEL" "$mode" \
        > "$OUT/mutant-$mode.log" 2>&1 && die 4 "mutant row passed test-draft-vocab-ext $mode"
    grep -q "placed: extension logit does not match the full head" "$OUT/mutant-$mode.log" ||
        die 4 "mutant row failed $mode for another reason (see $OUT/mutant-$mode.log)"
    log "mutant row fails $mode as required"
done

# ---------------------------------------------------------------- stage 1b: server-level matrix
run_driver() { # meta mode np variant
    local meta=$1 mode=$2 np=$3 variant=$4 bin=build-cpu/bin/bw-ext-driver ext=0
    case $variant in
        ref|ref2) bin=codex/base-src/build-cpu/bin/bw-ext-driver ;;
        on) ext=512 ;;
    esac
    env -u LLAMA_DRAFT_VOCAB_EXT_MUTANT DRIVER_VOCAB_EXT=$ext "$CPUGATE" "$bin" "$MODEL" "$OUT/$meta-$mode-np$np-$variant.json" \
        "$meta" "$np" "$mode" "$N_PREDICT" > "$OUT/$meta-$mode-np$np-$variant.log" 2>&1 ||
        die 8 "driver meta=$meta $mode np$np $variant (see $OUT/$meta-$mode-np$np-$variant.log)"
    log "ran meta=$meta $mode np$np $variant"
}
for meta in 0 1; do
    for mode in greedy prob; do
        for variant in ref ref2 off on; do
            run_driver "$meta" "$mode" 1 "$variant"
        done
    done
    for variant in ref ref2 off on; do
        run_driver "$meta" prob 2 "$variant"
    done
done
grep -h "set_draft_vocab_ext: seq\|draft vocab ext: seq" "$OUT"/*-on.log | sort | uniq -c > "$OUT/engagement.txt"

python3 codex/bw-ext-compare.py "$OUT" | tee "$OUT/compare.log"
stage1=${PIPESTATUS[0]}
case $stage1 in 0|2|5|6|7|8) ;; *) stage1=8 ;; esac
{
    echo "HEAD $SHA base $BASE model $MODEL"
    cat "$OUT"/test-single.log "$OUT"/test-tp.log | grep '^M:'
    echo "mutant row: fails single and tp on the extension logit check"
    cat "$OUT/compare.log"
} > "$OUT/summary.txt"
log "stage 1 verdict rc=$stage1"
[[ $stage1 == 0 ]] || die "$stage1" "CPU matrix (see $OUT/compare.log); no freeze, no items"

# ---------------------------------------------------------------- stage 2: CUDA build, freeze
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_stage() {
    export CUDA_HOME=/usr/local/cuda-13.2
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}" || return 1
    "$BUILDGATE" cmake --build build-cuda --target llama-server -j4 || return 1
    [[ $(git rev-parse HEAD) == "$SHA" ]] || { echo "HEAD moved during the build"; return 1; }
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 1
    local n; n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "CCCL DeviceTopK strings: $n"
    [[ $n == 25 ]] || { echo "expected 25 DeviceTopK strings (CCCL v3.4.3)"; return 1; }
    local short; short=$(git rev-parse --short HEAD)
    strings build-cuda/bin/libggml-base.so | grep -qx "$short" || { echo "ggml stamp $short not in libggml-base.so"; return 1; }
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack6 + per-request MTP draft vocab extension (--spec-draft-vocab-ext; meta range get/set); CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'name %s\n' "$NAME"
        printf 'commit %s (base stack6 %s + %s)\n' "$SHA" "$BASE" "$(git log --format=%h --reverse "$BASE..$SHA" -- . ':!codex' | paste -sd' ')"
        printf 'worktree %s, build dir build-cuda, HEAD at the commit when configured\n' "$ROOT"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES=""; DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed\n' "$n" "$short"
        printf 'round-trace = cp of llama-server (harness arms must not execute a file named llama-server); listed in SHA256SUMS\n'
        printf 'CPU evidence: %s/summary.txt\n' "$OUT"
    } > "$W/frozen/$NAME/RECIPE.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE") == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/$NAME"

# ---------------------------------------------------------------- stage 3: GPU items (checked, not submitted)
python3 codex/gen_items_vocab_ext.py "$OUT" > "$OUT/items.log" 2>&1 || die 10 "item generation (see $OUT/items.log)"
shopt -s nullglob
items=(codex/items/draft-vocab-ext-*.gpuq.toml)
(( ${#items[@]} > 0 )) || die 10 "no item files generated"
for f in "${items[@]}"; do
    gpuq submit --check "$f" >> "$OUT/items.log" 2>&1 || die 10 "gpuq submit --check $f (see $OUT/items.log)"
    log "item checked, NOT submitted: $f"
done

log "done"
exit 0
