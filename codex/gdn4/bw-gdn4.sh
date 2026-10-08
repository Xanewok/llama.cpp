#!/usr/bin/env bash
# Build-window payload (gpuq filler; never submits). Usage: bw-gdn4.sh [--preflight | OUT_DIR]
# One CUDA build dir, four builds of the same tree (stack6 RECIPE flags, 120a, CCCL v3.4.3):
#   HEAD -> base (gated_delta_net.cu from stack8) -> mutant (attn-score reduction halved) -> HEAD again, then freeze gdn-4col-s8.
#   base and mutant keep only test-backend-ops + libs under work/gdn-4col-s8-variants/{base,mutant}/bin (rpath $ORIGIN, SHA256SUMS);
#   the tracked file is restored by the EXIT trap and the tree must be clean before the freeze.
# Stage 3 renders gdn4-cuda-checks, gdn4-depth-j5q, gdn4-np2-real-j5q (CCX0+1 pinned) and runs gpuq submit --check only.
# rc: 0 pass | 8 preflight/setup | 9 CUDA build, variants, symbols or freeze | 10 items
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 8
ROOT=$PWD
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=gdn-4col-s8
VAR=$W/work/gdn-4col-s8-variants
F=ggml/src/ggml-cuda/gated_delta_net.cu
BASE_SHA=41d2954a21106c9a65c8dd2a0697e7d33ae4f22f
KERNEL4='gated_delta_net_cudaILi128ELb[01]ELb[01]ELi4E'
ITEMS=(gdn4-cuda-checks gdn4-depth-j5q gdn4-np2-real-j5q)
log() { echo "bw-gdn4: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "${BW_BUILDGATE:-$W/bin/buildgate}" "$W/bin/wt" "$CCCL" /usr/local/cuda-13.2/bin/nvcc "$W/frozen/stack8/build/bin/round-trace" \
             "$W/wt/stack8/codex/gpu_ab.sh" "$W/wt/stack8/codex/gpu_stack8.py" "$W/work/aa-load/ccx01.sh" "$EVAL/harness/mkitem.py" \
             "$EVAL/orchestration/gpuq/items/drafts"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    command -v gpuq > /dev/null || die 8 "preflight: gpuq not on PATH"
    command -v patchelf > /dev/null || die 8 "preflight: patchelf not on PATH"
    [[ $(head -1 "$W/frozen/stack8/SOURCE" | cut -d' ' -f1) == "$BASE_SHA" ]] || die 8 "preflight: frozen/stack8 is not $BASE_SHA"
    git merge-base --is-ancestor "$BASE_SHA" HEAD || die 8 "preflight: HEAD does not contain stack8"
    [[ ! -e $W/frozen/$NAME && ! -e $VAR ]] || die 8 "preflight: frozen/$NAME or $VAR exists already"
    for i in "${ITEMS[@]}"; do
        [[ ! -e $EVAL/orchestration/gpuq/items/drafts/$i.toml ]] || die 8 "preflight: draft $i exists already"
    done
    for i in gdn4-depth-j5q gdn4-np2-real-j5q; do
        [[ ! -e $EVAL/harness/experiments/$i.toml ]] || die 8 "preflight: experiment $i exists already"
    done
    [[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "preflight: tracked changes present"
    exit 0
fi
OUT=$(readlink -m "${1:-$ROOT/codex/gdn4/bw-results}")
mkdir -p "$OUT"
[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window"
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}
log "start HEAD=$SHA out=$OUT"
trap 'git checkout -q -- "$F"' EXIT

cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() { "$BUILDGATE" cmake --build build-cuda --target llama-server test-backend-ops -j4; }
is_elf() { [[ -f $1 && ! -L $1 && $(head -c4 "$1" | tail -c3) == ELF ]]; }
# GGML_CUDA_CCCL_VERSION v3.4.3: the DeviceTopK instantiations are the CCCL fingerprint of the stack6 recipe
has_symbols() { # LIB PATTERN WANT(0|1)
    local n; n=$(strings -a "$1" | grep -Ec "$2")
    if [[ $3 == 1 ]]; then (( n > 0 )); else (( n == 0 )); fi
}
stash() { # VARIANT: test-backend-ops and libs of the current build, relocatable
    local d=$VAR/$1/bin f
    mkdir -p "$d"
    cp -p build-cuda/bin/test-backend-ops "$d/" || return 1
    for f in build-cuda/bin/*.so*; do cp -a "$f" "$d/" || return 1; done
    for f in "$d"/*; do
        if is_elf "$f"; then patchelf --set-rpath '$ORIGIN' "$f" || return 1; fi
    done
    (cd "$d" && find . -maxdepth 1 \( -type f -o -type l \) | sed 's#^\./##' | grep -v '^SHA256SUMS$' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS) || return 1
    [[ -z $(ldd "$d/test-backend-ops" | grep 'not found' || true) ]] || return 1
    [[ -z $(ldd "$d/test-backend-ops" | awk '/libggml/ {print $3}' | grep -v "^$(cd "$d" && pwd -P)/" || true) ]]
}

cuda_stage() {
    export CUDA_HOME=/usr/local/cuda-13.2
    "$W/bin/wt" work gdn-4col-s8-variants --purpose "stack8-source (base) and reduction-mutant test-backend-ops builds for gdn4-cuda-checks (bw-gdn4)" > /dev/null || return 1
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}" || return 1
    build || return 1
    has_symbols build-cuda/bin/libggml-cuda.so "$KERNEL4" 1 || { echo "HEAD build lacks the four-column kernel"; return 1; }

    git show "$BASE_SHA:$F" > "$F" || return 1
    build || return 1
    has_symbols build-cuda/bin/libggml-cuda.so "$KERNEL4" 0 || { echo "base build carries the four-column kernel"; return 1; }
    stash base || return 1
    git checkout -q -- "$F" || return 1

    sed -i 's#warp_reduce_sum<lanes_per_col>(attn_partial)#warp_reduce_sum<lanes_per_col / 2>(attn_partial)#' "$F" || return 1
    [[ $(git diff --numstat -- "$F") == $'2\t2\t'"$F" ]] || { echo "mutant edit is not two lines"; git diff --numstat -- "$F"; return 1; }
    git diff -U0 -- "$F" > "$OUT/mutant.diff"
    build || return 1
    stash mutant || return 1
    git checkout -q -- "$F" || return 1

    build || return 1
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || { echo "tree moved or dirty after the variant builds"; return 1; }
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 1
    local n; n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "CCCL DeviceTopK strings: $n"
    [[ $n == 25 ]] || { echo "expected 25 DeviceTopK strings (CCCL v3.4.3)"; return 1; }
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null || { echo "ggml stamp $SHORT not in libggml-base.so"; return 1; }
    has_symbols build-cuda/bin/libggml-cuda.so "$KERNEL4" 1 || { echo "final build lacks the four-column kernel"; return 1; }
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack8 41d2954a2 + upstream #30087 (GDN four state columns per warp) + GDN perf cases in test-backend-ops; CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'name %s\n' "$NAME"
        printf 'commit %s (base stack8 %s, branch %s)\n' "$SHA" "$BASE_SHA" "$(git rev-parse --abbrev-ref HEAD)"
        printf 'worktree %s, build dir build-cuda\n' "$ROOT"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf ' (CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES="")\n'
        printf 'build: bin/buildgate cmake --build build-cuda --target llama-server test-backend-ops -j4\n'
        printf 'variants: %s/{base,mutant}/bin from the same build dir (base = %s:%s, mutant = see %s/mutant.diff)\n' "$VAR" "$BASE_SHA" "$F" "$OUT"
        printf 'round-trace = cp of llama-server; no CUDA binary executed\n'
    } > "$W/frozen/$NAME/RECIPE.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE" | cut -d' ' -f1) == "$SHA" ]] || return 1
    {
        echo "bw-gdn4 $SHA"
        echo "DeviceTopK strings $n; ggml stamp $SHORT"
        echo "candidate kernel symbol $KERNEL4: frozen yes, base no, mutant yes"
        echo "base $VAR/base/bin, mutant $VAR/mutant/bin; no CUDA binary executed, no Mac model gate (kernel-only change)"
    } > "$OUT/summary.txt"
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/variants/freeze (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/$NAME; variants in $VAR"

python3 codex/gdn4/make_items.py "$SHA" "$OUT/summary.txt" "$OUT" > "$OUT/items.log" 2>&1 || die 10 "item render/check (see $OUT/items.log)"
log "items checked, NOT submitted: $(grep -o 'drafts/[^ ]*' "$OUT/items.log" | tr '\n' ' ')"
log "done"
exit 0
