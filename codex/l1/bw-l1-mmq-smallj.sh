#!/usr/bin/env bash
# Build-window payload (gpuq filler; never submits). Usage: bw-l1-mmq-smallj.sh [--preflight | OUT_DIR]
# Stage 1  CUDA build (stack6 RECIPE flags) of llama-server + test-backend-ops, CCCL check, L1 string checks, round-trace copy, wt freeze l1-mmq-smallj.
# Stage 2  Render l1-mmq-cuda-checks, l1-mmq-perf, l1-mmq-np2-real-j5q-ccx01; gpuq submit --check only.
# rc: 0 pass | 8 preflight/setup | 9 CUDA build/freeze | 10 items
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 8
ROOT=$PWD
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=l1-mmq-smallj
ITEMS=(l1-mmq-np2-real-j5q)
log() { echo "bw-$NAME: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "${BW_BUILDGATE:-$W/bin/buildgate}" "$W/bin/wt" "$CCCL" "$W/frozen/stack8/build/bin/round-trace" \
             "$W/wt/stack7/codex/gpu_ab.sh" "$W/work/aa-load/ccx01.sh" "$EVAL/harness/mkitem.py" "$EVAL/orchestration/gpuq/items/drafts"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    command -v gpuq >/dev/null || die 8 "preflight: gpuq not on PATH"
    [[ ! -e $W/frozen/$NAME ]] || die 8 "preflight: $W/frozen/$NAME exists already"
    for i in "${ITEMS[@]}"; do
        [[ ! -e $EVAL/harness/experiments/$i.toml ]] || die 8 "preflight: experiment $i exists already"
    done
    for i in l1-mmq-cuda-checks l1-mmq-perf l1-mmq-np2-real-j5q-ccx01; do
        [[ ! -e $EVAL/orchestration/gpuq/items/drafts/$i.toml ]] || die 8 "preflight: draft $i exists already"
    done
    [[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "preflight: tracked changes present"
    exit 0
fi
OUT=$(readlink -m "${1:-$ROOT/codex/l1/bw-results}")
mkdir -p "$OUT"
[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window"
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}

cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_stage() {
    export CUDA_HOME=/usr/local/cuda-13.2
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}" || return 1
    "$BUILDGATE" cmake --build build-cuda --target llama-server test-backend-ops -j4 || return 1
    [[ $(git rev-parse HEAD) == "$SHA" ]] || { echo "HEAD moved during the build"; return 1; }
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 1
    local n; n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "CCCL DeviceTopK strings: $n"
    [[ $n == 25 ]] || { echo "expected 25 DeviceTopK strings (CCCL v3.4.3)"; return 1; }
    local short; short=$(git rev-parse --short HEAD)
    strings build-cuda/bin/libggml-base.so | grep -x "$short" >/dev/null || { echo "ggml stamp $short not in libggml-base.so"; return 1; }
    grep -aq 'mmq_tiles: type=' build-cuda/bin/libggml-cuda.so || { echo "libggml-cuda lacks the mmq_tiles log line"; return 1; }
    grep -aq 'GGML_CUDA_MMQ_TILES' build-cuda/bin/libggml-cuda.so || { echo "libggml-cuda lacks the GGML_CUDA_MMQ_TILES switch"; return 1; }
    grep -aq 'mmq-hash' build-cuda/bin/test-backend-ops || { echo "test-backend-ops lacks the mmq-hash mode"; return 1; }
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack8 41d2954a2 + shape-gated small-I MMQ tiles (J 16/24, tile grid >= 90 % of whole SM waves) + local GGML_CUDA_MMQ_TILES switch; CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'name %s\n' "$NAME"
        printf 'commit %s (base stack8 41d2954a21106c9a65c8dd2a0697e7d33ae4f22f, branch %s)\n' "$SHA" "$(git rev-parse --abbrev-ref HEAD)"
        printf 'worktree %s, build dir build-cuda\n' "$ROOT"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf ' (CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES="")\n'
        printf 'build: bin/buildgate cmake --build build-cuda --target llama-server test-backend-ops -j4\n'
        printf 'round-trace = cp of llama-server; GGML_CUDA_MMQ_TILES = auto (unset) | default | small | auto-drop; test-backend-ops mmq-hash DEVICE\n'
        printf 'DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed\n' "$n" "$short"
    } > "$W/frozen/$NAME/RECIPE.txt"
    cp "$W/frozen/$NAME/RECIPE.txt" "$OUT/build-evidence.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE" | cut -d' ' -f1) == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/$NAME"

python3 codex/l1/make_items.py "$SHA" "$OUT/build-evidence.txt" "$OUT" > "$OUT/items.log" 2>&1 || die 10 "item render/check (see $OUT/items.log)"
log "items checked, NOT submitted: $(grep -o 'drafts/[^ ]*' "$OUT/items.log" | tr '\n' ' ')"
log "done"
exit 0
