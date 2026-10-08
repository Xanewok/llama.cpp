#!/usr/bin/env bash
# Build-window payload (gpuq filler; never submits). Usage: bw-chain-m2.sh [--preflight | OUT_DIR]
# Stage 1  CPU gate at HEAD with GCC: fixtures build, np2-matrix.sh quick (plain + Meta k5 chain off/on, greedy slot 1 path
#          switches, off-vs-off floor, SWAP_SEQ and NO_RNG_SYNC mutants) judged by judge-np2.py quick.
# Stage 2  CUDA build (stack6 RECIPE flags) of llama-server, CCCL check, round-trace copy, wt freeze chain-m2.
# Stage 3  Render chain-m2-np2-real-j5q, chain-m2-np1-depth-j5q (+ CCX0+1 copies); gpuq submit --check only.
# rc: 0 pass | 2 CPU gate failed (judge rc in cpu-gate/judge.log) | 75 CPU gate skipped | 8 preflight/setup/CPU build
#     | 9 CUDA build/freeze | 10 items
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 8
ROOT=$PWD
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
NAME=chain-m2
ITEMS=(chain-m2-np2-real-j5q chain-m2-np1-depth-j5q)
log() { echo "bw-$NAME: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "${BW_BUILDGATE:-$W/bin/buildgate}" "${BW_GATE:-$W/bin/cpugate}" "$W/bin/wt" "$CCCL" "$MODEL" \
             "$W/frozen/stack7/build/bin/round-trace" "$W/wt/stack7/codex/gpu_ab.sh" "$W/work/aa-load/ccx01.sh" \
             "$EVAL/harness/mkitem.py" "$EVAL/orchestration/gpuq/items/drafts"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    command -v gpuq >/dev/null || die 8 "preflight: gpuq not on PATH"
    [[ ! -e $W/frozen/$NAME ]] || die 8 "preflight: $W/frozen/$NAME exists already"
    for i in "${ITEMS[@]}"; do
        [[ ! -e $EVAL/harness/experiments/$i.toml ]] || die 8 "preflight: experiment $i exists already"
    done
    [[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "preflight: tracked changes present"
    exit 0
fi
OUT=$(readlink -m "${1:-$ROOT/codex/m2/bw-results}")
mkdir -p "$OUT"
[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window"
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}

# ---------------------------------------------------------------- stage 1: CPU gate
{
    "$BUILDGATE" cmake -S codex/stack7-test -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON \
        -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON &&
    "$BUILDGATE" cmake --build build-cpu --target bw-np2-driver -j4
} > "$OUT/build-cpu.log" 2>&1 || die 8 "CPU build (see $OUT/build-cpu.log)"
M2_CELLS=quick M2_GATE=${BW_GATE:-$W/bin/cpugate} bash codex/m2/np2-matrix.sh "$ROOT/build-cpu/bin/bw-np2-driver" "$MODEL" \
    "$OUT/cpu-gate" > "$OUT/cpu-gate.log" 2>&1
mrc=$?
[[ $mrc == 75 ]] && die 75 "skipped: CPU gate (see $OUT/cpu-gate.log)"
[[ $mrc == 0 ]] || die 8 "np2 matrix rc=$mrc (see $OUT/cpu-gate.log)"
python3 codex/m2/judge-np2.py "$OUT/cpu-gate" quick > "$OUT/cpu-gate/judge.log" 2>&1 || die 2 "CPU gate (see $OUT/cpu-gate/judge.log)"
[[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || die 8 "CPU gate left the tree dirty or moved HEAD"
log "CPU gate pass"

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
    strings build-cuda/bin/libggml-base.so | grep -x "$short" >/dev/null || { echo "ggml stamp $short not in libggml-base.so"; return 1; }
    strings build-cuda/bin/libllama.so | grep -c 'the batch does not fit the MTP chain' > /dev/null || { echo "M2 chain refusal missing from libllama"; return 1; }
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack7 48b82b0d1 + MTP draft chain for several sequences (M2, np2); CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'name %s\n' "$NAME"
        printf 'commit %s (base stack7 48b82b0d1559e936dbf612c871f2583cd4c79f3e, branch %s)\n' "$SHA" "$(git rev-parse --abbrev-ref HEAD)"
        printf 'worktree %s, build dir build-cuda\n' "$ROOT"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf ' (CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES="")\n'
        printf 'build: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'round-trace = cp of llama-server (harness arms must not execute a file named llama-server)\n'
        printf 'DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed; CPU evidence %s\n' "$n" "$short" "$OUT/cpu-gate/summary.json"
    } > "$W/frozen/$NAME/RECIPE.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE" | cut -d' ' -f1) == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/$NAME"

# ---------------------------------------------------------------- stage 3: items (check only, never submit)
python3 codex/m2/make_items.py "$SHA" "$OUT/cpu-gate/summary.json" "$OUT" > "$OUT/items.log" 2>&1 || die 10 "item render/check (see $OUT/items.log)"
log "items checked, NOT submitted: $(grep -o 'drafts/[^ ]*' "$OUT/items.log" | tr '\n' ' ')"
log "done"
exit 0
