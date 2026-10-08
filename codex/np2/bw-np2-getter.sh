#!/usr/bin/env bash
# Build-window payload (gpuq filler; never submits). Usage: bw-np2-getter.sh [--preflight | OUT_DIR]
# Stage 1  codex/np2/cpu-gate.sh on this host (CPU build, getter gate scenarios, full rollback suite, np2 driver identity vs the
#          stack7 gate plain + Meta, extras/tail/row mutants).
# Stage 2  CUDA build (stack6 RECIPE flags) of llama-server + test-backend-ops, CCCL check, round-trace copy, wt freeze np2-getter.
# Stage 3  Render np2-getter-real-j5q, np2-getter-np1-depth-j5q (+ CCX0+1 copies) and np2-getter-cuda-checks; gpuq submit --check only.
# rc: 0 pass | 2 CPU gate failed (its rc in cpu-gate.log) | 75 CPU gate skipped for RAM | 8 preflight/setup | 9 CUDA build/freeze | 10 items
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 8
ROOT=$PWD
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=np2-getter
ITEMS=(np2-getter-real-j5q np2-getter-np1-depth-j5q)
log() { echo "bw-$NAME: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "${BW_BUILDGATE:-$W/bin/buildgate}" "${BW_GATE:-$W/bin/cpugate}" "$W/bin/wt" "$CCCL" \
             /mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf "$W/frozen/stack7/build/bin/round-trace" \
             "$W/wt/stack7/codex/gpu_ab.sh" "$W/work/aa-load/ccx01.sh" "$EVAL/harness/mkitem.py" "$EVAL/orchestration/gpuq/items/drafts"; do
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
OUT=$(readlink -m "${1:-$ROOT/codex/np2/bw-results}")
mkdir -p "$OUT"
[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window"
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}

# ---------------------------------------------------------------- stage 1: CPU gate
bash codex/np2/cpu-gate.sh "$OUT/cpu-gate" > "$OUT/cpu-gate.log" 2>&1
crc=$?
case $crc in
    0) ;;
    75) die 75 "skipped: RAM (CPU gate; see $OUT/cpu-gate.log)" ;;
    *) die 2 "CPU gate rc=$crc (see $OUT/cpu-gate.log)" ;;
esac
[[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || die 8 "CPU gate left the tree dirty or moved HEAD"
log "CPU gate pass"

# ---------------------------------------------------------------- stage 2: CUDA build, freeze
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
    grep -aq 'gdn_joint_mutant: engaged' build-cuda/bin/libggml-cuda.so || { echo "joint kernel mutant switch missing"; return 1; }
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack7 48b82b0d1 + GDN state by id at n_seqs > 1 (n_rs == n_seqs) + joint fused GDN kernel; CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'name %s\n' "$NAME"
        printf 'commit %s (base stack7 48b82b0d1559e936dbf612c871f2583cd4c79f3e, branch %s)\n' "$SHA" "$(git rev-parse --abbrev-ref HEAD)"
        printf 'worktree %s, build dir build-cuda\n' "$ROOT"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf ' (CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES="")\n'
        printf 'build: bin/buildgate cmake --build build-cuda --target llama-server test-backend-ops -j4\n'
        printf 'round-trace = cp of llama-server; test-backend-ops carries the local GGML_CUDA_GDN_JOINT_MUTANT negative control\n'
        printf 'DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed; CPU evidence %s\n' "$n" "$short" "$OUT/cpu-gate/summary.txt"
    } > "$W/frozen/$NAME/RECIPE.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE" | cut -d' ' -f1) == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/$NAME"

# ---------------------------------------------------------------- stage 3: items (check only, never submit)
python3 codex/np2/make_items.py "$SHA" "$OUT/cpu-gate/summary.txt" "$OUT" > "$OUT/items.log" 2>&1 || die 10 "item render/check (see $OUT/items.log)"
log "items checked, NOT submitted: $(grep -o 'drafts/[^ ]*' "$OUT/items.log" | tr '\n' ' ')"
log "done"
exit 0
