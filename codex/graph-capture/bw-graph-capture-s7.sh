#!/usr/bin/env bash
# Build-window payload (gpuq filler; submits nothing). Usage: bw-graph-capture-s7.sh [--preflight | OUT_DIR]
#
# CPU evidence comes from the Mac (codex/graph-capture/cpu-gates.sh via bin/bw-mac): the newest
# work/build-window-1007/graph-capture-s7-mac-a*/ whose RESULTS.txt ends in "PASS all CPU gates" and whose HEAD has the
# same non-codex tree as this HEAD. No CPU tests run here.
# Stage 1  CUDA build (stack7 RECIPE flags), stamps (CCCL DeviceTopK 25, ggml commit stamp, AR-node symbols), wt freeze.
# Stage 2  Render both depth items from codex/graph-capture/items/*.toml.in, mkitem, CCX0+1 copy, gpuq submit --check.
# Exit: 0 pass | 8 preflight, dirty tree or missing Mac evidence | 9 CUDA build/stamps/freeze | 10 item render/check.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 8
ROOT=$PWD
D=codex/graph-capture
NAME=graph-capture-s7
ITEMS=(graph-capture-s7-np1-j5q graph-capture-s7-np2-real-j5q)
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
log() { echo "bw-$NAME: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

mac_evidence() { # prints the RESULTS.txt of the newest passing Mac run of this code
    local n dir sha
    for n in $(ls -d "$W"/work/build-window-1007/$NAME-mac-a* 2>/dev/null | sed 's/.*-mac-a//' | grep -x '[0-9]*' | sort -n -r); do
        dir=$W/work/build-window-1007/$NAME-mac-a$n
        [[ -f $dir/RESULTS.txt && -f $dir/mac-run.txt ]] || continue
        [[ $(tail -1 "$dir/RESULTS.txt") == *"PASS all CPU gates" ]] || continue
        sha=$(sed -n 's/^HEAD=//p' "$dir/mac-run.txt")
        git diff --quiet "$sha" HEAD -- . ':!codex' 2>/dev/null || continue
        echo "$dir/RESULTS.txt"
        return 0
    done
    return 1
}

if [[ ${1:-} == --preflight ]]; then
    for p in "$W/bin/buildgate" "$W/bin/wt" "$CCCL" /usr/local/cuda-13.2/bin/nvcc "$EVAL/harness/mkitem.py" \
             "$W/frozen/stack7/RECIPE.txt" "$W/work/aa-load/ccx01.sh" "$ROOT/$D/ccx01_copy.py" \
             "$ROOT/$D/items/${ITEMS[0]}.toml.in" "$ROOT/$D/items/${ITEMS[1]}.toml.in"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    for t in gpuq cmake git python3 strings; do command -v "$t" > /dev/null || die 8 "preflight: $t not on PATH"; done
    [[ ! -e $W/frozen/$NAME ]] || die 8 "preflight: $W/frozen/$NAME exists"
    [[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "preflight: tracked changes present"
    mac_evidence > /dev/null || die 8 "preflight: no passing Mac CPU run of this code ($NAME-mac-a*)"
    for id in "${ITEMS[@]}"; do [[ ! -e $EVAL/harness/experiments/$id.toml ]] || die 8 "preflight: experiment $id exists"; done
    exit 0
fi
OUT=$(readlink -m "${1:?OUT}")
mkdir -p "$OUT"
bash "$0" --preflight || exit $?
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
EVIDENCE=$(mac_evidence)
log "start HEAD=$SHA out=$OUT cpu evidence $EVIDENCE"

# ---------------------------------------------------------------- stage 1: CUDA build, freeze
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_stage() {
    export CUDA_HOME=/usr/local/cuda-13.2
    "$W/bin/buildgate" cmake -B build-cuda "${cuda_flags[@]}" || return 1
    "$W/bin/buildgate" cmake --build build-cuda --target llama-server -j4 || return 1
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || { echo "tree moved"; return 1; }
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 1
    local n; n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "CCCL DeviceTopK strings: $n"
    [[ $n == 25 ]] || { echo "expected 25 DeviceTopK strings (CCCL v3.4.3)"; return 1; }
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null || { echo "ggml stamp $SHORT missing"; return 1; }
    strings build-cuda/bin/libggml-cuda.so | grep -c '^ggml_backend_comm_allreduce_node$' > /dev/null || { echo "AR node proc missing"; return 1; }
    strings build-cuda/bin/libggml-base.so | grep -c 'meta_ar_node: uid=' > /dev/null || { echo "meta AR node planner missing"; return 1; }
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack7 48b82b0d1 + AllReduce as CUDA graph nodes (GGML_META_AR_NODE, default off) + meta plan arena fix; CUDA 13.2 120a CCCL343 NCCL OFF" || return 1
    {
        printf 'name %s\ncommit %s (base stack7 48b82b0d1559e936dbf612c871f2583cd4c79f3e, branch %s)\n' "$NAME" "$SHA" "$(git rev-parse --abbrev-ref HEAD)"
        printf 'worktree %s, build dir build-cuda\nconfigure: cmake -B build-cuda ' "$ROOT"; printf '%q ' "${cuda_flags[@]}"
        printf ' (CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES="")\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'round-trace = cp of llama-server; DeviceTopK %s; stamp %s; no CUDA binary executed; CPU evidence %s\n' "$n" "$SHORT" "$EVIDENCE"
    } > "$W/frozen/$NAME/RECIPE.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE" | cut -d' ' -f1) == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/stamps/freeze (see $OUT/cuda.log)"
log "frozen $W/frozen/$NAME"

# ---------------------------------------------------------------- stage 2: items (check only, never submit)
item_stage() {
    local id exp draft
    mkdir -p "$OUT/items"
    for id in "${ITEMS[@]}"; do
        exp=$EVAL/harness/experiments/$id.toml
        draft=$EVAL/orchestration/gpuq/items/drafts/$id.toml
        sed -e "s/@COMMIT@/$SHA/g" -e "s/@DATE@/$(date +%F)/g" -e "s|@CPU_EVIDENCE@|$EVIDENCE|g" "$D/items/$id.toml.in" > "$exp" || return 1
        ! grep -q '@[A-Z_]*@' "$exp" || { echo "unsubstituted placeholder in $exp"; return 1; }
        (cd "$EVAL" && python3 -m harness.mkitem "$exp" --out "$draft") || return 1
        python3 "$D/ccx01_copy.py" "$draft" "$OUT/items/$id-ccx01.gpuq.toml" || return 1
        gpuq submit --check "$OUT/items/$id-ccx01.gpuq.toml" || return 1
        echo "checked, NOT submitted: $OUT/items/$id-ccx01.gpuq.toml"
    done
}
item_stage > "$OUT/items.log" 2>&1 || die 10 "item render/check (see $OUT/items.log)"
log "items checked, NOT submitted: $(ls "$OUT"/items/*.gpuq.toml | tr '\n' ' ')"
exit 0
