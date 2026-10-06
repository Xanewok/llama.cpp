#!/usr/bin/env bash
# Build-window payload (gpuq filler item; never submits anything). Usage: bw-host-trims-v2.sh [OUT_DIR]
#
# Stage 1  CPU build (stack6 recipe, -DGGML_CUDA=OFF) + np2 matrix on the 4B fixture: Meta(CPU,CPU) and plain, MTP draft,
#          two concurrent requests (short prompt on slot 0, long prompt on slot 1; second phase reuses the cached prompts),
#          -c 4096. Per-stream tokens and draft counters of every variant are compared with the stack6-equivalent
#          reference (LLAMA_HOST_TRIMS_SEQRM=0 LLAMA_HOST_TRIMS_BULKH=0).
# Stage 2  CUDA build + CCCL check + wt freeze + np2 IQ4 item generation + gpuq submit --check (no submit).
#
# Exit codes: 0 all pass | 2 identity failure | 3 stage 1 passed but the b115f8180 negative control did not fail (CPU does
# not reproduce the GPU defect; stage 2 still runs) | 4 mutant escaped | 5 reference not repeatable | 6 no prefix reuse
# in the test | 7 verify hooks | 8 CPU build/run error | 9 CUDA build/freeze/CCCL | 10 item generation or check.
# 2 4 5 6 7 8 stop before stage 2 (no freeze, no items).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
W=$HOME/repos/qwen38-eval-worktrees
cd "$(dirname "$(readlink -f "$0")")/.." || exit 8
ROOT=$PWD
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
log() { echo "bw-host-trims-v2: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "$W/bin/buildgate" "$W/bin/cpugate" "$W/bin/wt" "$MODEL" "$ROOT/codex/bw-np2-driver.cpp" "$ROOT/codex/bw-np2-compare.py" \
             "$ROOT/codex/bw-make-items.py" "$W/frozen/stack6/build/bin/round-trace" "/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    exit 0
fi
OUT=$(readlink -m "${1:-$ROOT/codex/bw-results}")
mkdir -p "$OUT"

[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window (wt freeze needs a clean tree)"
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=$W/bin/buildgate
CPUGATE=$W/bin/cpugate

# ---------------------------------------------------------------- stage 1: CPU build
{
    "$BUILDGATE" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF &&
    "$BUILDGATE" cmake --build build-cpu --target llama-server -j4 &&
    "$BUILDGATE" g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Ivendor \
        -Itools/mtmd -I. codex/bw-np2-driver.cpp build-cpu/tools/server/libserver-context.a build-cpu/common/libllama-common-base.a \
        -Lbuild-cpu/bin -Wl,-rpath,"$ROOT/build-cpu/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base \
        -o build-cpu/bin/bw-np2-driver
} > "$OUT/build-cpu.log" 2>&1 || die 8 "CPU build (see $OUT/build-cpu.log)"
log "CPU build ok"

# ---------------------------------------------------------------- stage 1: np2 matrix
# name:env assignments. ref = stack6 code path; legacy = b115f8180 (indexed seq_rm + hoisted pointer);
# fixv = default behaviour plus the VERIFY cross-checks; mut-* must differ from ref.
declare -a VARIANTS=(
    "ref:LLAMA_HOST_TRIMS_SEQRM=0 LLAMA_HOST_TRIMS_BULKH=0"
    "ref2:LLAMA_HOST_TRIMS_SEQRM=0 LLAMA_HOST_TRIMS_BULKH=0"
    "seqrm:LLAMA_HOST_TRIMS_SEQRM=1 LLAMA_HOST_TRIMS_BULKH=0"
    "bulk:LLAMA_HOST_TRIMS_SEQRM=0 LLAMA_HOST_TRIMS_BULKH=1"
    "fix:"
    "fixv:LLAMA_HOST_TRIMS_VERIFY=1"
    "legacy:LLAMA_HOST_TRIMS_SEQRM=1 LLAMA_HOST_TRIMS_BULKH=2"
    "mut-bulk:LLAMA_HOST_TRIMS_MUTANT=bulk_off"
    "mut-seqrm:LLAMA_HOST_TRIMS_MUTANT=seqrm_skip"
)
for meta in 0 1; do
    for spec in "${VARIANTS[@]}"; do
        name=${spec%%:*}
        assigns=${spec#*:}
        # shellcheck disable=SC2086
        env -u LLAMA_HOST_TRIMS_SEQRM -u LLAMA_HOST_TRIMS_BULKH -u LLAMA_HOST_TRIMS_VERIFY -u LLAMA_HOST_TRIMS_MUTANT \
            -u LLAMA_HOST_TRIMS_PROOF $assigns \
            LD_LIBRARY_PATH="$ROOT/build-cpu/bin" GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
            "$CPUGATE" build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/$meta-$name.json" "$meta" 2 1729 64 \
            > "$OUT/$meta-$name.log" 2>&1 || die 8 "run meta=$meta $name failed (see $OUT/$meta-$name.log)"
        log "ran meta=$meta $name"
    done
done

python3 codex/bw-np2-compare.py "$OUT" 0,1 | tee "$OUT/compare.log"
stage1=${PIPESTATUS[0]}
log "stage 1 verdict rc=$stage1"
case $stage1 in
    0|3) ;;
    *) die "$stage1" "np2 CPU matrix (see $OUT/compare.log); no freeze, no items" ;;
esac

# ---------------------------------------------------------------- stage 2: CUDA build, freeze, items
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
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
    "$W/bin/wt" freeze "$ROOT" host-trims-v2 --build build-cuda \
        --desc "stack6 + host-trims v2 (indexed seq_rm, bulk MTP verify rows read after catch-up, env switches); CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'commit %s\n' "$SHA"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES=""; DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed\n' "$n" "$short"
    } > "$W/frozen/host-trims-v2/RECIPE.txt"
    [[ $(head -1 "$W/frozen/host-trims-v2/SOURCE") == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/host-trims-v2"

python3 codex/bw-make-items.py > "$OUT/items.log" 2>&1 || die 10 "item generation (see $OUT/items.log)"
for f in codex/items/host-trims-v2-np2-iq4-*.gpuq.toml; do
    [[ -e $f ]] || die 10 "no item files generated"
    gpuq submit --check "$f" >> "$OUT/items.log" 2>&1 || die 10 "gpuq submit --check $f (see $OUT/items.log)"
    log "item checked, NOT submitted: $f"
done

log "done stage1 rc=$stage1"
exit "$stage1"
