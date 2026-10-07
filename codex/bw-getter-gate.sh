#!/usr/bin/env bash
# Build-window payload (gpuq filler item; never submits anything). Usage: bw-getter-gate.sh [--preflight] [OUT_DIR]
#
# Stage 1  CPU build (-DGGML_CUDA=OFF) of test-recurrent-state-rollback; on the 4B fixture (-c 4096):
#            gate    full test (rollback, split replay, getter gate) passes, and the gate probe counts getter nodes only in
#                    single-sequence ubatches and gathered-state nodes only in multi-sequence ones (log line checked, not just rc).
#            mutants each must fail the gate test for its own reason: row0 (LD_PRELOAD, every indexed read takes row 0),
#                    nogate (source: getter at any n_seqs = 743acb55f), invert (source: getter only for n_seqs != 1).
#                    Source mutants edit src/models/delta-net-base.cpp, rebuild, and are always undone (trap + git checkout).
#            restore the unmutated source rebuilt, the gate test passes again, the tree is clean.
# Stage 2  CUDA build (stack6 recipe, CCCL v3.4.3) + 25 DeviceTopK strings + wt freeze gdn-getter-gate + RECIPE.txt,
#          then the np2-busy and np1 items rendered with harness.mkitem and checked with gpuq submit --check (no submit).
#
# Exit codes: 0 all pass | 2 gate test failed on the unmutated build | 3 a mutant escaped (passed, or failed for another reason)
# | 4 restore failed (tree not clean or gate test failed after the undo) | 8 preflight, CPU build or CPU run error
# | 9 CUDA build/freeze/CCCL/RECIPE | 10 item generation or gpuq check.
# 2 3 4 8 stop before stage 2 (no CUDA build, no freeze, no items).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/.." || exit 8
ROOT=$PWD
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
GATE_SRC=src/models/delta-net-base.cpp
GATE_LINE='if (keep && cparams.offload_kqv && n_seqs == 1) {'
log() { echo "bw-getter-gate: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "$W/bin/buildgate" "$W/bin/cpugate" "$W/bin/wt" "$MODEL" "$ROOT/codex/bw-make-items.py" "$ROOT/codex/row0-mutant.cpp" \
             "$W/frozen/stack6/build/bin/round-trace" "$CCCL" "$EVAL/harness/mkitem.py" \
             "$EVAL/harness/experiments/gdn-getter-np2-busy-j5q.toml" "$EVAL/harness/experiments/gdn-getter-np1-cccl-j5q.toml"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    command -v patchelf > /dev/null && command -v gpuq > /dev/null || die 8 "preflight: patchelf or gpuq not on PATH"
    [[ ! -e $W/frozen/gdn-getter-gate ]] || die 8 "preflight: $W/frozen/gdn-getter-gate exists (wt freeze refuses)"
    grep -qF "$GATE_LINE" "$GATE_SRC" || die 8 "preflight: gate line not in $GATE_SRC (the source mutants would not apply)"
    bash -n "$0" || die 8 "preflight: bash -n"
    python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' "$ROOT/codex/bw-make-items.py" || die 8 "preflight: bw-make-items.py syntax"
    exit 0
fi
OUT=$(readlink -m "${1:-$ROOT/codex/bw-results}")
mkdir -p "$OUT"

[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window (wt freeze needs a clean tree)"
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=$W/bin/buildgate
CPUGATE=$W/bin/cpugate
restore() { git checkout -q -- "$GATE_SRC"; }
trap restore EXIT

# ---------------------------------------------------------------- stage 1: CPU build
{
    "$BUILDGATE" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF &&
    "$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback -j4 &&
    "$BUILDGATE" g++ -std=c++17 -O2 -fPIC -shared -Iggml/include codex/row0-mutant.cpp -ldl -o build-cpu/bin/row0-mutant.so
} > "$OUT/build-cpu.log" 2>&1 || die 8 "CPU build (see $OUT/build-cpu.log)"
log "CPU build ok"

TEST=build-cpu/bin/test-recurrent-state-rollback
# run_test NAME ARGS...: log in $OUT/NAME.log, rc in $RC; PRELOAD=lib.so preloads a shim into the test only
run_test() {
    local name=$1; shift
    "$CPUGATE" env LD_PRELOAD="${PRELOAD:-}" LD_LIBRARY_PATH="$ROOT/build-cpu/bin" "$TEST" "$@" -m "$MODEL" -c 4096 -t 8 > "$OUT/$name.log" 2>&1
    RC=$?
}
ENGAGED='test_getter_gate: gdn nodes, one sequence: [1-9][0-9]* by id, 0 gathered; several: 0 by id, [1-9][0-9]* gathered'

run_test gate-full
[[ $RC == 0 ]] || { log "full test rc=$RC (see $OUT/gate-full.log)"; die 2 "full test failed on the unmutated build"; }
grep -q "$ENGAGED" "$OUT/gate-full.log" || die 2 "gate probe line missing or wrong in $OUT/gate-full.log"
run_test gate --getter-gate
[[ $RC == 0 ]] && grep -q "$ENGAGED" "$OUT/gate.log" || die 2 "gate test (see $OUT/gate.log)"
log "gate build: full test and gate probe pass"

# expect_fail NAME REASON: the run in $OUT/NAME.log (rc in $RC) must have failed with a log line matching REASON
declare -a RESULTS=("gate:pass")
expect_fail() {
    if [[ $RC != 0 ]] && grep -q "$2" "$OUT/$1.log"; then
        RESULTS+=("$1:fails")
        log "mutant $1 fails as required (rc=$RC)"
    else
        RESULTS+=("$1:ESCAPED rc=$RC")
        die 3 "mutant $1 escaped or failed for another reason (rc=$RC, expected /$2/; see $OUT/$1.log)"
    fi
}

PRELOAD="$ROOT/build-cpu/bin/row0-mutant.so" run_test mut-row0 --getter-gate
grep -q "ROW0_MUTANT active" "$OUT/mut-row0.log" || die 3 "row0 shim was not loaded (see $OUT/mut-row0.log)"
expect_fail mut-row0 "logits differ from the single-sequence context"

source_mutant() { # NAME SED_REPLACEMENT REASON
    sed -i "s/n_seqs == 1) {/$2) {/" "$GATE_SRC"
    git diff --quiet -- "$GATE_SRC" && die 8 "source mutant $1 did not change $GATE_SRC"
    "$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback -j4 > "$OUT/build-$1.log" 2>&1 || die 8 "mutant $1 build (see $OUT/build-$1.log)"
    run_test "$1" --getter-gate
    restore
    expect_fail "$1" "$3"
}
source_mutant mut-nogate 'true' "multi-sequence ubatches must keep the gathered state"
source_mutant mut-invert 'n_seqs != 1' "single-sequence ubatches must read the state by id"

"$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback -j4 > "$OUT/build-restore.log" 2>&1 || die 4 "restore build (see $OUT/build-restore.log)"
[[ -z $(git status --porcelain --untracked-files=no) && $(git rev-parse HEAD) == "$SHA" ]] || die 4 "tree not clean or HEAD moved after the mutants"
run_test gate-restored --getter-gate
[[ $RC == 0 ]] && grep -q "$ENGAGED" "$OUT/gate-restored.log" || die 4 "gate test failed after the undo (see $OUT/gate-restored.log)"
log "restored; stage 1 passed"

python3 - "$OUT/summary.json" "$SHA" "${RESULTS[@]}" <<'EOF'
import json, sys
out, sha, *results = sys.argv[1:]
json.dump({"commit": sha, "model": "Qwen3.5-4B-UD-Q4_K_XL.gguf -c 4096", "results": dict(r.split(":", 1) for r in results)},
          open(out, "w"), indent=2)
EOF

# ---------------------------------------------------------------- stage 2: CUDA build, freeze, items
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
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
    "$W/bin/wt" freeze "$ROOT" gdn-getter-gate --build build-cuda \
        --desc "stack6 + GDN state getter (743acb55f) + v2 virtgpu guard + gate: getter only for n_seqs == 1, else gather + cache-copy fusion; + GGML_CUDA_DEBUG_GDN_STATE gdn_cache_fusion log; CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'commit %s\n' "$SHA"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES=""; DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed\n' "$n" "$short"
        printf 'base: stack6 ac7465885 + topic/gdn-getter-stack6 743acb55f; recipe flags identical to frozen/stack6/RECIPE.txt\n'
    } > "$W/frozen/gdn-getter-gate/RECIPE.txt"
    [[ $(head -1 "$W/frozen/gdn-getter-gate/SOURCE") == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/gdn-getter-gate"

python3 codex/bw-make-items.py > "$OUT/items.log" 2>&1 || die 10 "item generation (see $OUT/items.log)"
for f in codex/items/gdn-getter-gate-*.gpuq.toml; do
    [[ -e $f ]] || die 10 "no item files generated"
    gpuq submit --check "$f" >> "$OUT/items.log" 2>&1 || die 10 "gpuq submit --check $f (see $OUT/items.log)"
    log "item checked, NOT submitted: $f"
done

log "done"
exit 0
