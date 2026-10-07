#!/usr/bin/env bash
# CPU gates for GGML_META_AR_NODE on stack7 (payload for bin/bw-mac; never runs a CUDA binary). Usage: cpu-gates.sh OUT
#
# 1  CPU builds: HEAD, and base = HEAD with code-vs-stack7.diff reversed (its touched files must hash to stack7-blobs.txt;
#    bundles carry no history, so stack7 is rebuilt from the diff). The server_context fixture is linked against each.
# 2  test-ar-node-plan (pure planner) + AR_PLAN_WRONG_EXPECTATION=1 negative control (must fail).
# 3  test-alloc, test-sampling (HEAD); test-meta-plan-arena under GGML_META_AR_NODE=0/1 on HEAD, and as negative control
#    against the base ggml (no arena fix) where both children must die on a signal. Darwin uses libgmalloc so the freed
#    arena is unmapped (glibc builds use the test's own mallopt).
# 4  cpu_identity.py: base vs HEAD env unset / GGML_META_AR_NODE=1, Meta(CPU,CPU) np1 (plain, chain+fold) and np2,
#    plain CPU; forced-node stub abort; perturbation control.
# Exit: 0 pass | 2 a gate failed | 3 a negative control did not fail | 4 identity skipped: RAM (< 4 GiB for 10 min)
# | 5 identity skipped: another model process for 10 min | 8 build or setup. 4 and 5 are skips, never passes.
set -uo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
OUT=${1:?OUT}
cd "$(dirname "$0")/../.." || exit 8
ROOT=$PWD
D=codex/graph-capture
MODEL=${BW_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
BUILDGATE=${BW_BUILDGATE:-$HOME/repos/qwen38-eval-worktrees/bin/buildgate}
CPUGATE=${BW_GATE:-$HOME/repos/qwen38-eval-worktrees/bin/cpugate}
JOBS=${BW_JOBS:-4}
mkdir -p "$OUT"
log() { echo "cpu-gates: $(date '+%H:%M:%S') $*" | tee -a "$OUT/RESULTS.txt"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }
for t in cmake c++ git python3; do command -v "$t" > /dev/null || die 8 "missing tool $t"; done
[[ -f $MODEL ]] || die 8 "missing model $MODEL"
log "HEAD $(git rev-parse HEAD) host $(uname -sm) model $MODEL"

extra=()
read -r -a extra <<< "${BW_CPU_EXTRA:-}"
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF ${extra[@]+"${extra[@]}"})
build_tree() { # SRC BUILD TARGETS...
    local src=$1 build=$2; shift 2
    (cd "$src" &&
        "$BUILDGATE" cmake -B "$build" "${flags[@]}" &&
        "$BUILDGATE" cmake --build "$build" --target "$@" -j"$JOBS" &&
        "$BUILDGATE" c++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include \
            -Ivendor -Itools/mtmd -I. "$ROOT/$D/ar-node-server-cpu.cpp" "$build/tools/server/libserver-context.a" \
            "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$src/$build/bin" \
            -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/ar-node-server-cpu")
}

# ---- 1: builds
build_tree "$ROOT" build-cpu llama-server test-alloc test-sampling test-meta-plan-arena > "$OUT/build-head.log" 2>&1 ||
    die 8 "HEAD build (see $OUT/build-head.log)"
# outside the repo: git apply in a repo subdirectory would resolve paths against the repo root
BASE=${BW_BASE_DIR:-$(dirname "$OUT")/base-src}
[[ $BASE != "$ROOT"/* ]] || die 8 "BW_BASE_DIR must be outside the tree"
rm -rf "$BASE" && mkdir -p "$BASE" || die 8 "base dir"
git archive HEAD | tar -x -C "$BASE" || die 8 "base archive"
(cd "$BASE" && git apply -R "$ROOT/$D/code-vs-stack7.diff") > "$OUT/base-revert.log" 2>&1 ||
    die 8 "reverse diff (see $OUT/base-revert.log)"
while read -r blob path; do
    if [[ $blob == absent ]]; then
        [[ ! -s $BASE/$path ]] || die 8 "base still has $path"
        rm -f "$BASE/$path"
    else
        [[ $(git hash-object "$BASE/$path") == "$blob" ]] || die 8 "base $path does not hash to stack7 $blob"
    fi
done < "$D/stack7-blobs.txt"
! grep -rq meta_ar_node "$BASE/ggml/src" || die 8 "base still mentions meta_ar_node"
log "base tree = stack7 for every touched file (blob hashes match)"
build_tree "$BASE" build-cpu llama-server > "$OUT/build-base.log" 2>&1 || die 8 "base build (see $OUT/build-base.log)"
log "builds ok"

# ---- 2: planner unit test + wrong-expectation control
c++ -std=c++17 -O1 -Iggml/src "$D/test-ar-node-plan.cpp" -o build-cpu/bin/test-ar-node-plan > "$OUT/plan-build.log" 2>&1 ||
    die 8 "planner test build"
"$CPUGATE" build-cpu/bin/test-ar-node-plan > "$OUT/plan.log" 2>&1 || die 2 "test-ar-node-plan (see $OUT/plan.log)"
AR_PLAN_WRONG_EXPECTATION=1 "$CPUGATE" build-cpu/bin/test-ar-node-plan > "$OUT/plan-wrong.log" 2>&1 &&
    die 3 "planner negative control passed"
log "planner: $(tail -1 "$OUT/plan.log"); wrong-expectation control fails as it must"

# ---- 3: ggml tests + arena negative control
"$CPUGATE" build-cpu/bin/test-alloc > "$OUT/test-alloc.log" 2>&1 || die 2 "test-alloc (see $OUT/test-alloc.log)"
"$CPUGATE" build-cpu/bin/test-sampling > "$OUT/test-sampling.log" 2>&1 || die 2 "test-sampling (see $OUT/test-sampling.log)"
# DYLD_* is stripped when a SIP binary (bash, nice) starts, so env sets it right before the test binary
pre=(env)
[[ $(uname) != Darwin ]] || pre=(env DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib)
c++ -std=c++17 -O1 -g -Iggml/include tests/test-meta-plan-arena.cpp -L"$BASE/build-cpu/bin" -Wl,-rpath,"$BASE/build-cpu/bin" \
    -lggml -lggml-cpu -lggml-base -o build-cpu/bin/test-meta-plan-arena-on-base > "$OUT/arena-base-build.log" 2>&1 ||
    die 8 "arena test against the base ggml"
for v in 0 1; do
    "$CPUGATE" "${pre[@]}" GGML_META_AR_NODE=$v build-cpu/bin/test-meta-plan-arena > "$OUT/arena-head-$v.log" 2>&1 ||
        die 2 "test-meta-plan-arena HEAD GGML_META_AR_NODE=$v (see $OUT/arena-head-$v.log)"
    [[ $(grep -c PASSED "$OUT/arena-head-$v.log") == 2 ]] || die 2 "arena HEAD $v: expected 2 PASSED"
    "$CPUGATE" "${pre[@]}" GGML_META_AR_NODE=$v build-cpu/bin/test-meta-plan-arena-on-base > "$OUT/arena-base-$v.log" 2>&1 &&
        die 3 "arena negative control passed on the base (no fix), GGML_META_AR_NODE=$v"
    [[ $(grep -c 'FAILED: killed by signal' "$OUT/arena-base-$v.log") == 2 ]] ||
        die 3 "arena negative control did not die on a signal twice (see $OUT/arena-base-$v.log)"
done
log "test-alloc, test-sampling pass; arena: HEAD passes 0/1, base dies on a signal 0/1 ($(grep -m1 'killed by signal' "$OUT/arena-base-0.log"))"

# ---- 4: end-to-end identity
"$CPUGATE" python3 "$D/cpu_identity.py" "$BASE/build-cpu/bin/ar-node-server-cpu" "$ROOT/build-cpu/bin/ar-node-server-cpu" \
    "$MODEL" "$OUT/identity" > "$OUT/identity-console.log" 2>&1
case $? in
    0) ;;
    4) log "skipped: RAM (identity stage; earlier stages passed)"; exit 4 ;;
    5) log "skipped: busy (identity stage; another model process held the Mac; earlier stages passed)"; exit 5 ;;
    *) die 2 "identity (see $OUT/identity/identity.log)" ;;
esac
grep -E '^(PASS|SUMMARY|  rows=)' "$OUT/identity/identity.log" >> "$OUT/RESULTS.txt"
log "PASS all CPU gates"
exit 0
