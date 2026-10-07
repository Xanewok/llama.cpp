#!/usr/bin/env bash
# Build-window payload (gpuq filler item; never submits anything). Usage: bw-graph-capture-fix.sh [--preflight | OUT_DIR]
#
# Fix under test: 346505678 (the meta plan arena creates cgraph_main for all max_subgraphs, not only this graph's
# n_subgraphs). The 32k depth run of the unfixed spike segfaulted on a stale cgraph_main.
#
# Stage 1  CPU: build ggml, compile tests/test-meta-plan-arena.cpp against it and run it (it covers GGML_META_AR_NODE=0 and
#          =1 in forked children; the outer env is set to both as well). Negative control: the same test against a copy of
#          HEAD with 346505678 reverted must die on a signal. Then the CMake registration of the test is built and run
#          through ctest. Writes OUT/cpu-evidence.json (pinned by the GPU item).
# Stage 2  CUDA build with the stack6 recipe flags, CCCL check, round-trace copy, wt freeze graph-capture-fix, RECIPE.txt.
# Stage 3  Render the depth A/B item (stack6 vs the frozen fix, arms of the spike's depth experiment) and gpuq submit --check.
#
# Exit codes: 0 all pass | 2 the test fails on the fixed tree | 3 the negative control did NOT fail (the test does not reach
# the defect: no freeze, no item) | 4 negative control setup or build error | 5 the negative control failed for another
# reason than a signal | 8 preflight, dirty tree or CPU build/run error | 9 CUDA build/freeze/CCCL | 10 item generation
# or check. 2 3 4 5 8 stop before stage 2 (no freeze, no item).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
cd "$(dirname "$(readlink -f "$0")")/.." || exit 8
ROOT=$PWD
FIX=346505678
STACK6=ac7465885dc9f13de396ccd91b4be5eaffde3730
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
NEG=$W/work/graph-capture-fix-negctl
log() { echo "bw-graph-capture-fix: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "$W/bin/buildgate" "$W/bin/cpugate" "$W/bin/wt" "$ROOT/codex/bw-make-item.py" "$ROOT/codex/depth-experiment.toml" \
             "$ROOT/tests/test-meta-plan-arena.cpp" "$W/frozen/stack6/build/bin/round-trace" "$CCCL" /usr/local/cuda-13.2/bin/nvcc \
             /home/xanewok/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf \
             /home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    for t in gpuq cmake ctest g++ patch strings git tar; do command -v "$t" > /dev/null || die 8 "preflight: $t not in PATH"; done
    command -v patchelf > /dev/null || [[ -x ${PATCHELF:-} ]] || die 8 "preflight: wt freeze needs patchelf"
    bash -n "$0" || die 8 "preflight: bash -n"
    python3 -c 'import sys; compile(open(sys.argv[1]).read(), sys.argv[1], "exec")' "$ROOT/codex/bw-make-item.py" || die 8 "preflight: bw-make-item.py syntax"
    git merge-base --is-ancestor "$FIX" HEAD || die 8 "preflight: $FIX is not an ancestor of HEAD"
    git rev-parse --verify -q "$STACK6^{commit}" > /dev/null || die 8 "preflight: stack6 $STACK6 missing"
    git show "$FIX" -- ggml/src/ggml-backend-meta.cpp | git apply -R --check || die 8 "preflight: $FIX does not revert cleanly on HEAD"
    [[ ! -e $W/frozen/graph-capture-fix ]] || die 8 "preflight: $W/frozen/graph-capture-fix exists"
    free_gib=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
    [[ $free_gib -ge 12 ]] || die 8 "preflight: /home has ${free_gib} GiB free, need 12 (CPU + negative control + CUDA builds)"
    exit 0
fi
OUT=$(readlink -m "${1:-$ROOT/codex/bw-results}")
mkdir -p "$OUT"

[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window (wt freeze needs a clean tree)"
bash "$0" --preflight || exit $?
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=$W/bin/buildgate
CPUGATE=$W/bin/cpugate
export CUDA_HOME=/usr/local/cuda-13.2

# ---------------------------------------------------------------- stage 1: CPU
cpu_flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF)

# build_ggml SRC BUILD_DIR LOG: ggml only, then the test compiled directly against its shared libraries.
build_ggml_test() {
    local src=$1 build=$2
    (
        cd "$src" || exit 1
        "$BUILDGATE" cmake -B "$build" "${cpu_flags[@]}" -DLLAMA_BUILD_TESTS=OFF &&
        "$BUILDGATE" cmake --build "$build" --target ggml ggml-base ggml-cpu -j4 &&
        "$BUILDGATE" g++ -std=c++17 -O1 -g -Wall -Wextra -Iggml/include tests/test-meta-plan-arena.cpp \
            -L"$build/bin" -Wl,-rpath,"$src/$build/bin" -lggml -lggml-cpu -lggml-base -o "$build/bin/test-meta-plan-arena"
    )
}

# run_test BIN LABEL: both outer GGML_META_AR_NODE values; the test sets the variable itself per child as well.
run_test() {
    local bin=$1 label=$2 rc=0 v
    for v in 0 1; do
        env GGML_META_AR_NODE=$v "$CPUGATE" "$bin" > "$OUT/$label-$v.log" 2>&1
        rc=$?
        log "$label GGML_META_AR_NODE=$v rc=$rc: $(tr '\n' ' ' < "$OUT/$label-$v.log" | tail -c 300)"
        if [[ $label == test ]]; then
            [[ $rc == 0 && $(grep -c PASSED "$OUT/$label-$v.log") == 2 ]] || return 2
        else
            [[ $rc != 0 ]] || return 3
            # Both children must die on a signal; an ordinary GGML_ASSERT abort is not the defect.
            [[ $(grep -c 'FAILED: killed by signal' "$OUT/$label-$v.log") == 2 ]] || return 5
            ! grep -q 'GGML_ASSERT' "$OUT/$label-$v.log" || return 5
        fi
    done
    return 0
}

build_ggml_test "$ROOT" build-cpu > "$OUT/build-cpu.log" 2>&1 || die 8 "CPU build or test compile (see $OUT/build-cpu.log)"
log "CPU build ok"
run_test "$ROOT/build-cpu/bin/test-meta-plan-arena" test || die 2 "test-meta-plan-arena fails on the fixed tree (see $OUT/test-*.log)"

# Negative control: a plain copy of HEAD with the fix reverted. The copy has no .git; the revert is the inverse of the fix commit.
if [[ ! -e $NEG/PURPOSE ]]; then
    "$W/bin/wt" work graph-capture-fix-negctl --purpose "bw-graph-capture-fix.sh negative control: HEAD with $FIX reverted, ggml CPU build + test" \
        > /dev/null || die 4 "wt work for the negative control"
fi
[[ -d $NEG ]] || die 4 "no $NEG"
rm -rf "${NEG:?}/src"
{ mkdir -p "$NEG/src" && git archive HEAD | tar -x -C "$NEG/src"; } || die 4 "negative control tree setup"
git show --format= "$FIX" -- ggml/src/ggml-backend-meta.cpp > "$NEG/fix.patch" || die 4 "reading $FIX"
(cd "$NEG/src" && patch -R -p1 < "$NEG/fix.patch") > "$OUT/negctl-revert.log" 2>&1 ||
    die 4 "reverting $FIX in the negative control copy (see $OUT/negctl-revert.log)"
grep -q 'for (size_t i = 0; i < n_subgraphs; i++) {' "$NEG/src/ggml/src/ggml-backend-meta.cpp" || die 4 "revert did not restore the n_subgraphs loop bound"
! grep -q 'All max_subgraphs, not this call' "$NEG/src/ggml/src/ggml-backend-meta.cpp" || die 4 "fix comment still present after the revert"
build_ggml_test "$NEG/src" build-cpu > "$OUT/build-negctl.log" 2>&1 || die 4 "negative control build (see $OUT/build-negctl.log)"
run_test "$NEG/src/build-cpu/bin/test-meta-plan-arena" negctl
case $? in
    0) ;;
    3) die 3 "negative control passed: the test does not reach the stale cgraph_main (no freeze, no item; see $OUT/negctl-*.log)" ;;
    *) die 5 "negative control failed without 'killed by signal' (see $OUT/negctl-*.log)" ;;
esac
log "negative control dies on a signal, as required"

# The CMake registration (llama_build_and_test) of the test.
{
    "$BUILDGATE" cmake -B build-cpu "${cpu_flags[@]}" -DLLAMA_BUILD_TESTS=ON &&
    "$BUILDGATE" cmake --build build-cpu --target test-meta-plan-arena -j4 &&
    "$CPUGATE" ctest --test-dir build-cpu -R '^test-meta-plan-arena$' --output-on-failure
} > "$OUT/ctest.log" 2>&1 || die 8 "CMake test target or ctest (see $OUT/ctest.log)"
grep -q "100% tests passed" "$OUT/ctest.log" || die 8 "ctest did not report 1 passing test (see $OUT/ctest.log)"
log "ctest ok"

python3 - "$OUT" "$SHA" "$FIX" <<'EOF' || die 8 "writing cpu-evidence.json"
import hashlib, json, pathlib, sys
out, sha, fix = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
test = pathlib.Path("tests/test-meta-plan-arena.cpp")
json.dump({
    "commit": sha, "fix_commit": fix, "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
    "fixed_tree": "passes with GGML_META_AR_NODE=0 and 1 (split and ar_node sequences in forked children)",
    "negative_control": "HEAD with the fix reverted: killed by signal with GGML_META_AR_NODE=0 and 1",
    "logs": {p.name: p.read_text() for p in sorted(out.glob("test-*.log")) + sorted(out.glob("negctl-*.log"))},
}, open(out / "cpu-evidence.json", "w"), indent=1)
EOF
log "stage 1 passed"

# ---------------------------------------------------------------- stage 2: CUDA build, freeze
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_stage() {
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}" || return 1
    "$BUILDGATE" cmake --build build-cuda --target llama-server -j4 || return 1
    [[ $(git rev-parse HEAD) == "$SHA" ]] || { echo "HEAD moved during the build"; return 1; }
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 1
    local n; n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "CCCL DeviceTopK strings: $n"
    [[ $n == 25 ]] || { echo "expected 25 DeviceTopK strings (CCCL v3.4.3)"; return 1; }
    local short; short=$(git rev-parse --short HEAD)
    strings build-cuda/bin/libggml-base.so | grep -qx "$short" || { echo "ggml stamp $short not in libggml-base.so"; return 1; }
    "$W/bin/wt" freeze "$ROOT" graph-capture-fix --build build-cuda \
        --desc "stack6 + in-graph AllReduce node (spike) + arena fix 346505678; CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'commit %s (stack6 %s + 5537a7f14 + df61743ce spike + %s arena fix + test commits)\n' "$SHA" "$STACK6" "$FIX"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES=""; DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed\n' "$n" "$short"
        printf 'round-trace = cp of llama-server (harness arms must not execute a file named llama-server); listed in SHA256SUMS\n'
    } > "$W/frozen/graph-capture-fix/RECIPE.txt"
    [[ $(head -1 "$W/frozen/graph-capture-fix/SOURCE") == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/graph-capture-fix"

# ---------------------------------------------------------------- stage 3: item
python3 codex/bw-make-item.py "$OUT" > "$OUT/items.log" 2>&1 || die 10 "item generation (see $OUT/items.log)"
item=codex/items/graph-capture-fix-depth-j5q.gpuq.toml
[[ -e $item ]] || die 10 "no item file generated"
gpuq submit --check "$item" >> "$OUT/items.log" 2>&1 || die 10 "gpuq submit --check $item (see $OUT/items.log)"
log "item checked, NOT submitted: $item"

log "done"
exit 0
