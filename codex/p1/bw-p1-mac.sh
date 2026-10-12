#!/usr/bin/env bash
# Mac CPU gate for P1 (meta subgraph plans per graph uid), run through bin/bw-mac; argument = OUT dir.
# test-alloc plain + ASAN on the fixed tree, pkg 86's two cases on top (ASAN), the fix reverted (codex/p1/fix-meta.patch -R:
# the reuse case must die), and Meta(CPU,CPU) model identity: fixed meta == plain == unfixed meta.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 ASAN_OPTIONS=detect_leaks=0
export OMP_NUM_THREADS=4
cd "$(dirname "$0")/../.."
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL dirty tracked tree'; exit 9; }
git rev-parse HEAD > "$OUT/source-commit.txt"
: > "$OUT/stages.tsv"
fails=0

flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
asan='-fsanitize=address -fno-omit-frame-pointer'
asan_flags=(-DCMAKE_C_FLAGS="$asan" -DCMAKE_CXX_FLAGS="$asan" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address
    -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address)

configure() { "$BW_BUILDGATE" cmake -S . -B "$1" "${flags[@]}" "${extra[@]}" ${2:+"${asan_flags[@]}"}; }
build() { "$BW_BUILDGATE" cmake --build "$1" --target test-alloc -j"${BW_JOBS:-4}"; }
stage() { # NAME OK(0|1) RC WANT T0
    printf '%s\trc=%s\twant=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$4" "$([[ $2 == 1 ]] && echo PASS || echo FAIL)" "$((SECONDS-$5))" | tee -a "$OUT/stages.tsv"
    [[ $2 == 1 ]] || fails=$((fails+1))
}

# NAME RC-EXPECT(0|nz) BUILD-DIR [GREP...]
run_case() {
    local name=$1 want=$2 dir=$3 rc=0 t0=$SECONDS ok=1 g
    shift 3
    build "$dir" > "$OUT/$name.build.log" 2>&1 || { echo "FAIL build $name"; fails=$((fails+1)); return 0; }
    "$BW_GATE" "$dir/bin/test-alloc" > "$OUT/$name.txt" 2>&1 || rc=$?
    if [[ $want == 0 && $rc != 0 ]]; then ok=0; fi
    if [[ $want == nz && $rc == 0 ]]; then ok=0; fi
    for g in "$@"; do grep -aq -- "$g" "$OUT/$name.txt" || ok=0; done
    stage "$name" "$ok" "$rc" "$want" "$t0"
}

# NAME META(0|1) N_GEN [ENV=VAL...]: Meta(CPU,CPU) (1) or plain CPU (0) greedy tokens; the build is the current tree
run_model() {
    local name=$1 meta=$2 ngen=$3 rc=0 t0=$SECONDS ok=1
    shift 3
    "$BW_BUILDGATE" cmake --build build-plain --target llama -j"${BW_JOBS:-4}" > "$OUT/model-$name.build.log" 2>&1 || ok=0
    "$BW_GATE" clang++ -std=c++17 -O1 -Iinclude -Iggml/include codex/p1/driver.cpp -o build-plain/bin/p1-driver \
        -Lbuild-plain/bin -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/build-plain/bin" >> "$OUT/model-$name.build.log" 2>&1 || ok=0
    env "$@" "$BW_GATE" build-plain/bin/p1-driver "$BW_MODEL" "$meta" "$ngen" > "$OUT/model-$name.txt" 2> "$OUT/model-$name.log" || rc=$?
    [[ $rc == 0 ]] && grep -aq '^seq1:' "$OUT/model-$name.txt" && grep -aq '^embd:' "$OUT/model-$name.txt" || ok=0
    stage "model-$name" "$ok" "$rc" 0 "$t0"
}

configure build-plain
configure build-asan asan

R=test_meta_subgraph_reuse_two_splits
run_case fixed-plain 0 build-plain "$R PASSED"
run_case fixed-asan  0 build-asan  "$R PASSED"
run_model fixed-meta    1 24
run_model fixed-meta-r2 1 24
run_model fixed-cpu     0 24
# split census of the model graphs under Meta(CPU,CPU) (informational)
run_model fixed-meta-sched 1 2 GGML_SCHED_DEBUG=1
grep -a 'SPLIT #' "$OUT/model-fixed-meta-sched.log" | sed 's/:.*//; s/ *#[0-9]*//' | sort | uniq -c > "$OUT/model-split-census.txt" || true

# pkg 86's arena cases, without pkg 86's fix: the plans own their subgraphs, so the stale-arena bug cannot occur
git apply codex/p1/pkg86-tests.patch
run_case fixed-asan-pkg86 0 build-asan "$R PASSED" 'test_meta_subgraph_realloc_count PASSED' 'test_meta_subgraph_realloc_size PASSED'
git apply -R codex/p1/pkg86-tests.patch

# fix reverted: the second compute of the meta, CPU, meta graph rebuilds both splits under new uids
git apply -R codex/p1/fix-meta.patch
run_case unfixed-plain nz build-plain 'test_buft_get_alloc_size_n_does_not_allocate PASSED' "$R" 'compute() == uids'
run_model unfixed-meta    1 24
run_model unfixed-meta-r2 1 24
# master + pkg 86's arena fix: tells whether a P1-vs-master difference is pkg 86's stale-arena bug
git apply codex/p1/pkg86-fix.patch
run_model m86-meta 1 24
git apply -R codex/p1/pkg86-fix.patch
git apply codex/p1/fix-meta.patch
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }

# tokens of the greedy decode must agree everywhere
t0=$SECONDS
ok=1
for f in fixed-meta-r2 fixed-cpu unfixed-meta unfixed-meta-r2 m86-meta; do
    cmp <(grep -a '^seq' "$OUT/model-fixed-meta.txt") <(grep -a '^seq' "$OUT/model-$f.txt") || ok=0
done
stage model-token-identity "$ok" - - "$t0"
# the raw-embedding batch is not repeatable even on plain CPU master (a1/a2: 77 vs 2627), so it is recorded only
for f in fixed-meta fixed-meta-r2 fixed-cpu unfixed-meta unfixed-meta-r2 m86-meta; do
    printf '%s\t%s\n' "$f" "$(grep -a '^embd' "$OUT/model-$f.txt")"
done | tee "$OUT/model-embd.tsv"

echo "p1: fails=$fails"
[[ $fails == 0 ]]
