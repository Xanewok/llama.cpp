#!/usr/bin/env bash
# Mac CPU gate for the pkg 91 capacity fix (plans of the current allocation, not 8 LRU), run through bin/bw-mac; argument = OUT dir.
# test-alloc plain + ASAN on the fixed tree (2 and 10 Meta splits, one split allocated again); the capacity fix reverted
# (codex/p1/fix-capacity.patch -R, i.e. back to the 8-plan LRU): the 10-split case must die, the 2-split case pass.
# Meta(CPU,CPU) model identity: fixed meta == fixed meta again == plain CPU == reverted meta.
# Bounds: waits <= 10 min for >= 4 GiB available memory (rc 75 "skipped: RAM"), every step <= 10 min (SIGALRM, rc 142).
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

avail_gib() { vm_stat | awk -v ps="$(pagesize)" '/Pages (free|inactive|speculative)/ { gsub("\\.", "", $NF); n += $NF } END { printf "%d", n*ps/1073741824 }'; }
t0=$SECONDS
until (( $(avail_gib) >= 4 )); do
    (( SECONDS - t0 < 600 )) || { echo "skipped: RAM ($(avail_gib) GiB available after 10 min)" | tee "$OUT/ram-gate.txt"; exit 75; }
    sleep 15
done
echo "ram gate: $(avail_gib) GiB available after $((SECONDS - t0)) s" | tee "$OUT/ram-gate.txt"

bounded() { perl -e 'alarm shift; exec @ARGV or die "exec: $!"' 600 "$@"; }

flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
asan='-fsanitize=address -fno-omit-frame-pointer'
asan_flags=(-DCMAKE_C_FLAGS="$asan" -DCMAKE_CXX_FLAGS="$asan" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address
    -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address)

configure() { bounded "$BW_BUILDGATE" cmake -S . -B "$1" "${flags[@]}" "${extra[@]}" ${2:+"${asan_flags[@]}"}; }
build() { bounded "$BW_BUILDGATE" cmake --build "$1" --target test-alloc -j"${BW_JOBS:-4}"; }
stage() { # NAME OK(0|1) RC WANT T0
    printf '%s\trc=%s\twant=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$4" "$([[ $2 == 1 ]] && echo PASS || echo FAIL)" "$((SECONDS-$5))" | tee -a "$OUT/stages.tsv"
    [[ $2 == 1 ]] || fails=$((fails+1))
}

# NAME RC-EXPECT(0|nz) BUILD-DIR [GREP...]; a GREP starting with ! must be absent
run_case() {
    local name=$1 want=$2 dir=$3 rc=0 t0=$SECONDS ok=1 g
    shift 3
    build "$dir" > "$OUT/$name.build.log" 2>&1 || { echo "FAIL build $name"; fails=$((fails+1)); return 0; }
    bounded "$BW_GATE" "$dir/bin/test-alloc" > "$OUT/$name.txt" 2>&1 || rc=$?
    if [[ $want == 0 && $rc != 0 ]]; then ok=0; fi
    if [[ $want == nz && $rc == 0 ]]; then ok=0; fi
    for g in "$@"; do
        if [[ $g == '!'* ]]; then
            grep -aq -- "${g#!}" "$OUT/$name.txt" && ok=0
        else
            grep -aq -- "$g" "$OUT/$name.txt" || ok=0
        fi
    done
    stage "$name" "$ok" "$rc" "$want" "$t0"
}

# NAME META(0|1) N_GEN: Meta(CPU,CPU) (1) or plain CPU (0) greedy tokens; the build is the current tree
run_model() {
    local name=$1 meta=$2 ngen=$3 rc=0 t0=$SECONDS ok=1
    bounded "$BW_BUILDGATE" cmake --build build-plain --target llama -j"${BW_JOBS:-4}" > "$OUT/model-$name.build.log" 2>&1 || ok=0
    bounded "$BW_GATE" clang++ -std=c++17 -O1 -Iinclude -Iggml/include codex/p1/driver.cpp -o build-plain/bin/p1-driver \
        -Lbuild-plain/bin -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/build-plain/bin" >> "$OUT/model-$name.build.log" 2>&1 || ok=0
    bounded "$BW_GATE" build-plain/bin/p1-driver "$BW_MODEL" "$meta" "$ngen" > "$OUT/model-$name.txt" 2> "$OUT/model-$name.log" || rc=$?
    [[ $rc == 0 ]] && grep -aq '^seq1:' "$OUT/model-$name.txt" || ok=0
    stage "model-$name" "$ok" "$rc" 0 "$t0"
}

configure build-plain > "$OUT/configure-plain.log" 2>&1
configure build-asan asan > "$OUT/configure-asan.log" 2>&1

R2=test_meta_subgraph_reuse_two_splits
R10=test_meta_subgraph_reuse_ten_splits
RS=test_meta_subgraph_reuse_same_graph
run_case fixed-plain 0 build-plain "$R2 PASSED" "$R10 PASSED" "$RS PASSED"
run_case fixed-asan  0 build-asan  "$R2 PASSED" "$R10 PASSED" "$RS PASSED"
run_model fixed-meta    1 24
run_model fixed-meta-r2 1 24
run_model fixed-cpu     0 24

# capacity fix reverted (8-plan LRU): the 10-split case rebuilds every split on every compute, under new uids
git apply -R codex/p1/fix-capacity.patch
run_case reverted-plain nz build-plain "$R2 PASSED" "$R10" 'compute() == uids' "!$R10 PASSED"
run_model reverted-meta 1 24
git apply codex/p1/fix-capacity.patch
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }

t0=$SECONDS
ok=1
for f in fixed-meta-r2 fixed-cpu reverted-meta; do
    cmp -s <(grep -a '^seq' "$OUT/model-fixed-meta.txt") <(grep -a '^seq' "$OUT/model-$f.txt") || { echo "tokens differ: $f"; ok=0; }
done
stage model-token-identity "$ok" - - "$t0"

echo "p1-cap: fails=$fails"
[[ $fails == 0 ]]
