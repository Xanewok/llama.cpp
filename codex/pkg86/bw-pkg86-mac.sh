#!/usr/bin/env bash
# Mac CPU gate for pkg 86 (run through bin/bw-mac; argument = OUT dir). test-alloc, plain and ASAN, on the fixed tree and with the
# fix reverted (codex/pkg86/fix-meta.patch -R), one test case per binary run so a dead case does not hide the other.
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

# NAME RC-EXPECT(0|nz|any) BUILD-DIR CASE(all|count|size) [GREP...]: run test-alloc with only that case, check the rc and strings
T=tests/test-alloc.cpp
run_case() {
    local name=$1 want=$2 dir=$3 case=$4 rc=0 t0=$SECONDS
    shift 4
    cp "$T" "$OUT/test-alloc.cpp.save"
    [[ $case == count ]] && sed -i.bak '/run("test_meta_subgraph_realloc_size"/d' "$T"
    [[ $case == size ]] && sed -i.bak '/run("test_meta_subgraph_realloc_count"/d' "$T"
    rm -f "$T.bak"
    build "$dir" > "$OUT/$name.build.log" 2>&1 || { echo "FAIL build $name"; cp "$OUT/test-alloc.cpp.save" "$T"; return 1; }
    cp "$OUT/test-alloc.cpp.save" "$T"
    "$BW_GATE" "$dir/bin/test-alloc" > "$OUT/$name.txt" 2>&1 || rc=$?
    local ok=1 g
    if [[ $want == 0 && $rc != 0 ]]; then ok=0; fi
    if [[ $want == nz && $rc == 0 ]]; then ok=0; fi
    for g in "$@"; do grep -aq -- "$g" "$OUT/$name.txt" || ok=0; done
    printf '%s\trc=%s\twant=%s\tgate=%s\twall_s=%s\n' "$name" "$rc" "$want" "$([[ $ok == 1 ]] && echo PASS || echo FAIL)" "$((SECONDS-t0))" | tee -a "$OUT/stages.tsv"
    [[ $ok == 1 ]] || fails=$((fails+1))
}

configure build-plain
configure build-asan asan
build build-plain > "$OUT/build-plain.log" 2>&1
build build-asan > "$OUT/build-asan.log" 2>&1

# fixed tree
run_case fixed-plain-all 0 build-plain all 'test_meta_subgraph_realloc_count PASSED' 'test_meta_subgraph_realloc_size PASSED'
run_case fixed-asan-all  0 build-asan  all 'test_meta_subgraph_realloc_count PASSED' 'test_meta_subgraph_realloc_size PASSED'

# fix reverted: every case must die; ASAN names the stale subgraph, the size case aborts on a full visited hash (ggml_hash_insert, inlined into graph_compute)
git apply -R codex/pkg86/fix-meta.patch
run_case unfixed-asan-count  nz build-asan  count 'ERROR: AddressSanitizer' 'in ggml_backend_meta_graph_compute'
run_case unfixed-asan-size   nz build-asan  size  'ggml-impl.h:309: fatal error' 'ggml_backend_meta_graph_compute'
# plain: the freed arena need not fault on macOS, rc is recorded only
run_case unfixed-plain-count any build-plain count
run_case unfixed-plain-size  nz build-plain size  'fatal error'
git apply codex/pkg86/fix-meta.patch
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }

echo "pkg86: fails=$fails"
[[ $fails == 0 ]]
