#!/usr/bin/env bash
# Mac CPU gate of one pre-PR package on master (run through bin/bw-mac; argument = OUT dir; package name in codex/gate/PKG).
# Fail-before = production hunks reverted from codex/gate/prod*.patch (or a mutant), test kept; pass-after = the tree as committed.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 ASAN_OPTIONS=detect_leaks=0
cd "$(dirname "$0")/../.."
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL dirty tracked tree'; exit 9; }
git rev-parse HEAD > "$OUT/source-commit.txt"
: > "$OUT/stages.tsv"
fails=0
G=codex/gate
PKG=$(cat $G/PKG)

flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
if [[ $PKG == 88 ]]; then
    flags=("${flags[@]/-DLLAMA_BUILD_TOOLS=OFF/-DLLAMA_BUILD_TOOLS=ON}")
    flags=("${flags[@]/-DLLAMA_BUILD_SERVER=OFF/-DLLAMA_BUILD_SERVER=ON}")
fi
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
B=build-cpu

stage() { # NAME VERDICT(PASS|FAIL|INFO) RC T0
    printf '%s\trc=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$2" "$((SECONDS-$4))" | tee -a "$OUT/stages.tsv"
    [[ $2 != FAIL ]] || fails=$((fails+1))
}
# RAM gate: refuse to start below 4 GiB available
free_gib() { vm_stat | awk -v ps="$(pagesize)" '/Pages (free|inactive|speculative)/ { gsub("\\.", "", $NF); n += $NF } END { printf "%d", n*ps/1073741824 }'; }
g=$(free_gib); echo "free_gib=$g" > "$OUT/ram-gate.txt"
[[ $g -ge 4 ]] || { stage ram-gate FAIL "$g" "$SECONDS"; exit 8; }
configure() { "$BW_BUILDGATE" cmake -S . -B "$B" "${flags[@]}" "${extra[@]}" "$@"; }
build() { "$BW_BUILDGATE" cmake --build "$B" --target "$@" -j"${BW_JOBS:-4}"; }
# NAME TARGETS...
do_build() {
    local name=$1 t0=$SECONDS rc=0
    shift
    build "$@" > "$OUT/$name.log" 2>&1 || rc=$?
    stage "$name" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
    [[ $rc == 0 ]] || exit 8
}
# NAME EXPECT(0|nz|any) [GREP-STRING...] -- CMD...; 600 s cap; EXPECT nz = must exit nonzero; strings must be present (EXPECT 0) in stdout+stderr
run() {
    local name=$1 expect=$2 rc=0 t0=$SECONDS v ok=1 g
    shift 2
    local greps=()
    while [[ $1 != -- ]]; do greps+=("$1"); shift; done
    shift
    "$BW_GATE" perl -e 'alarm 600; exec @ARGV' "$@" > "$OUT/$name.txt" 2>&1 || rc=$?
    if [[ $expect == 0 ]]; then [[ $rc == 0 ]] || ok=0; elif [[ $expect == nz ]]; then [[ $rc != 0 ]] || ok=0; fi
    for g in ${greps[@]+"${greps[@]}"}; do grep -aq -- "$g" "$OUT/$name.txt" || ok=0; done
    stage "$name" "$([[ $ok == 1 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
}
# stage the absence of a string (negative control: the fixed marker must not appear)
absent() { # NAME FILE STRING
    local t0=$SECONDS
    stage "$1" "$(grep -aq -- "$3" "$OUT/$2.txt" && echo FAIL || echo PASS)" - "$t0"
}
revert() { git apply -R "$G/$1"; }
restore() { git checkout -q -- . ; [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }; }

case $PKG in
86)
    # ASAN and plain, one case per run so a dead case does not hide the other
    asan='-fsanitize=address -fno-omit-frame-pointer'
    asan_flags=(-DCMAKE_C_FLAGS="$asan" -DCMAKE_CXX_FLAGS="$asan" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address
        -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address)
    t0=$SECONDS rc=0
    { "$BW_BUILDGATE" cmake -S . -B build-plain "${flags[@]}" "${extra[@]}" &&
      "$BW_BUILDGATE" cmake -S . -B build-asan "${flags[@]}" "${extra[@]}" "${asan_flags[@]}" &&
      "$BW_BUILDGATE" cmake --build build-plain --target test-alloc -j"${BW_JOBS:-4}" &&
      "$BW_BUILDGATE" cmake --build build-asan --target test-alloc -j"${BW_JOBS:-4}"; } > "$OUT/build.log" 2>&1 || rc=$?
    stage build "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
    [[ $rc == 0 ]] || exit 8
    T=tests/test-alloc.cpp
    case_run() { # NAME EXPECT DIR CASE(all|count|size) GREP...
        local name=$1 expect=$2 dir=$3 c=$4
        shift 4
        cp "$T" "$OUT/test-alloc.cpp.save"
        [[ $c == count ]] && sed -i.bak '/run("test_meta_subgraph_realloc_size"/d' "$T"
        [[ $c == size ]] && sed -i.bak '/run("test_meta_subgraph_realloc_count"/d' "$T"
        rm -f "$T.bak"
        "$BW_BUILDGATE" cmake --build "$dir" --target test-alloc -j"${BW_JOBS:-4}" > "$OUT/$name.build.log" 2>&1 || { cp "$OUT/test-alloc.cpp.save" "$T"; echo "FAIL build $name"; exit 8; }
        cp "$OUT/test-alloc.cpp.save" "$T"
        run "$name" "$expect" "$@" -- "$dir/bin/test-alloc"
    }
    case_run fixed-plain-all 0 build-plain all 'test_meta_subgraph_realloc_count PASSED' 'test_meta_subgraph_realloc_size PASSED'
    case_run fixed-asan-all  0 build-asan  all 'test_meta_subgraph_realloc_count PASSED' 'test_meta_subgraph_realloc_size PASSED'
    revert prod.patch
    case_run unfixed-asan-count  nz build-asan  count 'ERROR: AddressSanitizer' 'in ggml_backend_meta_graph_compute'
    case_run unfixed-asan-size   nz build-asan  size  'ggml-impl.h:309: fatal error' 'ggml_backend_meta_graph_compute'
    # plain: the freed arena need not fault on macOS; rc recorded, verdict not asserted
    case_run unfixed-plain-count any build-plain count
    case_run unfixed-plain-size  nz build-plain size  'fatal error'
    restore
    ;;
87)
    configure
    do_build build test-llama-archs
    for s in 1 2 3 4 5; do run "fixed-s$s" 0 'MTP draft state reset: OK' -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$s"; done
    run fixed-all-archs 0 -- "$B/bin/test-llama-archs"
    revert prod.patch
    do_build unfixed-build test-llama-archs
    for s in 1 2 3 4 5; do
        run "unfixed-s$s" nz -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$s"
        absent "unfixed-s$s-no-ok-line" "unfixed-s$s" 'MTP draft state reset: OK'
    done
    restore
    ;;
01-03)
    configure
    do_build build llama test-backend-sampler test-sampling
    S=("$B/bin/test-backend-sampler" --model "$BW_MODEL" --device cpu)
    run test-sampling 0 -- "$B/bin/test-sampling"
    run fixed-full 0 -- "${S[@]}"
    run fixed-min_p_one 0 'backend min-p p >= 1 test PASSED' -- "${S[@]}" --test test_min_p_one
    run fixed-min_keep 0 'backend min_keep sampling test PASSED' -- "${S[@]}" --test test_min_keep
    run fixed-logit_bias_candidates 0 'backend logit bias on candidates test PASSED' -- "${S[@]}" --test test_logit_bias_candidates
    # stepwise reverts: 01 -> 03 -> 02 (each patch is that commit's src/llama-sampler.cpp hunks)
    revert prod-01.patch
    do_build unfixed-01-build llama test-backend-sampler
    run unfixed-01-logit_bias_candidates nz -- "${S[@]}" --test test_logit_bias_candidates
    run unfixed-01-control-min_keep 0 'backend min_keep sampling test PASSED' -- "${S[@]}" --test test_min_keep
    run unfixed-01-control-min_p_one 0 'backend min-p p >= 1 test PASSED' -- "${S[@]}" --test test_min_p_one
    revert prod-03.patch
    do_build unfixed-03-build llama test-backend-sampler
    run unfixed-03-min_keep nz -- "${S[@]}" --test test_min_keep
    run unfixed-03-control-min_p_one 0 'backend min-p p >= 1 test PASSED' -- "${S[@]}" --test test_min_p_one
    revert prod-02.patch
    do_build unfixed-02-build llama test-backend-sampler
    run unfixed-02-min_p_one nz -- "${S[@]}" --test test_min_p_one
    restore
    ;;
88)
    configure
    do_build build test-speculative server-context test-arg-parser
    run fixed-test-speculative 0 -- "$B/bin/test-speculative"
    for t in ngram-simple ngram-map-k ngram-map-k4v ngram-mod ngram-cache; do
        t0=$SECONDS
        stage "uses_prompt-$t" "$(grep -aq "^$t: uses_prompt" "$OUT/fixed-test-speculative.txt" && echo PASS || echo FAIL)" - "$t0"
    done
    run fixed-test-arg-parser 0 -- "$B/bin/test-arg-parser"
    # mutant: ngram_simple reads the prompt but does not declare it
    python3 - common/speculative.cpp <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.index('struct common_speculative_impl_ngram_simple')
old = 'bool uses_prompt() const override { return true; }'
j = s.index(old, i)
open(p, 'w').write(s[:j] + old.replace('true', 'false') + s[j + len(old):])
PY
    git diff --stat > "$OUT/mutant.diffstat"
    do_build mutant-build test-speculative
    run mutant-test-speculative nz 'CHECK failed' -- "$B/bin/test-speculative"
    restore
    ;;
82)
    configure
    do_build build test-backend-ops
    run fixed-clean 0 'tests passed' -- "$B/bin/test-backend-ops" test -o TOP_K -b CPU
    # mutant backend: CPU TOP_K writes an out-of-range index into dst[0] when TOPK_GARBAGE is set
    python3 - ggml/src/ggml-cpu/ops.cpp <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = '        std::copy(tmp, tmp + top_k, dst_data);\n'
i = s.index('static void ggml_compute_forward_top_k_f32')
j = s.index(a, i)
s = s[:j] + a + '        if (getenv("TOPK_GARBAGE")) { dst_data[0] = 1 << 30; }\n' + s[j + len(a):]
open(p, 'w').write(s)
PY
    do_build mutant-build test-backend-ops
    export TOPK_GARBAGE=1
    run mutant-fixed-test nz '0/1 backends passed' -- "$B/bin/test-backend-ops" test -o TOP_K -b CPU
    # same mutant, master's err(): the test reads src through the garbage index (crash or garbage compare)
    git apply -R "$G/prod.patch"
    do_build mutant-unfixed-build test-backend-ops
    run mutant-unfixed-test nz -- "$B/bin/test-backend-ops" test -o TOP_K -b CPU
    absent mutant-unfixed-no-summary mutant-unfixed-test '0/1 backends passed'
    unset TOPK_GARBAGE
    restore
    ;;
*) echo "unknown PKG $PKG"; exit 8 ;;
esac

echo "pkg$PKG: fails=$fails"
[[ $fails == 0 ]]
