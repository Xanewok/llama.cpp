#!/usr/bin/env bash
# Mac CPU gate for the zero-copy token table PR, run through bin/bw-mac; argument = OUT dir.
# CPU-only parts: the tree builds (llama, tests), test-host-mapped skips with rc 0, test-alloc passes, and greedy tokens on
# plain CPU and Meta(CPU,CPU) equal the same tree with the PR reverted (codex/zc-pr/zc.patch -R). A CPU-only build has no
# host-mapped buffer type, so this checks that the meta/llama changes are inert there; the CUDA paths run in the GPU window.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
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

stage() { # NAME OK(0|1) RC T0
    printf '%s\trc=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$([[ $2 == 1 ]] && echo PASS || echo FAIL)" "$((SECONDS-$4))" | tee -a "$OUT/stages.tsv"
    [[ $2 == 1 ]] || fails=$((fails+1))
}

# NAME TEST [GREP...]: rc 0 and every pattern in stdout
run_test() {
    local name=$1 bin=$2 rc=0 t0=$SECONDS ok=1 g
    shift 2
    "$BW_GATE" "build-cpu/bin/$bin" > "$OUT/$name.txt" 2> "$OUT/$name.log" || rc=$?
    [[ $rc == 0 ]] || ok=0
    for g in "$@"; do grep -aq -- "$g" "$OUT/$name.txt" || { echo "missing in $name: $g"; ok=0; }; done
    stage "$name" "$ok" "$rc" "$t0"
}

# NAME META(0|1) NO_HOST(0|1): greedy tokens of two sequences; the build is the current tree
run_model() {
    local name=$1 rc=0 t0=$SECONDS ok=1
    "$BW_BUILDGATE" cmake --build build-cpu --target llama -j"${BW_JOBS:-4}" > "$OUT/model-$name.build.log" 2>&1 || ok=0
    "$BW_GATE" clang++ -std=c++17 -O1 -Iinclude -Iggml/include codex/zc-pr/driver.cpp -o build-cpu/bin/zc-driver \
        -Lbuild-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/build-cpu/bin" >> "$OUT/model-$name.build.log" 2>&1 || ok=0
    "$BW_GATE" build-cpu/bin/zc-driver "$BW_MODEL" "$2" "$3" 24 > "$OUT/model-$name.txt" 2> "$OUT/model-$name.log" || rc=$?
    [[ $rc == 0 ]] && grep -aq '^seq1:' "$OUT/model-$name.txt" || ok=0
    stage "model-$name" "$ok" "$rc" "$t0"
}

t0=$SECONDS rc=0
{ "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}" &&
  "$BW_BUILDGATE" cmake --build build-cpu --target llama test-host-mapped test-alloc -j"${BW_JOBS:-4}"; } > "$OUT/build.log" 2>&1 || rc=$?
stage build "$([[ $rc == 0 ]] && echo 1 || echo 0)" "$rc" "$t0"
[[ $rc == 0 ]] || exit 8

run_test test-host-mapped test-host-mapped '^test-host-mapped: SKIP'
run_test test-alloc test-alloc
run_model zc-cpu    0 0
run_model zc-meta   1 0
run_model zc-nohost 0 1

git apply -R codex/zc-pr/zc.patch
run_model base-cpu  0 0
run_model base-meta 1 0
git apply codex/zc-pr/zc.patch
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }

t0=$SECONDS ok=1
for f in zc-meta zc-nohost base-cpu base-meta; do
    cmp <(grep -a '^seq' "$OUT/model-zc-cpu.txt") <(grep -a '^seq' "$OUT/model-$f.txt") || { echo "tokens differ: $f"; ok=0; }
done
stage model-token-identity "$ok" - "$t0"
# where token_embd lives (informational: a CPU-only build has no mapped host buffer type)
grep -a 'model buffer size' "$OUT"/model-*.log > "$OUT/model-buffers.txt" || true

echo "zc-pr mac: fails=$fails"
[[ $fails == 0 ]]
