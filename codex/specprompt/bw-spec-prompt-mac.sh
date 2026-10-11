#!/usr/bin/env bash
# Mac CPU gate of spec-prompt (STEP fix 5): CPU build, test-speculative must pass; mutant (an n-gram drafter that reads the
# prompt but does not declare it) must fail; the server compiles.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-spec-prompt-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-speculative server-context test-arg-parser -j4
}
units() {
    "$BW_GATE" build-cpu/bin/test-speculative
    for t in ngram-simple ngram-map-k ngram-map-k4v ngram-mod ngram-cache; do
        grep -ac "^$t: uses_prompt" "$OUT/units.log" > /dev/null
    done
    "$BW_GATE" build-cpu/bin/test-arg-parser
}
mutant() {
    local f=common/speculative.cpp
    cp "$f" "$OUT/speculative.cpp.orig"
    trap 'cp "$OUT/speculative.cpp.orig" "$f"' RETURN
    python3 - "$f" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.index('struct common_speculative_impl_ngram_simple')
old = 'bool uses_prompt() const override { return true; }'
j = s.index(old, i)
open(p, 'w').write(s[:j] + old.replace('true', 'false') + s[j + len(old):])
PY
    "$BW_BUILDGATE" cmake --build build-cpu --target test-speculative -j4
    local rc=0
    "$BW_GATE" build-cpu/bin/test-speculative || rc=$?
    [[ $rc != 0 ]] || { echo 'FAIL mutant escaped'; return 1; }
}
stage 10 build build
stage 11 units units
stage 12 mutant mutant
echo 'bw-spec-prompt-mac: PASS'
