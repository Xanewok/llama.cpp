#!/usr/bin/env bash
# Mac CPU gates for K6 (residual fusion, shared activation reuse). Run through bw-mac; CUDA code is not compiled here.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
GATE=$ROOT/codex/mac-model-gate.sh
CXXF=(-std=c++17 -O1 -Iggml/include -Iggml/src -Iggml/src/ggml-cuda)
LIBS=(-Lbuild-cpu/bin "-Wl,-rpath,$ROOT/build-cpu/bin" -lggml-base)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    [[ -z ${K6_ONLY:-} || " $K6_ONLY " == *" $code "* ]] || return 0
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "k6-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops -j4
}
# Compile a host contract against a header directory; $1 is the output name, $2 the source, $3 an include override.
host_build() {
    local include=()
    [[ -z ${3:-} ]] || include=("-I$3")
    "$BW_BUILDGATE" c++ ${include[@]+"${include[@]}"} "${CXXF[@]}" -Icodex/k6 "$2" "${LIBS[@]}" -o "build-cpu/bin/$1"
}
# The mutated header copy must differ from the real one, so a stale sed pattern cannot pass silently.
mutate() {
    local header=$1 dir=$2 pattern=$3 replacement=$4
    mkdir -p "$dir"
    sed "s|$pattern|$replacement|" "$header" > "$dir/$(basename "$header")"
    if cmp -s "$header" "$dir/$(basename "$header")"; then
        echo "FAIL mutation pattern did not change $header: $pattern"; return 1
    fi
}
must_fail() {
    local rc=0
    "$@" || rc=$?
    [[ $rc == 1 ]] || { echo "FAIL negative control did not fail with rc 1 (rc=$rc): $*"; return 1; }
}
contracts() {
    host_build q8-cache-contract codex/k6/q8-cache-contract.cpp
    build-cpu/bin/q8-cache-contract
    host_build add-rms-norm-matcher codex/k6/add-rms-norm-matcher.cpp
    build-cpu/bin/add-rms-norm-matcher
    local m=build-cpu/mutants
    mutate ggml/src/ggml-cuda/q8-cache.h "$m/stale" 'if (start <= src ? src - start < size : start - src < src_size) {' 'if (false) {'
    host_build q8-cache-stale codex/k6/q8-cache-contract.cpp "$m/stale"
    must_fail build-cpu/bin/q8-cache-stale
    mutate ggml/src/ggml-cuda/q8-cache.h "$m/bound" 'GGML_CUDA_Q8_REUSE_MAX_ROWS = 16' 'GGML_CUDA_Q8_REUSE_MAX_ROWS = 17'
    host_build q8-cache-bound codex/k6/q8-cache-contract.cpp "$m/bound"
    must_fail build-cpu/bin/q8-cache-bound
    mutate ggml/src/ggml-cuda/add-rms-norm.h "$m/cap" 'GGML_CUDA_ADD_RMS_NORM_MAX_ROWS = 16' 'GGML_CUDA_ADD_RMS_NORM_MAX_ROWS = 17'
    host_build matcher-cap codex/k6/add-rms-norm-matcher.cpp "$m/cap"
    must_fail build-cpu/bin/matcher-cap
    mutate ggml/src/ggml-cuda/add-rms-norm.h "$m/norm" 'valid = valid && !overlaps(norm, t);' '(void) t;'
    host_build matcher-norm codex/k6/add-rms-norm-matcher.cpp "$m/norm"
    must_fail build-cpu/bin/matcher-norm
    mutate ggml/src/ggml-cuda/add-rms-norm.h "$m/resid" ' && !overlaps(mul, add);' ';'
    host_build matcher-resid codex/k6/add-rms-norm-matcher.cpp "$m/resid"
    must_fail build-cpu/bin/matcher-resid
}
backend_ops() {
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_SHARED -j 1 > "$OUT/shared.log" 2>&1
    grep -aq 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS' "$OUT/shared.log"
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o RMS_NORM_MUL_ADD,ADD_RMS_NORM -j 1 > "$OUT/add-rms.log" 2>&1
    grep -aq 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS' "$OUT/add-rms.log"
    tail -n 3 "$OUT/shared.log" "$OUT/add-rms.log"
}
static_checks() {
    bash -n codex/k6/k6-mac.sh
    PYTHONPATH="$ROOT/codex/shlint" python3 -m gpuq.shlint codex/k6/k6-mac.sh
}
stage 20 cpu-build build
stage 21 host-contracts-and-mutants contracts
stage 22 cpu-backend-ops backend_ops
stage 31 payload-lint static_checks
python3 - "$OUT" <<'PY_SUMMARY'
import hashlib, json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), source_sha256=h.hexdigest(), stages=stages), indent=2) + '\n')
PY_SUMMARY
