#!/usr/bin/env bash
# Mac CPU gates for K10: the CUDA top-k kernels run on CPU threads (tests/test-topk-batched.cpp), their negative
# controls, and the compile + CPU-side behaviour of the new test-backend-ops cases. Run through bin/bw-mac.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
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
    echo "bw-k10-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
emu_build() {
    nice -n 19 clang++ -std=c++17 -O1 -pthread tests/test-topk-batched.cpp -o "$OUT/test-topk-batched-emu"
}
emu_run() {
    "$BW_GATE" "$OUT/test-topk-batched-emu"
}
emu_mutants() {
    local m
    for m in 1 2 3 4 5 6; do
        if env TOPK_MUTANT=$m TOPK_FAILFAST=1 "$BW_GATE" "$OUT/test-topk-batched-emu" > "$OUT/mutant-$m.log" 2>&1; then
            echo "FAIL mutant $m escaped"; return 1
        fi
        grep -aq '^FAIL ' "$OUT/mutant-$m.log"
    done
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
cmake_build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops test-topk-batched -j4
}
cmake_run() {
    "$BW_GATE" build-cpu/bin/test-topk-batched
    # canonical cases are CUDA-only: not supported on CPU, and the other TOP_K cases still pass
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o TOP_K -j 1 > "$OUT/top-k-cpu.log" 2>&1
    grep -aq 'not supported' "$OUT/top-k-cpu.log"
    grep -aq 'canonical=1' "$OUT/top-k-cpu.log"
}
canonical_control() {
    # the CPU order (partial sort, first two swapped) is not canonical: the exact-order check must reject it
    if env TOPK_CANONICAL_ANY=1 "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o TOP_K -p 'canonical=1' -j 1 > "$OUT/canonical-control.log" 2>&1; then
        echo 'FAIL canonical check accepted the CPU order'; return 1
    fi
    grep -aq 'canonical=1' "$OUT/canonical-control.log"
}
static_checks() {
    bash -n codex/k10/bw-k10-mac.sh codex/k10/bw-k10.sh
    python3 -m py_compile codex/k10/k10_perf.py
    PYTHONPATH="$ROOT/codex/shlint" "$BW_GATE" python3 -m gpuq.shlint codex/k10/bw-k10-mac.sh codex/k10/bw-k10.sh
}
stage 10 emu-build emu_build
stage 11 emu-kernels emu_run
stage 12 emu-mutants emu_mutants
stage 20 cpu-build cmake_build
stage 21 cpu-tests cmake_run
stage 22 canonical-control canonical_control
stage 31 static static_checks
python3 - "$OUT" <<'PY_SUMMARY'
import hashlib, json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
files = subprocess.check_output(['git','ls-files','src','ggml','tests','common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
stages = [dict(code=int(c),name=n,rc=int(rc),wall_s_M=int(t)) for c,n,rc,t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=h.hexdigest(),stages=stages),indent=2)+'\n')
PY_SUMMARY
