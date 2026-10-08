#!/usr/bin/env bash
# K9 Mac gates: CPU build of test-backend-ops plus the mmq-chain harness on the CPU backend. PDL, MMQ and the CUDA edits are not
# compiled or exercised here; this validates the harness (stable hashes, alternating inputs differ, corruption is caught, perf mode runs).
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
GATE=$ROOT/codex/mac-model-gate.sh
T=build-cpu/bin/test-backend-ops
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-k9-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops -j4
}
chain() {
    "$GATE" $T mmq-chain CPU hash 8 quick > "$OUT/hash-1.out" 2> "$OUT/hash-1.err"
    "$GATE" $T mmq-chain CPU hash 8 quick > "$OUT/hash-2.out" 2> "$OUT/hash-2.err"
    cmp "$OUT/hash-1.out" "$OUT/hash-2.out"
    [[ $(grep -c '^chain ' "$OUT/hash-1.out") == 2 ]]
    awk '$1 == "chain" && $5 == $6 {bad = 1} END {exit bad}' "$OUT/hash-1.out"
    grep -c 'recompute mismatches 0 of 8' "$OUT/hash-1.err" > "$OUT/n-clean.txt"
    [[ $(cat "$OUT/n-clean.txt") == 2 ]]
    local rc=0
    "$GATE" $T mmq-chain CPU hash 8 quick corrupt > "$OUT/corrupt.out" 2> "$OUT/corrupt.err" || rc=$?
    [[ $rc == 1 ]]
    grep -aq 'mmq-chain: FAIL 2 recomputes differ' "$OUT/corrupt.err"
    "$GATE" $T mmq-chain CPU perf 3 quick > "$OUT/perf.out" 2> "$OUT/perf.err"
    [[ $(grep -c '^perf ' "$OUT/perf.out") == 2 ]]
}
static_checks() {
    bash -n codex/bw-k9-mac.sh codex/bw-k9.sh
    python3 -m py_compile codex/k9_summary.py
    PYTHONPATH="$ROOT/codex/shlint" "$BW_GATE" python3 -m gpuq.shlint codex/bw-k9-mac.sh codex/bw-k9.sh
}
stage 20 cpu-build build
stage 21 chain-harness chain
stage 31 static-checks static_checks
python3 - "$OUT" <<'PY_SUMMARY'
import hashlib, json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=h.hexdigest(), stages=stages), indent=2) + '\n')
PY_SUMMARY
