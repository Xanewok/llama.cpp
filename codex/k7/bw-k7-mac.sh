#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
GATE=$ROOT/codex/k7/model-gate.sh
MODEL=${BW_MODEL:?}
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-k7: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops llama-completion -j4
}
units() {
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o RMS_NORM_SCALE_PAIR -j 1 > "$OUT/pair.log" 2>&1
    grep -c 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$OUT/pair.log" > /dev/null
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o RMS_NORM -j 1 > "$OUT/rms-norm.log" 2>&1
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o SCALE -j 1 > "$OUT/scale.log" 2>&1
}
# Each guard must be what rejects its unsafe layout: with it removed the test has to fail with its own message.
mutants() {
    local mask msg
    for mask in 1 2 4 8; do
        case $mask in
            1) msg='first output over the second input' ;;
            2) msg='second output over the first input' ;;
            4) msg='outputs overlap' ;;
            8) msg='second input is a view of the first output' ;;
        esac
        if "$GATE" env GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT=$mask build-cpu/bin/test-backend-ops test -b CPU -o RMS_NORM_SCALE_PAIR -j 1 > "$OUT/alias-mutant-$mask.log" 2>&1; then
            echo "FAIL alias-guard mutant $mask escaped"; return 1
        fi
        grep -c "FAIL RMS_NORM_SCALE_PAIR matcher accepted: $msg" "$OUT/alias-mutant-$mask.log" > /dev/null
    done
}
# Real qwen35 graph: the q and k chains must be adjacent for the matcher (sched debug lists non-view nodes).
graph() {
    "$GATE" env GGML_SCHED_DEBUG=2 build-cpu/bin/llama-completion -m "$MODEL" -c 256 -t 4 -ngl 0 -n 2 -p 'Hello' -no-cnv --no-warmup -lv 5 > "$OUT/graph-raw.log" 2>&1
    python3 codex/k7/analyze_graph.py "$OUT/graph-raw.log"
    rm -f "$OUT/graph-raw.log"
}
stage 20 cpu-build build
stage 21 pair-and-neighbour-op-tests units
stage 22 alias-guard-mutant mutants
stage 23 real-graph-adjacency graph
python3 - "$OUT" <<'PY_RESULT'
import json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [l.split('\t') for l in (p/'stages.tsv').read_text().splitlines()]
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=head, stages=[dict(code=int(r[0]), name=r[1], rc=int(r[2]), wall_s_M=int(r[3])) for r in rows]), indent=2)+'\n')
PY_RESULT
echo 'bw-k7: PASS'
