#!/usr/bin/env bash
# Fable B/C/D existence window: one CUDA build of this tree, then tests/test-fable-bcd on CUDA1 and CUDA0 (levers B, C, D),
# plus a PDL-off stock diagnostic for C. Perf-only: the build carries the K9-fix MMQ PDL substrate and prefetch-above-the-wait
# variants (probe code, local), none of it is served. Verdict rules are fixed in codex/fable_bcd_summary.py and REPORT-fable-bcd.md.
# rc: 0 pass | 5 control void | 6 integrity | 8 deterministic failure (build, run, missing DONE)
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export BUILDGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BASE=45d90d3e4ec1b713bef3804bac5bfc93cbff94c1
SCR=/mnt/scratch/fable-bcd
BIN=$SCR/build/bin/test-fable-bcd
OBJ=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
CUOBJDUMP=/usr/local/cuda-13.2/bin/cuobjdump

preflight() {
    [[ $(git branch --show-current) == topic/fable-bcd ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    [[ $(grep -c -- "fable_probe_api" ggml/src/ggml-cuda/fable-probe.cu) -ge 1 ]] || return 1
    [[ $(grep -c -- "test-fable-bcd" tests/CMakeLists.txt) -ge 1 ]] || return 1
    local p
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc "$CUOBJDUMP" codex/fable_bcd_summary.py \
             tests/test-fable-bcd.cpp ggml/src/ggml-cuda/fable-probe.cu /mnt/scratch; do
        [[ -e $p ]] || return 1
    done
    [[ -w /mnt/scratch ]] || return 1
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-fable-bcd: FAIL preflight'; exit 8; }
    echo 'bw-fable-bcd: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-fable-bcd && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/fable-bcd-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-fable-bcd: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] && return 0
    [[ $rc == 5 || $rc == 6 ]] && exit "$rc"
    echo "bw-fable-bcd: FAIL $name"
    exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
builds() {
    mkdir -p "$SCR"
    nice -n 19 "$W/bin/buildgate" cmake -S "$ROOT" -B "$SCR/build" "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build "$SCR/build" --target test-fable-bcd -j4
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
stamps() {
    local n
    n=$(strings "$SCR/build/bin/libggml-cuda.so" | grep -c fable_probe_api || true)
    echo "fable_probe_api_count_M=$n"
    [[ $n -ge 1 ]]
    [[ -x $BIN ]]
    [[ $(grep -c '120a' "$SCR/build/$OBJ/flags.make" || true) -ge 1 ]]
    strings "$SCR/build/bin/libggml-base.so" | grep -c "$(git rev-parse --short HEAD)" > /dev/null
}
# informational: the prefetch instructions must exist in the objects that carry the variants (count of CCTL in SASS; not a gate)
sass() {
    local o n
    for o in mmvq.cu.o template-instances/mmq-instance-iq4_xs.cu.o fable-probe.cu.o; do
        n=$(nice -n 19 "$CUOBJDUMP" -sass "$SCR/build/$OBJ/$o" | grep -c 'CCTL' || true)
        echo "CCTL_lines_M $o $n"
    done
}
clocks() {
    nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv,noheader 2>/dev/null \
        | sed "s/^/$1,/" >> "$OUT/clocks.csv" || true
}
run_one() { # TAG DEV ENV_ASSIGN PROC EXTRA_ARGS...
    local tag=$1 dev=$2 envs=$3 proc=$4 rc=0
    shift 4
    clocks "$tag-pre"
    env CUDA_VISIBLE_DEVICES="$VISIBLE" $envs timeout 2400 nice -n 19 "$BIN" --dev "$dev" --proc "$proc" "$@" \
        > "$OUT/$tag.tsv" 2> "$OUT/$tag.err" || rc=$?
    clocks "$tag-post"
    printf '%s\t%s\n' "$tag" "$rc" >> "$OUT/runs.tsv"
    [[ $rc == 0 ]] || { echo "FAIL $tag rc=$rc"; return 8; }
    [[ $(grep -ac '^DONE' "$OUT/$tag.tsv" || true) == 1 ]] || { echo "FAIL $tag: no DONE line"; return 8; }
}
runs() {
    [[ $VISIBLE == GPU-* ]] || return 6
    run_one run-CUDA1 CUDA1 FABLE_X=1 main --lever all --rounds 7
    run_one run-nopdl-CUDA1 CUDA1 GGML_CUDA_PDL=0 nopdl --lever C --rounds 7 --stock-only
    run_one run-CUDA0 CUDA0 FABLE_X=1 main --lever all --rounds 7
}
summary() {
    local rc=0
    python3 codex/fable_bcd_summary.py "$OUT" > "$OUT/summary.out" 2> "$OUT/verdict-stderr.txt" || rc=$?
    cat "$OUT/verdict-stderr.txt"
    return $rc
}
stage 40 cuda-build builds
stage 42 stamps stamps
stage 43 sass-info sass
stage 46 runs runs
stage 48 summary summary
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, verdict=(p/'verdict.txt').read_text().splitlines(),
    summary='summary.tsv', served='served.tsv'), indent=2) + '\n')
PY_RESULT
echo 'bw-fable-bcd: PASS'
