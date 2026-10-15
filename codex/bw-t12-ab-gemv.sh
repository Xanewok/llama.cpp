#!/usr/bin/env bash
# Patch window on the fs2 base (harness/bw-run.sh payload): build this tree, gate OPS on CUDA0 and CUDA1, record perf of PERF_P,
# then freeze frozen/$NAME. Contract: nothing is frozen unless every gate passed.
# usage: OUT | --preflight. rc: 0 ok, 6 integrity, 8 gate failed.
set -euo pipefail
NAME=t12-ab-gemv
OPS=MUL_MAT
PERF_P='type_a=f32,type_b=f32,m=(24|31|32|33|48),n=(1|6|12),k=5120'
DESC='fs2 + T-12 mmvf for thin F32 matrices (2 <= rows < MMF tile, <= 16 columns)'
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$HOME/repos/qwen38-eval-worktrees
REF=$W/frozen/stack11-fa-s2/build/bin/test-backend-ops
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
VISIBLE=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
export GIT_NO_LAZY_FETCH=1 BUILDGATE_MIN_GIB=10 TMPDIR=/mnt/scratch/tmp-build
export PATH="/usr/local/cuda-13.2/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2

if [[ ${1:-} == --preflight ]]; then
    for p in "$REF" "$CCCL" "$W/bin/buildgate" "$W/bin/wt"; do [[ -e $p ]] || { echo "preflight: missing $p"; exit 6; }; done
    [[ ! -e $W/frozen/$NAME ]] || { echo "preflight: frozen/$NAME exists"; exit 6; }
    exit 0
fi
OUT=${1:?out dir}
mkdir -p "$OUT" "$TMPDIR"
cd "$ROOT"
[[ -f build-cuda/CMakeCache.txt ]] || cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON > "$OUT/configure.log" 2>&1
"$W/bin/buildgate" cmake --build build-cuda -j4 --target llama-server test-backend-ops > "$OUT/build.log" 2>&1 \
    || { echo "bw-run: FAIL build (tail $OUT/build.log)"; tail -20 "$OUT/build.log"; exit 8; }
cp build-cuda/bin/llama-server build-cuda/bin/round-trace

# LOG: no FAIL line and every "N/M tests passed" complete; prints the case count.
cases() {
    grep -aEo '[0-9]+/[0-9]+ tests passed' "$1" | awk -F'[ /]' '{ if ($1 != $2) bad = 1; s += $1 } END { print bad ? -1 : s + 0 }'
}
for dev in CUDA0 CUDA1; do
    new=$OUT/ops-$dev.log ref=$OUT/ops-$dev-ref.log
    CUDA_VISIBLE_DEVICES=$VISIBLE build-cuda/bin/test-backend-ops test -o "$OPS" -b "$dev" > "$new" 2>&1 || true
    CUDA_VISIBLE_DEVICES=$VISIBLE "$REF" test -o "$OPS" -b "$dev" > "$ref" 2>&1 || true
    n=$(cases "$new") m=$(cases "$ref")
    if grep -aq 'FAIL' "$new" || (( n < 1 || n < m )); then echo "bw-run: FAIL $OPS on $dev: new=$n ref=$m ($new)"; exit 8; fi
    echo "pass: $OPS $dev cases new=$n ref=$m"
    CUDA_VISIBLE_DEVICES=$VISIBLE build-cuda/bin/test-backend-ops perf -o "$OPS" -b "$dev" -p "$PERF_P" > "$OUT/perf-$dev.log" 2>&1 || true
done
"$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda --desc "$DESC" > "$OUT/freeze.log" 2>&1 \
    || { echo "bw-run: FAIL freeze (tail $OUT/freeze.log)"; tail -5 "$OUT/freeze.log"; exit 8; }
echo "frozen: $W/frozen/$NAME"
