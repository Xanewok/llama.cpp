#!/usr/bin/env bash
# Mac CPU gate for the zero-copy GET_ROWS probe, run through bin/bw-mac; argument = OUT dir.
# Builds test-zc-getrows CPU-only and runs its harness on the CPU device (arms a, c): graph build, views, ids refresh,
# timing loops and the equality judge. No zero-copy meaning; the GPU window measures that.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=4
cd "$(dirname "$0")/../.."
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL dirty tracked tree'; exit 9; }
git rev-parse HEAD > "$OUT/source-commit.txt"
: > "$OUT/stages.tsv"
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
t0=$SECONDS rc=0
{ "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}" &&
  "$BW_BUILDGATE" cmake --build build-cpu --target test-zc-getrows -j"${BW_JOBS:-4}"; } > "$OUT/build.log" 2>&1 || rc=$?
printf 'build\trc=%s\twall_s=%s\n' "$rc" "$((SECONDS-t0))" | tee -a "$OUT/stages.tsv"
[[ $rc == 0 ]] || exit 8
t0=$SECONDS rc=0
"$BW_GATE" build-cpu/bin/test-zc-getrows --devs CPU --arms ac --rows 1,12,512 --iters 30 --reps 3 --threads 4 \
    > "$OUT/smoke.txt" 2>&1 || rc=$?
ok=1
[[ $rc == 0 ]] || ok=0
for p in '^zc-getrows: PASS$' '^zc-eq: dev=CPU rows=512 a_vs_c differing=0/2621440 \(info\)$' \
         '^zc: dev=CPU arm=a rows=12 mode=pipelined cond=idle n=3 ' '^zc: dev=CPU arm=c rows=1 mode=sync cond=idle n=30 '; do
    grep -aEq "$p" "$OUT/smoke.txt" || { echo "missing: $p"; ok=0; }
done
printf 'smoke\trc=%s\tgate=%s\twall_s=%s\n' "$rc" "$([[ $ok == 1 ]] && echo PASS || echo FAIL)" "$((SECONDS-t0))" | tee -a "$OUT/stages.tsv"
[[ $ok == 1 ]] || exit 8
