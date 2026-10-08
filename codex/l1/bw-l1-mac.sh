#!/usr/bin/env bash
# Mac payload (bin/bw-mac): CPU-only compile of the test-backend-ops edits (the MMQ kernels are CUDA-only and compile in the build window).
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/../.."
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 10
OUT=${1:?output}
mkdir -p "$OUT"
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
"$BW_BUILDGATE" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF "${extra[@]}" > "$OUT/configure.log" 2>&1
"$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops -j"${BW_JOBS:-4}" > "$OUT/build.log" 2>&1
# The new MUL_MAT cases exist and run (CPU vs CPU: shape/case plumbing only).
build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT -p 'type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|8704|9216),n=(9|12|16|17|24|25),k=1024,' > "$OUT/cases.log" 2>&1
grep -Eq '^ +([1-9][0-9]*)/\1 tests passed' "$OUT/cases.log"
grep -Eo '^ +[0-9]+/[0-9]+ tests passed' "$OUT/cases.log" | tee "$OUT/summary.txt"
