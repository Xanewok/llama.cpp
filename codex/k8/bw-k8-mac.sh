#!/usr/bin/env bash
# Mac payload (bin/bw-mac): CPU-only gates for K8. The MMQ kernels are CUDA-only and compile in the build window.
#   1 stream-k partition model (codex/k8/mmq_sk_sim.cpp)  2 CPU build of test-backend-ops  3 new MUL_MAT/MUL_MAT_ID cases CPU vs CPU
#   4 mmq-hash plumbing on CPU (rows <= 3072: first loop and the expert path run and print stable lines)
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/../.."
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 10
OUT=${1:?output}
mkdir -p "$OUT"
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
stage() {
    local name=$1 rc=0 t0=$SECONDS
    shift
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-k8-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 1
}
sim() {
    nice -n 19 c++ -std=c++17 -O2 -ffp-contract=off -o "$OUT/mmq_sk_sim" codex/k8/mmq_sk_sim.cpp
    nice -n 19 "$OUT/mmq_sk_sim" | tee "$OUT/sim.txt"
    grep -aq '^PASS$' "$OUT/sim.txt"
}
build() {
    "$BW_BUILDGATE" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops -j"${BW_JOBS:-4}"
}
cases() {
    nice -n 19 build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT -p 'type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|5120|6144),n=(9|12|24|25),k=1024,' > "$OUT/mm.log" 2>&1
    grep -aEq '^ +([1-9][0-9]*)/\1 tests passed' "$OUT/mm.log"
    nice -n 19 build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID -p 'type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,n_mats=4,n_used=2,b=0,m=(3072|5120),n=(24|48),k=1024,' > "$OUT/id.log" 2>&1
    grep -aEq '^ +([1-9][0-9]*)/\1 tests passed' "$OUT/id.log"
}
hash() {
    nice -n 19 build-cpu/bin/test-backend-ops mmq-hash CPU 3072 > "$OUT/hash-a.out"
    nice -n 19 build-cpu/bin/test-backend-ops mmq-hash CPU 3072 > "$OUT/hash-b.out"
    cmp "$OUT/hash-a.out" "$OUT/hash-b.out"
    [[ $(grep -ac '^id_' "$OUT/hash-a.out") == 8 ]]
    [[ $(wc -l < "$OUT/hash-a.out") -gt 100 ]]
}
stage 1-sim sim
stage 2-build build
stage 3-cases cases
stage 4-hash hash
echo "bw-k8-mac: PASS"
