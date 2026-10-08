#!/usr/bin/env bash
# bw-mac payload: CPU build of the fixtures, the np2 M2 identity matrix and the np1 chain regression. Usage: bw-m2-mac.sh OUT
# rc: 0 pass, 75 skipped (RAM gate), 10 build, else the first failing judge's rc (np2 judge, then 20 + np1 failure).
set -uo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
OUT=${1:?out}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODEL=${BW_MODEL:-$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
GATE=$ROOT/codex/m2/gate.sh
# shellcheck disable=SC2086
cmake -S codex/stack7-test -B build-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_CCACHE=${BW_CCACHE:-ON} -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF \
    ${BW_CPU_EXTRA:--DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod} \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$OUT/cmake.log" 2>&1 || exit 10
nice -n 19 cmake --build build-cpu --target bw-np2-driver chain-cpu -j3 > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 10; }
echo "build ok"

bash codex/m2/np2-matrix.sh "$ROOT/build-cpu/bin/bw-np2-driver" "$MODEL" "$OUT/np2" | tee "$OUT/np2-matrix.log"
rc=${PIPESTATUS[0]}
[[ $rc == 75 ]] && exit 75
python3 codex/m2/judge-np2.py "$OUT/np2" | tee "$OUT/np2-judge.log"
np2=${PIPESTATUS[0]}

export S7_GATE=$GATE S7_MODEL=$MODEL S7_VOCAB=$HOME/qwen38-stack7/models/swift_draft_vocab_ids.json CHAIN_THREADS=4 GPUQ_BUILD_WINDOW=1
python3 codex/chain_cpu_identity.py --out "$OUT/np1" --n-predict 48 > "$OUT/np1.log" 2>&1
np1=$?
if grep -rqs 'skipped: RAM' "$OUT/np1"; then
    grep -rhs 'skipped: RAM' "$OUT/np1" | sed -n 1p
    exit 75
fi
[[ $np1 == 0 ]] && { python3 codex/check-chain.py "$OUT/np1" | tee "$OUT/np1-judge.log"; np1=${PIPESTATUS[0]}; }
echo "np2 rc=$np2 np1 rc=$np1"
[[ $np2 != 0 ]] && exit "$np2"
[[ $np1 != 0 ]] && exit $((20 + np1))
exit 0
