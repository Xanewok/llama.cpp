#!/usr/bin/env bash
# bw-mac payload body: CPU build of bw-np2-driver, the hostless cells of one PART and the judge. Usage: mac.sh OUT PART.
# rc: 0 pass, 75 RAM gate skip, 10 build, 20 judge (its rc in judge.log).
set -uo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
OUT=${1:?out} PART=${2:?part}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODEL=${BW_MODEL:-$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
# shellcheck disable=SC2086
cmake -S codex/stack7-test -B build-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_CCACHE=${BW_CCACHE:-ON} -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF \
    ${BW_CPU_EXTRA:--DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod} \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$OUT/cmake.log" 2>&1 || exit 10
nice -n 19 cmake --build build-cpu --target bw-np2-driver -j3 > "$OUT/build.log" 2>&1 || { grep -a -B2 -A8 'error' "$OUT/build.log" | tail -80; exit 10; }
echo "build ok"
bash codex/hostless/cells.sh "$ROOT/build-cpu/bin/bw-np2-driver" "$MODEL" "$OUT/cells" "$PART" | tee "$OUT/cells.log"
[[ ${PIPESTATUS[0]} == 75 ]] && exit 75
python3 codex/hostless/judge.py "$OUT/cells" | tee "$OUT/judge.log"
[[ ${PIPESTATUS[0]} == 0 ]] || exit 20
exit 0
