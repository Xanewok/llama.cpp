#!/usr/bin/env bash
# bw-mac payload for chain-m2-s10: CPU build, the np2 M2 identity matrix (fold C on, the stack10 default) and its judge, the np1
# chain identity, and test-llama-archs -a qwen35 -b CPU (pkg 87 test). Usage: bw-m2s10-mac.sh OUT.
# rc: 0 pass, 75 RAM gate skip, 10 build, 30 np2 judge, 31 np2 engagement strings, 40 np1, 50 test-llama-archs.
set -uo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
OUT=${1:?out}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODEL=${BW_MODEL:-$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
MGATE=$ROOT/codex/stack10/mac/model-gate.sh
unset LLAMA_MTP_CHAIN LLAMA_MTP_FOLD_C LLAMA_MTP_FOLD_DEBUG_MUTANT LLAMA_ROUND_TRACE LLAMA_MTP_CHAIN_DUMP LLAMA_MTP_DRAFT_DUMP
# shellcheck disable=SC2086
cmake -S codex/stack7-test -B build-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_CCACHE=${BW_CCACHE:-ON} -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF \
    ${BW_CPU_EXTRA:--DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod} \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF > "$OUT/cmake.log" 2>&1 || exit 10
nice -n 19 cmake --build build-cpu --target bw-np2-driver chain-cpu test-llama-archs -j3 > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 10; }
echo "build ok"

M2_GATE=$MGATE bash codex/m2/np2-matrix.sh "$ROOT/build-cpu/bin/bw-np2-driver" "$MODEL" "$OUT/np2" | tee "$OUT/np2-matrix.log"
[[ ${PIPESTATUS[0]} == 75 ]] && exit 75
python3 codex/m2/judge-np2.py "$OUT/np2" | tee "$OUT/np2-judge.log"
np2=${PIPESTATUS[0]}
# the chain engages for both slots with fold C on; the chain-off arm keeps the step loop
eng=0
grep -aq 'MTP chain: k=5 seqs=2' "$OUT/np2/0-k5-s1729-on.log" || eng=1
grep -aq 'mtp_fold_decode: engaged n_seq=2' "$OUT/np2/0-k5-s1729-on.log" || eng=1
grep -aq 'mtp_fold_decode: engaged n_seq=2' "$OUT/np2/0-k5-s1729-off.log" || eng=1
[[ $(grep -ac 'MTP chain: k=' "$OUT/np2/0-k5-s1729-off.log") == 0 ]] || eng=1
echo "np2 engagement strings rc=$eng"
# post-DC host gap by DC rows (old M2: 1.1 ms at 10 rows vs 0.09 at 5, output_reorder): report only
python3 codex/m2/dcgap.py "$OUT"/np2/0-k5-s*-on.trace.jsonl "$OUT"/np2/0-k3-s1729-uni-on.trace.jsonl | tee "$OUT/dcgap.txt"

export S7_GATE=$MGATE S7_MODEL=$MODEL S7_VOCAB=$HOME/qwen38-stack7/models/swift_draft_vocab_ids.json CHAIN_THREADS=4 GPUQ_BUILD_WINDOW=1
python3 codex/chain_cpu_identity.py --out "$OUT/np1" --n-predict 48 > "$OUT/np1.log" 2>&1
np1=$?
[[ $np1 == 0 ]] && { python3 codex/check-chain.py "$OUT/np1" | tee "$OUT/np1-judge.log"; np1=${PIPESTATUS[0]}; }
echo "np1 rc=$np1"

bash "$MGATE" build-cpu/bin/test-llama-archs -a qwen35 -b CPU > "$OUT/test-llama-archs.log" 2>&1
archs=$?
echo "test-llama-archs rc=$archs"; tail -3 "$OUT/test-llama-archs.log"

echo "np2 rc=$np2 eng rc=$eng np1 rc=$np1 archs rc=$archs"
[[ $np2 == 0 ]] || exit 30
[[ $eng == 0 ]] || exit 31
[[ $np1 == 0 ]] || exit 40
[[ $archs == 0 ]] || exit 50
exit 0
