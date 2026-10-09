#!/usr/bin/env bash
# CPU model of the chunked AllReduce kernel protocol for the allreduce.cu of REV (default HEAD). CPU only, no CUDA.
# usage: ar-duplex-cpumodel.sh [REV [quick|mutated]]   rc: 0 pass | 1 a case failed
# mutated: the flag wait of the kernel is compiled out; the run passes only if the identity check catches it.
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 CUDA_VISIBLE_DEVICES=''
HERE=$(cd "$(dirname "$0")" && pwd)
REV=${1:-HEAD}
OUT=${AR_MODEL_OUT:-$(mktemp -d)}
git -C "$HERE/.." show "$REV:ggml/src/ggml-cuda/allreduce.cu" > "$OUT/allreduce.cu"
python3 - "$OUT" <<'PY'
import re, sys
src = open(sys.argv[1] + '/allreduce.cu').read()
m = re.search(r'^// (Byte spacing|Number of blocks the chunked kernel)', src, re.M)
e = src.index('// Combined load-convert-add kernel.')
body = src[m.start():e]
open(sys.argv[1] + '/kernel.h', 'w').write(body)
open(sys.argv[1] + '/local', 'w').write('1' if 'int                         mutant' in src else '0')
PY
if [[ ${2:-} == mutated ]]; then
    n=$(grep -c 'while (ggml_cuda_ar_signal_get[_a-z]*(other_slot) != token)' "$OUT/kernel.h" || true)
    [[ $n == 1 ]] || { echo "cpumodel: mutation site count $n != 1"; exit 6; }
    sed -i 's/while (ggml_cuda_ar_signal_get[_a-z]*(other_slot) != token)/while (false)/' "$OUT/kernel.h"
fi
LOCAL=$(cat "$OUT/local")
g++ -std=c++20 -O1 -pthread -DKERNEL_HEADER="\"$OUT/kernel.h\"" -DAR_API_LOCAL="$LOCAL" "$HERE/ar-duplex-cpumodel.cpp" -o "$OUT/model"
exec nice -n 19 taskset -c 6-11,18-23 "$OUT/model" "${2:-}"
