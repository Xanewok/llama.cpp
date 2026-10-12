#!/usr/bin/env bash
# CPU model of the chunked AllReduce kernel protocol for the allreduce.cu of REV (default HEAD). CPU only, no CUDA.
# usage: ar-cpumodel.sh [REV|WORKTREE [quick|mutated|full [rf|sf]]]   rc: 0 pass | 1 a case failed
# rf/sf: the local kernel's order (read-first default, store-first = the ar-duplex order); ignored by kernels without the knob.
# mutated: the flag wait of the kernel is compiled out; the run passes only if the identity check catches it.
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 CUDA_VISIBLE_DEVICES=''
HERE=$(cd "$(dirname "$0")" && pwd)
REV=${1:-HEAD}
OUT=${AR_MODEL_OUT:-$(mktemp -d)}
if [[ $REV == WORKTREE ]]; then
    cp "$HERE/../ggml/src/ggml-cuda/allreduce.cu" "$OUT/allreduce.cu"   # bw-mac bundles carry HEAD only
else
    git -C "$HERE/.." show "$REV:ggml/src/ggml-cuda/allreduce.cu" > "$OUT/allreduce.cu"
fi
python3 - "$OUT" <<'PY'
import re, sys
src = open(sys.argv[1] + '/allreduce.cu').read()
m = re.search(r'^// (Byte spacing|Number of blocks the chunked kernel)', src, re.M)
e = src.index('// Combined load-convert-add kernel.')
body = src[m.start():e]
open(sys.argv[1] + '/kernel.h', 'w').write(body)
open(sys.argv[1] + '/local', 'w').write('1' if 'int                         mutant' in src else '0')
open(sys.argv[1] + '/flag_t', 'w').write('uint64_t' if 'uint64_t *                  arrival_mine' in src else 'int')
open(sys.argv[1] + '/sfapi', 'w').write('1' if 'int                         store_first' in src else '0')
open(sys.argv[1] + '/multi', 'w').write('1' if 'FLAG_WORDS' in body or 'FLAG_INTS' in body else '0')
PY
if [[ ${2:-} == mutated ]]; then
    n=$(grep -c 'while (ggml_cuda_ar_signal_get[_a-z]*(other_slot) != token)' "$OUT/kernel.h" || true)
    [[ $n == 1 ]] || { echo "cpumodel: mutation site count $n != 1"; exit 6; }
    sed -i.orig 's/while (ggml_cuda_ar_signal_get[_a-z]*(other_slot) != token)/while (false)/' "$OUT/kernel.h"
fi
LOCAL=$(cat "$OUT/local")
g++ -std=c++20 -O1 -pthread -DKERNEL_HEADER="\"$OUT/kernel.h\"" -DAR_API_LOCAL="$LOCAL" -DFLAG_T="$(cat "$OUT/flag_t")" -DMULTI_FLAG="$(cat "$OUT/multi")" -DSTORE_FIRST_API="$(cat "$OUT/sfapi")" "$HERE/ar-cpumodel.cpp" -o "$OUT/model"
# AR_MODEL_RUN: runner prefix (the Mac has no cpugate/taskset)
if [[ -n ${AR_MODEL_RUN:-} ]]; then
    # shellcheck disable=SC2086
    exec $AR_MODEL_RUN "$OUT/model" "${2:-}" "${3:-rf}"
fi
exec "$HOME/repos/qwen38-eval-worktrees/bin/cpugate" nice -n 19 taskset -c 6-11,18-23 "$OUT/model" "${2:-}" "${3:-rf}"
