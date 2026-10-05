#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=/home/xanewok/repos/qwen38-eval PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [ "${1:-}" = --preflight ]; then
    exec nice -n 19 "$here/gpu_ab.sh" --parser-test
fi
if [ "${GPUQ_ACTIVE:-}" != 1 ]; then echo 'GPU queue required' >&2; exit 8; fi
mode=${1:?mode required}
out=${2:?output directory required}
mkdir -p "$out"
bin="$here/frozen/q8-dedupe-v2/build/bin"
start=$(date --iso-8601=seconds)
budget=${4:-600}
deadline=$((SECONDS + budget))
cap() {
    local left=$((deadline - SECONDS - 20))
    if [ "$left" -le 0 ]; then echo 'INVALID: budget expired' >> "$out/verdict.txt"; return 7; fi
    if [ "$left" -gt 80 ]; then left=80; fi
    echo "$left"
}
export LD_LIBRARY_PATH="$bin" GGML_CUDA_DEBUG_Q8_REUSE=1
unset GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_GRAPH_OPT GGML_CUDA_DEBUG_SSM_CONV
cleanup() {
    timeout 10 journalctl -k -b --since "$start" --no-pager > "$out/kernel-journal.txt" 2> "$out/kernel-journal.err" || true
}
trap cleanup EXIT
(cd "$bin"; sha256sum -c SHA256SUMS) > "$out/frozen-check.txt" 2>&1 || exit 6
status=0
for arm in reference patch; do
    limit=$(cap) || exit 7
    if timeout --signal=TERM --kill-after=10 "$limit" "$bin/q8-exact" "$arm" "$out/$arm.bin" > "$out/$arm.log" 2>&1; then :; else
        rc=$?
        echo "INVALID: arm=$arm rc=$rc" >> "$out/verdict.txt"
        if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then status=7; else status=8; fi
    fi
done
if [ "$status" = 0 ]; then
    if cmp -s "$out/reference.bin" "$out/patch.bin"; then echo 'PASS: bitwise outputs' >> "$out/verdict.txt"; else
        echo 'INVALID: bitwise outputs differ' >> "$out/verdict.txt"; status=8
    fi
fi
if [ "$mode" != --canary ]; then
    for device in CUDA0 CUDA1; do
        limit=$(cap) || exit 7
        if timeout --signal=TERM --kill-after=10 "$limit" "$bin/test-backend-ops" -b "$device" -o MUL_MAT_SHARED,RMS_NORM_MUL_ADD > "$out/backend-$device.log" 2>&1; then :; else
            rc=$?
            echo "INVALID: backend=$device rc=$rc" >> "$out/verdict.txt"
            if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then status=7; else status=8; fi
        fi
    done
fi
cleanup
if python3 - "$out" <<'PY'
import sys
import re
from pathlib import Path
from harness import lib
out = Path(sys.argv[1])
lib.verify_log_patterns((out/'reference.log').read_text(), [r'PASS: exact cases=126\b'], 'reference')
patch_log = (out/'patch.log').read_text()
lib.verify_log_patterns(patch_log, [r'PASS: exact cases=126\b',
    r'q8_reuse: device=0 graph=\d+ rows=6 skipped=[1-9]\d*\b',
    r'q8_reuse: device=1 graph=\d+ rows=6 skipped=[1-9]\d*\b',
    r'add_rms_norm: device=0 rows=6 columns=2560\b',
    r'add_rms_norm: device=1 rows=6 columns=2560\b'], 'patch')
for arm in ('reference', 'patch'):
    text = (out/f'{arm}.log').read_text()
    for device in (0, 1):
        # Every case has four computes; stable graphs evaluate directly/capture twice and replay twice.
        count = len(re.findall(rf'q8_reuse: device={device} graph=\d+ ', text))
        if count != 126:
            raise RuntimeError(f'graph replay gate: arm={arm} device={device} evaluations={count}, expected 126')
if re.search(lib.XID_RE, (out/'kernel-journal.txt').read_text()):
    raise SystemExit(3)
PY
then :; else
    rc=$?
    if [ "$rc" = 3 ]; then status=3; elif [ "$status" = 0 ]; then status=8; fi
fi
if [ "$status" != 0 ]; then exit "$status"; fi
echo 'PASS: exact outputs and engagement' >> "$out/verdict.txt"
