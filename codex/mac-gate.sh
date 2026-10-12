#!/usr/bin/env bash
# Mac CPU gate (bin/bw-mac wt/ar-readfirst codex/mac-gate.sh): CPU protocol model of HEAD's chunked kernel (the local commit's
# kernel text) in read-first and store-first order, quick + wait-removed mutant; syntax of the window files; perf-gate and
# postrun-proof selftests.
# No CUDA on the Mac: allreduce.cu, the driver and the window are not compiled or run here.
# rc: 0 pass | 1 a model case failed | 5 negative control void | 7 a step hit its time cap
set -euo pipefail
OUT=${1:?OUT}
cd "$(dirname "$0")/.."
cap() { perl -e 'alarm shift; exec @ARGV or die' "$@"; }   # no coreutils timeout on the Mac
step() {  # NAME CAP_S CMD...: output $OUT/NAME.out, status line in $OUT/stages.tsv
    local name=$1 c=$2 rc=0 t0=$SECONDS
    shift 2
    cap "$c" "$@" > "$OUT/$name.out" 2>&1 || rc=$?
    printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$OUT/stages.tsv"
    [[ $rc == 142 ]] && { echo "mac-gate: $name hit its ${c}s cap"; exit 7; }
    return "$rc"
}
export AR_MODEL_RUN='nice -n 19'
for order in rf sf; do
    AR_MODEL_OUT=$(mktemp -d) step "model-quick-$order" 240 bash codex/ar-cpumodel.sh WORKTREE quick "$order" || { echo "mac-gate: FAIL model quick $order"; exit 1; }
    grep -aq '^cpumodel: PASS' "$OUT/model-quick-$order.out" || { echo "mac-gate: FAIL model quick $order printed no PASS"; exit 1; }
    grep -aq "^model-config order=$order store_first_api=1" "$OUT/model-quick-$order.out" || { echo "mac-gate: FAIL order $order not applied"; exit 1; }
    rc=0
    AR_MODEL_OUT=$(mktemp -d) step "model-mutated-$order" 120 bash codex/ar-cpumodel.sh WORKTREE mutated "$order" || rc=$?
    [[ $rc == 0 ]] && grep -aq '^cpumodel-mutated: PASS' "$OUT/model-mutated-$order.out" || { echo "mac-gate: negative control void ($order, rc=$rc)"; exit 5; }
done
for s in codex/*.sh; do bash -n "$s"; done
step summary-selftest 60 python3 codex/ar_summary.py --selftest || { echo 'mac-gate: FAIL ar_summary selftest'; exit 1; }
step postrun-selftest 60 bash codex/postrun_proofs.sh --selftest || { echo 'mac-gate: FAIL postrun_proofs selftest'; exit 1; }
python3 - <<'PY'
from pathlib import Path
for p in Path('codex').glob('*.py'):
    compile(p.read_text(), str(p), 'exec')
PY
echo 'mac-gate: PASS'
