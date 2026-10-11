#!/usr/bin/env bash
# Post-run proofs on every measured boot of hostless-np2-real-iq4 (default verbosity: the hostless lines are WARN, the chain line
# INFO). Usage: postrun_proofs.sh RUN_DIR | --selftest
# both arms: `MTP chain: k=5 seqs=2`, no step-by-step fallback, no chain batch refusal, no chain refusal.
# L0 boots: no hostless line. L3 boots: level 3, the table and id slots on the meta device, deferred chain drafts read by the
# target batch, device ordering and the output lag engaged.
set -uo pipefail
L3_WANT=('hostless: level=3 mutant=none' 'hostless: tok_embd buffer=Meta' 'hostless: draft id slots=64 buffer=Meta'
    'hostless: chain drafts deferred k=5 seqs=' 'hostless: target batch reads device draft ids' 'hostless: device order engaged backends='
    'hostless: output lag engaged')
check_chain() {
    grep -aq 'MTP chain: k=5 seqs=2$' "$1" || return 1
    if grep -aq -e 'drafting step by step' -e 'the batch does not fit the MTP chain' -e 'MTP chain disabled' "$1"; then return 1; fi
}
check_l0() {
    check_chain "$1" || return 1
    if grep -aq 'hostless' "$1"; then return 1; fi
}
check_l3() {
    local p
    check_chain "$1" || return 1
    for p in "${L3_WANT[@]}"; do grep -aqF "$p" "$1" || return 1; done
    if grep -aq 'hostless verify: MISMATCH' "$1"; then return 1; fi
}
if [[ ${1:-} == --selftest ]]; then
    d=$(mktemp -d)
    trap 'rm -rf "$d"' EXIT
    printf 'I spec draft: MTP chain: k=5 seqs=2\n' > "$d/l0"
    { cat "$d/l0"; printf '%s\n' "${L3_WANT[@]}"; } > "$d/l3"
    head -n 7 "$d/l3" > "$d/l3-nolag"
    { cat "$d/l3"; printf 'hostless verify: MISMATCH row=3 device=1 host=2\n'; } > "$d/l3-mismatch"
    { cat "$d/l3"; printf 'W spec draft: llama_process[chain] returned -1; drafting step by step\n'; } > "$d/l3-fallback"
    check_l0 "$d/l0" && check_l3 "$d/l3" || { echo 'postrun_proofs: FAIL selftest (good logs)'; exit 8; }
    for bad in l3-nolag l3-mismatch l3-fallback l0; do
        if check_l3 "$d/$bad"; then echo "postrun_proofs: FAIL selftest (L3 accepts $bad)"; exit 8; fi
    done
    for bad in l3 l3-fallback; do
        if check_l0 "$d/$bad"; then echo "postrun_proofs: FAIL selftest (L0 accepts $bad)"; exit 8; fi
    done
    echo 'postrun_proofs: PASS selftest'; exit 0
fi
RUN=${1:?run dir}
n=0 bad=0
for dir in "$RUN"/[0-9]*-L0 "$RUN"/[0-9]*-L3; do
    [[ -f $dir/server.log ]] || continue
    n=$((n + 1))
    case $dir in
        *-L3) check_l3 "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: L3 not engaged"; bad=1; } ;;
        *-L0) check_l0 "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: L0 not plain M2"; bad=1; } ;;
    esac
done
[[ $n == 4 ]] || { echo "postrun_proofs: FAIL $RUN: $n boot logs, expected 4"; exit 5; }
[[ $bad == 0 ]] || exit 6
echo "postrun_proofs: PASS $n boots ($RUN)"
