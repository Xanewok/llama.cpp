#!/usr/bin/env bash
# Post-run engagement proofs on every measured boot of an h2d-contig item (the engagement line is WARN, so default-verbosity
# boots carry it). Usage: postrun_proofs.sh RUN_DIR | --selftest
# patch boots: `input_scatter: engaged device=0|1 ... mutant=0` for both devices, no mutant line. up boots (frozen/stack10): none.
set -uo pipefail
check_patch() {
    grep -aEq 'input_scatter: engaged device=0 entries=[1-9][0-9]* bytes=[1-9][0-9]* mutant=0$' "$1" || return 1
    grep -aEq 'input_scatter: engaged device=1 entries=[1-9][0-9]* bytes=[1-9][0-9]* mutant=0$' "$1" || return 1
    if grep -aq 'input_scatter: .*mutant=1' "$1"; then return 1; fi
}
check_up() {
    if grep -aq 'input_scatter:' "$1"; then return 1; fi
}
if [[ ${1:-} == --selftest ]]; then
    d=$(mktemp -d)
    trap 'rm -rf "$d"' EXIT
    printf '0.01.0 W input_scatter: engaged device=0 entries=5 bytes=213376 mutant=0\n0.01.1 W input_scatter: engaged device=1 entries=5 bytes=213376 mutant=0\n' > "$d/patch"
    printf '0.01.0 W NCCL not compiled in\n' > "$d/up"
    printf '0.01.0 W input_scatter: engaged device=0 entries=5 bytes=213376 mutant=0\n' > "$d/onedev"
    sed 's/mutant=0/mutant=1/' "$d/patch" > "$d/mutant"
    check_patch "$d/patch" && check_up "$d/up" || { echo 'postrun_proofs: FAIL selftest (good logs)'; exit 8; }
    for bad in up onedev mutant; do
        if check_patch "$d/$bad"; then echo "postrun_proofs: FAIL selftest (patch accepts $bad)"; exit 8; fi
    done
    for bad in patch onedev; do
        if check_up "$d/$bad"; then echo "postrun_proofs: FAIL selftest (up accepts $bad)"; exit 8; fi
    done
    echo 'postrun_proofs: PASS selftest'; exit 0
fi
RUN=${1:?run dir}
n=0 bad=0
for dir in "$RUN"/[0-9]*-up "$RUN"/[0-9]*-patch; do
    [[ -f $dir/server.log ]] || continue
    n=$((n + 1))
    case $dir in
        *-patch) check_patch "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: packed upload not engaged on both devices"; bad=1; } ;;
        *-up) check_up "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: control ran the packed upload"; bad=1; } ;;
    esac
done
[[ $n == 4 ]] || { echo "postrun_proofs: FAIL $RUN: $n boot logs, expected 4"; exit 5; }
[[ $bad == 0 ]] || exit 6
echo "postrun_proofs: PASS $n boots ($RUN)"
