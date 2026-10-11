#!/usr/bin/env bash
# Post-run engagement proofs on every measured boot of an embd-dev item (the engagement line is WARN, so default-verbosity
# boots carry it). Usage: postrun_proofs.sh RUN_DIR | --selftest
# patch boots: target, draft and fold rows on a non-CPU backend, IQ4_XS 2720 B rows, mutant=0 only. up boots (frozen/stack10): none.
set -uo pipefail
line() {  # NODE SHARED
    printf '%s' "embd_rows: engaged node=$1 backend=(?!CPU\\b)\\S+ shared=$2 type=iq4_xs rows=[1-9]\\d* row_bytes=2720 mutant=0\$"
}
check_patch() {
    grep -aPq "$(line '(?!mtp_)\S+' 0)" "$1" || return 1
    grep -aPq "$(line 'mtp_tok_embd-\d+' 0)" "$1" || return 1
    grep -aPq "$(line 'mtp_tok_embd-\d+' 1)" "$1" || return 1
    if grep -aPq 'embd_rows: .*mutant=[^0]' "$1"; then return 1; fi
}
check_up() {
    if grep -aq 'embd_rows:' "$1"; then return 1; fi
}
if [[ ${1:-} == --selftest ]]; then
    d=$(mktemp -d)
    trap 'rm -rf "$d"' EXIT
    l='0.01.0 W embd_rows: engaged node=%s backend=%s shared=%s type=iq4_xs rows=6 row_bytes=2720 mutant=%s\n'
    # shellcheck disable=SC2059
    { printf "$l" model.input_embed 'Meta(CUDA0,CUDA1)' 0 0; printf "$l" mtp_tok_embd-64 'Meta(CUDA0,CUDA1)' 0 0
      printf "$l" mtp_tok_embd-64 'Meta(CUDA0,CUDA1)' 1 0; } > "$d/patch"
    printf '0.01.0 W NCCL not compiled in\n' > "$d/up"
    grep -v 'shared=1' "$d/patch" > "$d/nofold"
    sed 's/backend=Meta(CUDA0,CUDA1)/backend=CPU/' "$d/patch" > "$d/cpu"
    sed '1s/mutant=0/mutant=2/' "$d/patch" > "$d/mutant"
    grep -v 'model.input_embed' "$d/patch" > "$d/notarget"
    check_patch "$d/patch" && check_up "$d/up" || { echo 'postrun_proofs: FAIL selftest (good logs)'; exit 8; }
    for bad in up nofold cpu mutant notarget; do
        if check_patch "$d/$bad"; then echo "postrun_proofs: FAIL selftest (patch accepts $bad)"; exit 8; fi
    done
    for bad in patch nofold; do
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
        *-patch) check_patch "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: device embedding rows not engaged (target, draft, fold)"; bad=1; } ;;
        *-up) check_up "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: control ran the device embedding rows"; bad=1; } ;;
    esac
done
[[ $n == 4 ]] || { echo "postrun_proofs: FAIL $RUN: $n boot logs, expected 4"; exit 5; }
[[ $bad == 0 ]] || exit 6
echo "postrun_proofs: PASS $n boots ($RUN)"
