#!/usr/bin/env bash
# Post-run engagement proofs on every measured boot of a spec-prompt item (the engagement line is WARN, so default-verbosity
# boots carry it). Usage: postrun_proofs.sh RUN_DIR | --selftest
# patch boots: `spec_prompt: engaged, no drafter reads the prompt` once. up boots (frozen/stack10): none.
set -uo pipefail
PAT='spec_prompt: engaged, no drafter reads the prompt'
check_patch() {
    [[ $(grep -ac "$PAT" "$1") == 1 ]]
}
check_up() {
    ! grep -aq 'spec_prompt:' "$1"
}
if [[ ${1:-} == --selftest ]]; then
    d=$(mktemp -d)
    trap 'rm -rf "$d"' EXIT
    printf '0.01.0 W srv load_model: spec_prompt: engaged, no drafter reads the prompt, per-round copy skipped\n' > "$d/patch"
    printf '0.01.0 W NCCL not compiled in\n' > "$d/up"
    cat "$d/patch" "$d/patch" > "$d/twice"
    check_patch "$d/patch" && check_up "$d/up" || { echo 'postrun_proofs: FAIL selftest (good logs)'; exit 8; }
    for bad in up twice; do
        if check_patch "$d/$bad"; then echo "postrun_proofs: FAIL selftest (patch accepts $bad)"; exit 8; fi
    done
    if check_up "$d/patch"; then echo 'postrun_proofs: FAIL selftest (up accepts patch)'; exit 8; fi
    echo 'postrun_proofs: PASS selftest'; exit 0
fi
RUN=${1:?run dir}
n=0 bad=0
for dir in "$RUN"/[0-9]*-up "$RUN"/[0-9]*-patch; do
    [[ -f $dir/server.log ]] || continue
    n=$((n + 1))
    case $dir in
        *-patch) check_patch "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: prompt copy not skipped"; bad=1; } ;;
        *-up) check_up "$dir/server.log" || { echo "postrun_proofs: FAIL $dir: control carries the patch"; bad=1; } ;;
    esac
done
[[ $n == 4 ]] || { echo "postrun_proofs: FAIL $RUN: $n boot logs, expected 4"; exit 5; }
[[ $bad == 0 ]] || exit 6
echo "postrun_proofs: PASS $n boots ($RUN)"
