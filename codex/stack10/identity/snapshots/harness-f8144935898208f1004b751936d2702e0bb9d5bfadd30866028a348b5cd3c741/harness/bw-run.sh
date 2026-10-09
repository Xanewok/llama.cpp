#!/usr/bin/env bash
# gpuq owns the build slot and kills the whole cgroup on expiry.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 GPUQ_BUILD_WINDOW=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
script=${1:?payload} worktree=${2:?worktree} head_file=${3:?HEAD file} out=${4:?out dir or --preflight}
code=$(dirname "$(dirname "$script")")
expected=$(cat "$head_file")
export BW_WORKTREE="$worktree" BW_CODEX="$code/codex"
check_output() {
    local path base prefix
    path=$(readlink -m "$1") || return 1
    base=$(basename "$path") prefix=${BW_NAME:?window name}-a
    [[ $(dirname "$path") == "${BW_OUT_ROOT:?output root}" &&
       $base == "$prefix"* && ${base#"$prefix"} =~ ^[0-9]+$ ]]
}
check_tree() {
    local actual dirty
    actual=$(git -C "$worktree" rev-parse HEAD) || return 1
    dirty=$(git -C "$worktree" status --porcelain --untracked-files=no) || return 1
    [[ $actual == "$expected" && -z $dirty ]] || return 1
    (cd "$worktree" && sha256sum --check --quiet "$code/SOURCES.sha256") || return 1
}
payload() {
    # Existing payloads use $0 to find their source tree; source the frozen bytes with that argv[0].
    /usr/bin/bash -c 'source "$1" "${@:2}"' "$worktree/codex/$(basename "$script")" "$script" "$@"
}
cd "$worktree"
if [[ $out == --preflight ]]; then
    if ! check_output "$BW_OUT_ROOT/$BW_NAME-a1"; then echo 'bw-run: FAIL output path'; exit 10; fi
    if ! check_tree; then echo 'bw-run: FAIL inputs changed: dirty tree, HEAD or script bytes'; exit 10; fi
    for path in "$code"/codex/*.sh; do [[ ! -e $path ]] || bash -n "$path" || exit 10; done
    python3 - "$code/codex" <<'PY' || exit 10
import sys
from pathlib import Path
for path in Path(sys.argv[1]).glob('*.py'):
    compile(path.read_text(), str(path), 'exec')
PY
    if payload --preflight; then echo "bw-run: preflight HEAD $expected"; exit 0; fi
    echo 'bw-run: FAIL payload preflight'; exit 10
fi
if ! check_output "$out"; then echo 'bw-run: FAIL output path'; exit 6; fi
mkdir -p "$out"
if ! check_tree; then echo 'bw-run: FAIL inputs changed: dirty tree, HEAD or script bytes'; exit 6; fi
printf '%s\n' "$expected" > "$out/source-commit.txt"
set +e
timeout --signal=TERM --kill-after=30 "${BUDGET_S:?run budget}" /usr/bin/bash -c 'source "$1" "${@:2}"' \
    "$worktree/codex/$(basename "$script")" "$script" "$out" > "$out/payload.log" 2>&1
rc=$?
set -e
if [[ $rc == 124 || $rc == 137 ]]; then
    echo 'bw-run: FAIL rc=7 timeout' | tee -a "$out/payload.log"
    exit 7
fi
ok_rcs=${OK_RCS:-0}
for ok in ${ok_rcs//,/ }; do
    if [[ $rc == "$ok" ]]; then
        if ! check_tree; then
            echo 'bw-run: FAIL rc=6 inputs changed during payload' | tee -a "$out/payload.log"
            exit 6
        fi
        echo "bw-run: $(date '+%H:%M:%S') payload rc=0 (original rc=$rc; $script)" | tee -a "$out/payload.log"
        exit 0
    fi
done
echo "bw-run: $(date '+%H:%M:%S') payload rc=$rc ($script)" | tee -a "$out/payload.log"
echo 'bw-run: FAIL rc=8 payload' | tee -a "$out/payload.log"
exit 8
