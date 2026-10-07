#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:?output}
mkdir -p "$OUT"
W=${S7_W:-$HOME/repos/qwen38-eval-worktrees}
BUILDGATE=$ROOT/codex/build.sh
CPUGATE=${S7_GATE:-$W/bin/cpugate}
MODEL=${S7_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
JOBS=${S7_JOBS:-4}
GATE_SRC=src/models/delta-net-base.cpp
SHA=$(git rev-parse HEAD)
log() { echo "getter: $*"; }
die() { local rc=$1; shift; log "$*"; exit "$rc"; }
restore() { git restore --source=HEAD -- "$GATE_SRC"; }
trap restore EXIT
TEST=build-cpu/bin/test-recurrent-state-rollback
SHIM_EXT=so; [[ $(uname) != Darwin ]] || SHIM_EXT=dylib
# run_test NAME ARGS...: log in $OUT/NAME.log, rc in $RC; PRELOAD=lib preloads a shim into the test only (flat namespace on Darwin)
run_test() {
    local name=$1; shift
    RC=0
    local PRELOAD_ENV=(LD_PRELOAD="${PRELOAD:-}")
    if [[ $(uname) == Darwin ]]; then
        PRELOAD_ENV=()
        [[ -z ${PRELOAD:-} ]] || PRELOAD_ENV=(DYLD_INSERT_LIBRARIES="$PRELOAD" DYLD_FORCE_FLAT_NAMESPACE=1)
    fi
    "$CPUGATE" env ${PRELOAD_ENV[@]+"${PRELOAD_ENV[@]}"} LD_LIBRARY_PATH="$ROOT/build-cpu/bin" "$TEST" "$@" -m "$MODEL" -c 4096 -t 8 > "$OUT/$name.log" 2>&1 || RC=$?
}
ENGAGED='test_getter_gate: gdn nodes, one sequence: [1-9][0-9]* by id, 0 gathered; several: 0 by id, [1-9][0-9]* gathered'

run_test gate-full
[[ $RC == 0 ]] || { log "full test rc=$RC (see $OUT/gate-full.log)"; die 21 "full test failed on the unmutated build"; }
grep "$ENGAGED" "$OUT/gate-full.log" >/dev/null || die 21 "gate probe line missing or wrong in $OUT/gate-full.log"
run_test gate --getter-gate
[[ $RC == 0 ]] && grep "$ENGAGED" "$OUT/gate.log" >/dev/null || die 21 "gate test (see $OUT/gate.log)"
log "gate build: full test and gate probe pass"

# expect_fail NAME REASON: the run in $OUT/NAME.log (rc in $RC) must have failed with a log line matching REASON
declare -a RESULTS=("gate:pass")
expect_fail() {
    if [[ $RC != 0 ]] && grep "$2" "$OUT/$1.log" >/dev/null; then
        RESULTS+=("$1:fails")
        log "mutant $1 fails as required (rc=$RC)"
    else
        RESULTS+=("$1:ESCAPED rc=$RC")
        die 22 "mutant $1 escaped or failed for another reason (rc=$RC, expected /$2/; see $OUT/$1.log)"
    fi
}

PRELOAD="$ROOT/build-cpu/bin/row0-mutant.$SHIM_EXT" run_test mut-row0 --getter-gate
grep "ROW0_MUTANT active" "$OUT/mut-row0.log" >/dev/null || die 22 "row0 shim was not loaded (see $OUT/mut-row0.log)"
expect_fail mut-row0 "logits differ from the single-sequence context"

source_mutant() { # NAME SED_REPLACEMENT REASON
    sed "s/n_seqs == 1) {/$2) {/" "$GATE_SRC" > "$GATE_SRC.s7tmp" && cat "$GATE_SRC.s7tmp" > "$GATE_SRC" && rm -f "$GATE_SRC.s7tmp"
    git diff --quiet -- "$GATE_SRC" && die 8 "source mutant $1 did not change $GATE_SRC"
    "$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback -j"$JOBS" > "$OUT/build-$1.log" 2>&1 || die 8 "mutant $1 build (see $OUT/build-$1.log)"
    run_test "$1" --getter-gate
    printf "getter_mutant: %s engaged\n" "$1" >> "$OUT/$1.log"
    restore
    expect_fail "$1" "$3"
}
source_mutant mut-nogate 'true' "multi-sequence ubatches must keep the gathered state"
source_mutant mut-invert 'n_seqs != 1' "single-sequence ubatches must read the state by id"

"$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback -j"$JOBS" > "$OUT/build-restore.log" 2>&1 || die 23 "restore build (see $OUT/build-restore.log)"
[[ -z $(git status --porcelain --untracked-files=no) && $(git rev-parse HEAD) == "$SHA" ]] || die 23 "tree not clean or HEAD moved after the mutants"
run_test gate-restored --getter-gate
[[ $RC == 0 ]] && grep "$ENGAGED" "$OUT/gate-restored.log" >/dev/null || die 23 "gate test failed after the undo (see $OUT/gate-restored.log)"
log "restored; stage 1 passed"
