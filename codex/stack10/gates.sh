#!/usr/bin/env bash
# Stage library of bw-stack10.sh (sourced, never run alone). Contract: a component's CUDA gate is that component's own function,
# extracted from its window by gates_table.py and run in a subshell with OUT=$OUT/<comp>; nothing here re-implements a check.
# Needs from the caller: ROOT W EVAL OUT SHA SHORT VISIBLE COMPONENTS CPUS GPUS CCCL BUILDGATE flags S9_LIB CAND_LIB P55_LIB.

has() { [[ " $COMPONENTS " == *" $1 "* ]]; }

# gpuq's fixed map: payload codes below are the window's own, so anything unknown is deterministic.
code_of() {
    case $1 in
        0|3|4|5|6|7) echo "$1" ;;
        124|137) echo 7 ;;
        *) echo 8 ;;
    esac
}

FAILS=()
STAGE_RCS=()
NRE=0
# The window's "FAIL" signature classifies a run as deterministic; hazard, integrity, void-control and transient codes must keep theirs.
fail_tag() { if [[ $1 == 8 ]]; then echo FAIL; else echo STOP; fi; }
note_stage() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$OUT/stages.tsv"; }

# Xid / fallen-off-the-bus since $2 (epoch) in the kernel journal; unreadable journal is an integrity failure.
journal_gate() {
    local name=$1 since=$2 rc=0
    PYTHONPATH="$EVAL" nice -n 19 python3 - "$OUT/$name.journal" "$since" <<'PY_JOURNAL' || rc=$?
import sys
from pathlib import Path
from harness import lib
n = lib.record_journal(float(sys.argv[2]), Path(sys.argv[1]))
raise SystemExit(6 if n is None else 3 if n else 0)
PY_JOURNAL
    return "$rc"
}

# Fatal stage: build, stamps and the stock reference must hold or no gate means anything.
fatal() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    rc=$(code_of "$rc")
    note_stage "$code" "$name" "$rc" "$((SECONDS-t0))"
    echo "bw-stack10: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || { echo "bw-stack10: $(fail_tag "$rc") $name rc=$rc"; finish_fail "$rc"; }
}

# Gate: a failure is recorded and the window goes on, so one run names every failing component for the drop map;
# an Xid stops everything at once.
gate() {
    local code=$1 name=$2 rc=0 jrc=0 since t0=$SECONDS
    shift 2
    since=$(date +%s)
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    rc=$(code_of "$rc")
    journal_gate "$name" "$since" || jrc=$?
    if [[ $jrc != 0 ]]; then rc=$jrc; fi
    note_stage "$code" "$name" "$rc" "$((SECONDS-t0))"
    echo "bw-stack10: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    if [[ $rc != 0 ]]; then
        FAILS+=("$name")
        STAGE_RCS+=("$rc")
        echo "bw-stack10: $(fail_tag "$rc") $name rc=$rc"
        [[ $rc != 3 ]] || finish_fail 3
    fi
    return 0
}

# Worst code of the failed gates: hazard, integrity, void control, deterministic, timeout, transient.
finish_fail() {
    local worst=${1:-} rc order
    if [[ -z $worst ]]; then
        for order in 3 6 5 8 7 4; do
            for rc in "${STAGE_RCS[@]}"; do
                if [[ $rc == "$order" ]]; then worst=$order; break 2; fi
            done
        done
    fi
    python3 - "$OUT" "$SHA" "$worst" "${FAILS[*]:-}" "$COMPONENTS" <<'PY_FAIL'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='FAIL', commit=sys.argv[2], rc=int(sys.argv[3]), failed_gates=sys.argv[4].split(),
                                              components=sys.argv[5].split(), frozen=False, items_rendered=False), indent=2)+'\n')
PY_FAIL
    echo "bw-stack10: $(fail_tag "$worst") gates=[${FAILS[*]:-}] rc=$worst; nothing frozen, nothing rendered"
    exit "$worst"
}

# One component's extracted gate functions, in OUT/<comp>, pinned to the window's CPUs like the p55 gate.
comp_run() {
    local comp=$1 fn
    shift
    mkdir -p "$OUT/$comp"
    (
        set -euo pipefail
        OUT=$OUT/$comp
        taskset -pc "$CPUS" "$BASHPID" > /dev/null
        eval "$(python3 codex/stack10/gates_table.py code "$comp")"
        for fn in "$@"; do "$fn"; done
    )
}

# Test binaries of the selected components exist only where the component is in the tree.
targets() {
    local t=(llama-server test-backend-ops)
    if has k4; then t+=(test-recurrent-state-rollback); fi
    if has k2; then t+=(test-fattn-kv-max-cache); fi
    if has k11b; then t+=(test-meta-input-stage); fi
    echo "${t[@]}"
}
build() {
    nice -n 19 "$BUILDGATE" cmake -S . -B build-cuda "${flags[@]}"
    nice -n 19 "$BUILDGATE" cmake --build build-cuda -j4 --target $(targets)
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}

stamps() {
    local n lib
    for lib in libggml-cuda.so libggml-base.so libllama.so; do [[ -s build-cuda/bin/$lib ]]; done
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null
    if has k9; then [[ $(grep -ac 'mmq_pdl:' build-cuda/bin/libggml-cuda.so) -gt 0 ]]; fi
    if has k11a; then [[ $(strings build-cuda/bin/libllama.so | grep -c 'backend busy at a skipped sync') -ge 1 ]]; fi
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}

# bin/wt freeze layout (RUNPATH $ORIGIN, SHA256SUMS, SOURCE) for the gates that run against a copy outside the tree.
assemble_bin() {
    local src=$1 dest=$2 source_line=$3 f so real
    mkdir "$dest"
    cp -p "$src/test-backend-ops" "$dest/"
    for so in "$src"/*.so*; do cp -a "$so" "$dest/"; done
    for f in "$dest"/*; do
        [[ -L $f ]] || patchelf --set-rpath '$ORIGIN' "$f"
    done
    real=$(cd "$dest" && pwd -P)
    while read -r f; do
        [[ $(readlink -f "$f") == "$real"/* ]] || { echo "resolves outside $dest: $f"; return 1; }
    done < <(env -u LD_LIBRARY_PATH ldd "$dest/test-backend-ops" | awk '/libggml/ {print $3}')
    [[ -z $(env -u LD_LIBRARY_PATH ldd "$dest/test-backend-ops" | grep 'not found' || true) ]]
    (cd "$dest" && find . -maxdepth 1 \( -type f -o -type l \) | sed 's#^\./##' | grep -v '^SHA256SUMS$' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS)
    printf '%s\n' "$source_line" > "$dest/SOURCE"
}

# Stock-MMQ reference: HEAD with the K8 and K9 ggml commits reverted (stock_tree.py verifies the MMQ files equal the base).
# Built in a bin/wt work dir (same depth as this tree, so the shared ccache hits on every unchanged unit).
stock_build() {
    STOCK_DIR=$("$W/bin/wt" work "s10-stock-a$ATTEMPT" --purpose "stack10 window attempt $ATTEMPT: HEAD with the K8 and K9 ggml commits reverted, the stock-MMQ reference of the K8 and K9 gates; scratch, safe to delete after the window")
    echo "STOCK_DIR=$STOCK_DIR" > "$OUT/stock-dir.txt"
    python3 codex/stack10/stock_tree.py make "$STOCK_DIR"
    nice -n 19 "$BUILDGATE" cmake -S "$STOCK_DIR" -B "$STOCK_DIR/build-cuda" "${flags[@]}"
    nice -n 19 "$BUILDGATE" cmake --build "$STOCK_DIR/build-cuda" --target test-backend-ops -j4
    local n
    n=$(strings "$STOCK_DIR/build-cuda/bin/libggml-cuda.so" | grep -c DeviceTopK)
    echo "stock DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    [[ $(grep -ac 'mmq_pdl:' "$STOCK_DIR/build-cuda/bin/libggml-cuda.so") == 0 ]]
    [[ $(grep -ac 'GGML_CUDA_MMQ_SK_MUTANT' "$STOCK_DIR/build-cuda/bin/libggml-cuda.so") == 0 ]]
    [[ $(grep -ac 'mmq-hash' "$STOCK_DIR/build-cuda/bin/test-backend-ops") -gt 0 ]]
    assemble_bin "$STOCK_DIR/build-cuda/bin" "$STOCK_DIR/stock-bin" "stock-mmq of $SHA (K8 and K9 ggml commits reverted)"
    assemble_bin build-cuda/bin "$STOCK_DIR/k8-bin" "$SHA CUDA 13.2 120a CCCL343 NCCL OFF, stack10 test-backend-ops build"
    [[ $(grep -ac 'GGML_CUDA_MMQ_SK_MUTANT' "$STOCK_DIR/k8-bin/libggml-cuda.so") -gt 0 ]]
}

g_base() { comp_run base ops; }
g_k7() { comp_run k7 ops; }
g_k4() { comp_run k4 ops model; }
g_k6a() { comp_run k6a ops; }
g_k6b() { comp_run k6b ops; }
g_k6_memcheck() { comp_run k6 memcheck; }
g_k2() { comp_run k2 ops; }
g_k10() { comp_run k10 ops controls; }
g_k9() {
    K=$ROOT/build-cuda/bin/test-backend-ops
    B=$STOCK_DIR/build-cuda/bin/test-backend-ops
    comp_run k9 ops
}

# K8: its own k8-gpu.sh, derived by k8_gpu_patch.py (stock arm = the stock-MMQ build, merged build assembled outside the tree).
g_k8() {
    mkdir -p "$OUT/k8"
    python3 codex/stack10/k8_gpu_patch.py codex/k8/gpu/k8-gpu.sh "$OUT/k8/k8-gpu.sh" "$STOCK_DIR/k8-bin" "$STOCK_DIR/stock-bin"
    bash -n "$OUT/k8/k8-gpu.sh"
    (cd "$ROOT/codex/k8/gpu" && python3 -m py_compile perf_summary.py)
    env CUDA_VISIBLE_DEVICES="$VISIBLE" COMMIT="$SHA" OUT="$OUT/k8/checks-canary" BUDGET_S=600 bash "$OUT/k8/k8-gpu.sh" checks --canary
    env CUDA_VISIBLE_DEVICES="$VISIBLE" COMMIT="$SHA" OUT="$OUT/k8/checks" BUDGET_S=9000 bash "$OUT/k8/k8-gpu.sh" checks --run
}

g_k11b() { comp_run k11b ops; }

# K11a: the component's sync-count capture (three server boots under nsys), run last and only when nothing else failed.
g_k11a() {
    mkdir -p "$OUT/k11a"
    [[ -n ${PORT:-} ]]
    env CUDA_VISIBLE_DEVICES="$VISIBLE" DRY_RUN=0 ROOT="$ROOT" OUT="$OUT/k11a/sync" PORT="$PORT" XID_SINCE="$STARTED" BUDGET_S=3000 \
        bash codex/k11a/sync_count.sh
    [[ -s $OUT/k11a/sync/sync-count.json ]]
}

# SASS gate on the multi-query q8_0 verify FA kernels: instances equal stack9's, and the regressed stack10-cand library fails the same gate.
g_sass() {
    mkdir -p "$OUT/sass"
    python3 codex/stack10/sass_gate.py "$OUT/sass" "$S9_LIB/libggml-cuda.so" build-cuda/bin/libggml-cuda.so --control "$CAND_LIB/libggml-cuda.so"
}
g_fafix() { comp_run fafix identity; }

# test-backend-ops under the window's CPU pin. Contract: rc 0 or 1 only (1 = a case failed, full_checks.py judges which);
# a crash or timeout is rc >= 2.
tbo_sql() {
    local bin=$1 limit=$2 out=$3 err=$4 rc=0
    shift 4
    nice -n 19 "$W/bin/cpugate" env CUDA_VISIBLE_DEVICES="$VISIBLE" taskset -c "$CPUS" timeout --kill-after=30s "$limit" \
        "$bin" "$@" --output sql > "$out" 2> "$err" || rc=$?
    echo "test-backend-ops $* rc_M=$rc"
    [[ $rc == 0 || $rc == 1 ]]
}

# One fresh rerun of a failing case: 0 it ran and passed, 2 it ran and failed, 1 it did not run, 9 the run crashed.
rerun_case() {
    local bin=$1 commit=$2 op=$3 dev=$4 params=$5 tag=$6 re rc=0
    re=$(python3 codex/stack10/full_checks.py --regex "$params")
    tbo_sql "$bin" 300s "$tag.sql" "$tag.err" -b "$dev" -o "$op" -p "$re" || return 9
    python3 codex/stack10/full_checks.py --rerun "$tag.sql" "$op" "$dev" "$commit" "$params" || rc=$?
    return "$rc"
}

# A failing row is excused only by evidence: scope out (a case no selected component's launch config reaches) passes 3 fresh
# reruns of this build; or the stack9-p55 binary fails the same case 3 of 3 (pre-existing, not a regression). Scope in
# (changed quants, MMQ columns) has only the second way.
excused() {
    local scope=$1 op=$2 dev=$3 params=$4 tag=$5 a rc ok=1
    if [[ $scope == out ]]; then
        for a in 1 2 3; do
            rc=0
            rerun_case build-cuda/bin/test-backend-ops "$SHA" "$op" "$dev" "$params" "$tag.new$a" || rc=$?
            if [[ $rc != 0 ]]; then ok=0; break; fi
        done
        if [[ $ok == 1 ]]; then echo "excused after 3 clean reruns: $op $params"; return 0; fi
    fi
    for a in 1 2 3; do
        rc=0
        rerun_case "$P55_LIB/test-backend-ops" "$P55_SHA" "$op" "$dev" "$params" "$tag.base$a" || rc=$?
        [[ $rc == 2 ]] || { echo "NOT excused ($scope, base rerun $a rc=$rc): $op $params"; return 1; }
    done
    echo "excused, fails 3 of 3 on stack9-p55 too: $op $params"
}

# The only gate that runs every case of the suite: byte-identity markers of the base and selected components, no FAIL line,
# per-op coverage, and flake handling after p55_checks.py. Covers p55's MMVQ gate (changed quants, n=5..8 coverage).
g_full() {
    local dev base scope op params csv
    mkdir -p "$OUT/full"
    python3 codex/stack10/full_checks.py --fixtures
    csv=base,$(echo "$COMPONENTS" | tr ' ' ',')
    for dev in CUDA0 CUDA1; do
        base=$OUT/full/$dev
        tbo_sql build-cuda/bin/test-backend-ops 12600s "$base.sql" "$base.err" -b "$dev"
        # One failing case skips the K7 and K2 self-tests (test-backend-ops chains them after `n_ok == tests_run` with &&),
        # so their markers come from a run that selects no case.
        tbo_sql build-cuda/bin/test-backend-ops 1800s "$base.self.sql" "$base.self.err" -b "$dev" \
            -o RMS_NORM_SCALE_PAIR,FLASH_ATTN_EXT -p no-such-case
        nice -n 19 python3 codex/stack10/full_checks.py check "$base.sql" "$dev" "$SHA" "$csv" "$base.self.sql"
        while IFS=$'\t' read -r -u 3 scope op params; do
            [[ -n $params ]] || continue
            excused "$scope" "$op" "$dev" "$params" "$base.re$((++NRE))"
        done 3< "$base.sql.retry"
    done
}

freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    [[ ! -e $W/frozen/stack10 ]]
    nice -n 19 "$W/bin/wt" freeze "$ROOT" stack10 --build build-cuda --desc "stack9-p55 + K7 K4 K6a K6b K2 K10 K9 K8 K11a K11b ($COMPONENTS); CUDA gates passed on both GPUs; CUDA 13.2 120a CCCL343 NCCL OFF"
    [[ $(head -n 1 "$W/frozen/stack10/SOURCE") == "$SHA" ]]
    {
        printf 'commit %s\nconfigure: cmake -S . -B build-cuda ' "$SHA"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: bin/buildgate cmake --build build-cuda -j4 --target %s\nbuild stage: CUDA_VISIBLE_DEVICES=""; gates ran in codex/window (components: %s)\n' "$(targets)" "$COMPONENTS"
    } > "$W/frozen/stack10/RECIPE.txt"
}

items() {
    nice -n 19 python3 codex/stack10/render_stack10_items.py --out "$ROOT/codex/stack10/rendered" --components "$COMPONENTS"
}
