#!/usr/bin/env bash
# Stage library of bw-stack11-s2.sh (sourced, never run alone). Helpers of the stack11-base window (fatal/gate/finish_fail) as they were,
# then the s2 stages. A gate runs in a subshell under set -e with OUT=$OUT/<dir>, pinned to $CPUS.
# Needs from the caller: ROOT KROOT W EVAL OUT SHA KSHA SHORT KSHORT VISIBLE STARTED COMPONENTS CPUS CCCL BUILDGATE flags BASE_FROZEN WIN.

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
# The window's "FAIL" signature classifies a run as deterministic; hazard, integrity, void-control and transient codes keep theirs.
fail_tag() { if [[ $1 == 8 ]]; then echo FAIL; else echo STOP; fi; }
note_stage() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$OUT/stages.tsv"; }

# Xid / fallen-off-the-bus since $2 (epoch) in the kernel journal; an unreadable journal is an integrity failure.
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

# Fatal stage: build, stamps, the np2 smoke, freeze and render must hold or nothing after them means anything.
fatal() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    rc=$(code_of "$rc")
    note_stage "$code" "$name" "$rc" "$((SECONDS-t0))"
    echo "bw-s2: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || { echo "bw-s2: $(fail_tag "$rc") $name rc=$rc"; finish_fail "$rc"; }
}

# Gate: a failure is recorded and the window goes on, so one run names every failing component for the drop map; an Xid stops at once.
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
    echo "bw-s2: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    if [[ $rc != 0 ]]; then
        FAILS+=("$name")
        STAGE_RCS+=("$rc")
        echo "bw-s2: $(fail_tag "$rc") $name rc=$rc"
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
    echo "bw-s2: $(fail_tag "$worst") gates=[${FAILS[*]:-}] rc=$worst; nothing frozen, nothing rendered"
    exit "$worst"
}

# DIR FN...: the gate functions in OUT/DIR on the window's CPUs.
in_dir() {
    local dir=$1 fn
    shift
    mkdir -p "$OUT/$dir"
    (
        set -euo pipefail
        OUT=$OUT/$dir
        taskset -pc "$CPUS" "$BASHPID" > /dev/null
        for fn in "$@"; do "$fn"; done
    )
}

# LOG DEV TIMEOUT [ENV=VAL...] -- ARGS...: build-cuda test-backend-ops in test mode on one device; rc in RC.
tb() {
    local log=$1 dev=$2 tmo=$3
    shift 3
    local envs=()
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ${envs[@]+"${envs[@]}"} timeout --kill-after=30s "$tmo" nice -n 19 build-cuda/bin/test-backend-ops test \
        -b "$dev" "$@" -j 1 > "$log" 2>&1 || RC=$?
}
# LOG [MIN]: the last tb ran clean: rc 0, no FAIL line, every "N/M tests passed" has N == M, at least MIN cases in all.
all_passed() {
    local n
    [[ $RC == 0 ]] || { echo "rc=$RC: $1"; return 1; }
    if grep -aq 'FAIL' "$1"; then echo "FAIL line in $1"; return 1; fi
    n=$(grep -aEo '[0-9]+/[0-9]+ tests passed' "$1" | awk -F'[ /]' '{ if ($1 != $2) bad = 1; s += $1 } END { print bad ? -1 : s + 0 }') || true
    (( ${n:-0} >= ${2:-1} )) || { echo "passed cases ${n:-0} < ${2:-1}: $1"; return 1; }
    echo "pass: $(basename "$1") cases=$n"
}


# --- s2 stages ------------------------------------------------------------------------------------------------------------------------
# k = KROOT (frozen/stack11-s2: stack11-base + 87/88/91/90, K7 and K10 kept); nok = ROOT (frozen/stack11-s2-nok: K7 and K10 removed).
tree_of() { case $1 in k) echo "$KROOT" ;; nok) echo "$ROOT" ;; *) return 1 ;; esac; }
frozen_of() { case $1 in k) echo stack11-s2 ;; nok) echo stack11-s2-nok ;; *) return 1 ;; esac; }
targets() { echo llama-server test-backend-ops test-alloc test-host-mapped test-speculative; }

build_tree() {
    cd "$(tree_of "$1")"
    mkdir -p /mnt/scratch/tmp-build
    TMPDIR=/mnt/scratch/tmp-build nice -n 19 "$BUILDGATE" cmake -S . -B build-cuda "${flags[@]}"
    # shellcheck disable=SC2046
    TMPDIR=/mnt/scratch/tmp-build nice -n 19 "$BUILDGATE" cmake --build build-cuda -j4 --target $(targets)
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
build_k() { build_tree k; }
build_nok() { build_tree nok; }

stamps() {
    local t d so lib n m
    for t in k nok; do
        d=$(tree_of "$t")
        so=$d/build-cuda/bin
        for lib in libggml-cuda.so libggml-base.so libllama.so libllama-common.so; do [[ -s $so/$lib ]]; done
        [[ $(strings "$so/libggml-base.so" | grep -cx "$(git -C "$d" rev-parse --short=9 HEAD)") -ge 1 ]]
        [[ $(grep -ac 'CUDA_Host_Mapped' "$so/libggml-cuda.so") -ge 1 ]]
        [[ $(grep -ac 'fattn_kv_max' "$so/libggml-cuda.so") == 0 ]]
        n=$(grep -ac 'rms_norm_scale_pair_fusion: device' "$so/libggml-cuda.so") || true
        m=$(grep -ac 'top_k_batched: engaged' "$so/libggml-cuda.so") || true
        echo "$t: rms_norm_scale_pair_fusion_count_M=$n top_k_batched_count_M=$m"
        if [[ $t == k ]]; then [[ $n -ge 1 && $m -ge 1 ]]; else [[ $n == 0 && $m == 0 ]]; fi
        [[ -z $(git -C "$d" status --porcelain --untracked-files=no) ]]
        (cd "$so" && sha256sum round-trace llama-server test-backend-ops test-host-mapped test-alloc ./*.so) > "$OUT/build-sha256-$t.txt"
    done
    [[ $(git -C "$KROOT" rev-parse HEAD) == "$KSHA" && $(git -C "$ROOT" rev-parse HEAD) == "$SHA" ]]
}

# The stack10 crash: both slots prefilled at 8k, joint replay, 3 repetitions must survive; the pre-fix mutant must die.
np2_smoke() {
    [[ -n ${PORT:-} ]]
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$KROOT" PORT="$PORT" XID_SINCE="$STARTED" bash "$WIN/np2_smoke.sh" "$OUT/np2-smoke"
}

# CPU-device unit tests of the stack (the Mac gate's own tests) on the CUDA build of each tree.
unit_tree() {
    local t=$1 d n
    d=$(tree_of "$t")
    timeout --kill-after=10s 600 env CUDA_VISIBLE_DEVICES='' nice -n 19 "$d/build-cuda/bin/test-alloc" > "$OUT/test-alloc-$t.txt" 2>&1
    for n in test_meta_subgraph_realloc_count test_meta_subgraph_realloc_size test_meta_subgraph_reuse_two_splits \
             test_meta_subgraph_reuse_ten_splits test_meta_subgraph_reuse_same_graph test_meta_host_mapped_weight; do
        grep -aq "$n PASSED" "$OUT/test-alloc-$t.txt" || { echo "missing: $n PASSED"; return 8; }
    done
    timeout --kill-after=10s 600 env CUDA_VISIBLE_DEVICES='' nice -n 19 "$d/build-cuda/bin/test-speculative" > "$OUT/test-speculative-$t.txt" 2>&1
    grep -aq 'uses_prompt' "$OUT/test-speculative-$t.txt" || { echo 'missing: uses_prompt'; return 8; }
}
unit_k() { unit_tree k; }
unit_nok() { unit_tree nok; }
g_unit() { in_dir unit unit_k unit_nok; }

# pass: rc 0 and every required line present
gate_log() {  # LOG RC PATTERN...
    local log=$1 rc=$2 p
    shift 2
    [[ $rc == 0 ]] || return 8
    for p in "$@"; do grep -aEq "$p" "$log" || { echo "missing: $p"; return 8; }; done
}
# test-host-mapped, mapped buffer in use: bitwise for >= 100 type x shape cases per device, sched placement, ops rejected
gate_mapped() {  # LOG RC
    local dev n
    gate_log "$1" "$2" '^test-host-mapped: PASS \(2 devices\)$' || return 8
    for dev in CUDA0 CUDA1; do
        gate_log "$1" 0 "^host_mapped: dev=$dev buft=CUDA_Host_Mapped$" "^host_mapped: dev=$dev bitwise OK, sched OK$" \
            "^host_mapped: dev=$dev type=iq4_xs rows=512x1 differing=0/131072$" \
            "^host_mapped: dev=$dev types=2[0-9] supports_op mul_mat=0 add=0 \(want 0 0\)$" \
            "^host_mapped: dev=$dev control cpu-buffer get_rows=CPU; forced to device: copied=1$" \
            "^host_mapped: dev=$dev fallback=0 get_rows=$dev splits=1 weight_copied=0 equal=1$" \
            "^host_mapped: dev=$dev same weight: get_rows=$dev mul_mat=CPU weight_copied=0$" || return 8
        n=$(grep -ac "^host_mapped: dev=$dev type=.* differing=0/" "$1" || true)
        [[ $n -ge 100 ]] || { echo "only $n bitwise cases on $dev"; return 8; }
    done
    if grep -aEq 'FAIL|differing=[1-9]' "$1"; then return 8; fi
}
gate_nopin() {  # LOG RC
    local dev
    gate_log "$1" "$2" '^test-host-mapped: PASS \(2 devices\)$' || return 8
    for dev in CUDA0 CUDA1; do
        gate_log "$1" 0 "^host_mapped: dev=$dev fallback=1, bitwise test skipped$" \
            "^host_mapped: dev=$dev fallback=1 get_rows=CPU splits=[0-9]+ weight_copied=0 equal=1$" \
            "^host_mapped: dev=$dev same weight: get_rows=CPU mul_mat=CPU weight_copied=0$" || return 8
    done
    if grep -aq 'FAIL' "$1"; then return 8; fi
}
# stderr apart: ggml_cuda_init prints mid-line and breaks anchored gates
host_mapped_run() {  # TREE TAG [ENV=VAL]
    local rc=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ${3:+"$3"} timeout 900 nice -n 19 "$(tree_of "$1")/build-cuda/bin/test-host-mapped" \
        > "$OUT/thm-$1-$2.log" 2> "$OUT/thm-$1-$2.err" || rc=$?
    return "$rc"
}
# mapped buffer in use; the NO_PINNED control must fall back to a CPU buffer with equal values (a fallback that did not happen
# would not exercise the CPU gather path)
mapped_tree() {
    local t=$1 rc=0
    host_mapped_run "$t" zc || rc=$?
    gate_mapped "$OUT/thm-$t-zc.log" "$rc" || return 8
    rc=0
    host_mapped_run "$t" nopin GGML_CUDA_NO_PINNED=1 || rc=$?
    gate_nopin "$OUT/thm-$t-nopin.log" "$rc" || return 8
}
mapped_k() { mapped_tree k; }
mapped_nok() { mapped_tree nok; }
g_mapped() { in_dir mapped mapped_k mapped_nok; }

# OPS per tree on both GPUs: rc 0, at least one passing row, and every failing case also fails on frozen/stack11-base (pre-existing)
ops_tree() {  # TREE OPS
    local t=$1 ops=$2 dev who bin rc n
    for dev in CUDA0 CUDA1; do
        for who in new base; do
            if [[ $who == new ]]; then bin=$(tree_of "$t")/build-cuda/bin/test-backend-ops; else bin=$BASE_FROZEN/build/bin/test-backend-ops; fi
            rc=0
            env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout --kill-after=30s 3600 nice -n 19 "$bin" test -b "$dev" -o "$ops" -j 1 \
                > "$OUT/ops-$t-$dev-$who.txt" 2>&1 || rc=$?
            echo "ops $t $dev $who rc_M=$rc"
            [[ $rc == 0 || $rc == 1 ]] || return 8
            grep -a 'FAIL' "$OUT/ops-$t-$dev-$who.txt" | sort -u > "$OUT/ops-$t-$dev-$who.fail" || true
        done
        n=$(grep -aEo '[0-9]+/[0-9]+ tests passed' "$OUT/ops-$t-$dev-new.txt" | awk -F'[ /]' '{ s += $1 } END { print s + 0 }')
        (( n >= 1 )) || { echo "no passing rows on $dev"; return 8; }
        if [[ -n $(comm -23 "$OUT/ops-$t-$dev-new.fail" "$OUT/ops-$t-$dev-base.fail") ]]; then
            echo "failing on $t $dev only (not on frozen/stack11-base):"
            comm -23 "$OUT/ops-$t-$dev-new.fail" "$OUT/ops-$t-$dev-base.fail"
            return 8
        fi
        grep -aEq 'GET_ROWS\(type=iq4_xs,n=[0-9]+,.*\): .*OK' "$OUT/ops-$t-$dev-new.txt" || { echo 'no GET_ROWS iq4_xs row'; return 8; }
    done
}
OPS_COMMON=GET_ROWS,TOP_K,RMS_NORM,SCALE,RMS_NORM_MUL_ADD,RMS_NORM_MUL_ROPE,ADD_RMS_NORM
ops_k() { ops_tree k "$OPS_COMMON,RMS_NORM_SCALE_PAIR"; }
ops_nok() { ops_tree nok "$OPS_COMMON"; }
g_ops_k() { in_dir ops-k ops_k; }
g_ops_nok() { in_dir ops-nok ops_nok; }

# served-flag health boots: each profile's exact serve.sh argv/env on this build's round-trace
g_health_k() {
    local p
    mkdir -p "$OUT/health"
    [[ -n ${PORT:-} ]]
    for p in swift15-j5q swift15-j5q-duo swift15-iq4 swift15-iq4-duo; do
        env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$KROOT" PORT="$PORT" XID_SINCE="$STARTED" \
            python3 "$WIN/health.py" "$p" "$OUT/health/$p"
    done
}
# the nok build: one boot of the drop item's profile, K7 and K10 lines not expected
g_health_nok() {
    mkdir -p "$OUT/health-nok"
    [[ -n ${PORT:-} ]]
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$ROOT" PORT="$PORT" XID_SINCE="$STARTED" ENGAGEMENT_DROP='k7 k10' \
        python3 "$WIN/health.py" swift15-j5q-duo "$OUT/health-nok/swift15-j5q-duo"
}

freeze_tree() {
    local t=$1 d name desc
    d=$(tree_of "$t")
    name=$(frozen_of "$t")
    [[ $(git -C "$d" rev-parse HEAD) == $([[ $t == k ]] && echo "$KSHA" || echo "$SHA") && -z $(git -C "$d" status --porcelain --untracked-files=no) ]]
    [[ ! -e $W/frozen/$name ]]
    if [[ $t == k ]]; then
        desc="stack11-s2: stack11-base + 87 draft-state reset, 88 spec prompt on demand, 91 Meta per-uid cache tests, 90 CUDA_Host_Mapped token table; K7 and K10 kept; unit tests, host-mapped bitwise, GET_ROWS ops, np2 smoke and served-flag health boots passed on both GPUs; CUDA 13.2 120a CCCL343 NCCL OFF"
    else
        desc="stack11-s2-nok: stack11-s2 with K7 (RMS norm scale pair) and K10 (batched top-k) removed from the source; unit tests, host-mapped bitwise, ops and one health boot passed on both GPUs; CUDA 13.2 120a CCCL343 NCCL OFF"
    fi
    nice -n 19 "$W/bin/wt" freeze "$d" "$name" --build build-cuda --desc "$desc"
    [[ $(head -n 1 "$W/frozen/$name/SOURCE") == $(git -C "$d" rev-parse HEAD) ]]
    {
        printf 'commit %s\nconfigure: cmake -S . -B build-cuda ' "$(git -C "$d" rev-parse HEAD)"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: TMPDIR=/mnt/scratch/tmp-build bin/buildgate cmake --build build-cuda -j4 --target %s\n' "$(targets)"
        printf 'round-trace = copy of llama-server; gates ran in %s/codex/window/%s\n' "$ROOT" "${OUT##*/}"
    } > "$W/frozen/$name/RECIPE.txt"
}
freeze_k() { freeze_tree k; }
freeze_nok() { freeze_tree nok; }

items() {
    nice -n 19 python3 "$WIN/render_items.py" --out "$WIN/rendered" --components "$COMPONENTS" \
        --evidence "$ROOT/codex/stack11-s2/mac-evidence/with-k.tsv"
}

# GET_ROWS through Meta(CUDA0,CUDA1) on the real devices, the mapped weight against the device-resident one (meta_rows.cpp: bitwise, one
# gather split on Meta, no weight copy; each GPU first once; --nopin control falls back to the CPU path). Built against each tree.
meta_rows_tree() {
    local t=$1 d bin lib rc=0 n
    d=$(tree_of "$t")
    lib=$d/build-cuda/bin
    bin=$OUT/meta_rows-$t
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -I"$d/ggml/include" "$WIN/meta_rows.cpp" -L "$lib" -lggml -lggml-base -o "$bin"
    env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" timeout 900 nice -n 19 "$bin" > "$OUT/meta-rows-$t.txt" 2> "$OUT/meta-rows-$t.err" || rc=$?
    gate_log "$OUT/meta-rows-$t.txt" "$rc" '^meta_rows: PASS \(2 orders\)$' \
        '^meta_rows: order=CUDA0,CUDA1 types=[0-9]+ checks=[0-9]+ bitwise OK, placement OK$' \
        '^meta_rows: order=CUDA1,CUDA0 types=[0-9]+ checks=[0-9]+ bitwise OK, placement OK$' || return 8
    if grep -aq 'FAIL' "$OUT/meta-rows-$t.txt"; then return 8; fi
    n=$(grep -ac 'b_on_meta=1 splits=1 weight_copied=0 differing=0/' "$OUT/meta-rows-$t.txt" || true)
    [[ $n -ge 200 ]] || { echo "only $n mapped-weight gathers on Meta"; return 8; }
    rc=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" GGML_CUDA_NO_PINNED=1 timeout 900 nice -n 19 "$bin" --nopin > "$OUT/meta-rows-$t-nopin.txt" 2> "$OUT/meta-rows-$t-nopin.err" || rc=$?
    gate_log "$OUT/meta-rows-$t-nopin.txt" "$rc" '^meta_rows: PASS nopin \(2 orders\)$' || return 8
    n=$(grep -ac 'b_on_meta=0 .*differing=0/' "$OUT/meta-rows-$t-nopin.txt" || true)
    [[ $n -ge 200 ]] || { echo "nopin control: only $n CPU-path gathers"; return 8; }
}
meta_rows_k() { meta_rows_tree k; }
meta_rows_nok() { meta_rows_tree nok; }
g_meta_rows() { in_dir meta-rows meta_rows_k meta_rows_nok; }

# Placement proof on the frozen builds (placement.py): the four served shapes, base = frozen/stack11-base as the negative control.
g_placement() {
    local p b root
    mkdir -p "$OUT/placement"
    [[ -n ${PORT:-} ]]
    for p in swift15-j5q swift15-j5q-duo swift15-iq4 swift15-iq4-duo; do
        for b in base s2 nok; do
            case $b in
                base) root=$W/frozen/stack11-base ;;
                s2) root=$W/frozen/stack11-s2 ;;
                nok) root=$W/frozen/stack11-s2-nok ;;
            esac
            env CUDA_VISIBLE_DEVICES="$VISIBLE" PORT="$PORT" XID_SINCE="$STARTED" python3 "$WIN/placement.py" run "$p" "$b" "$root" "$OUT/placement"
        done
    done
    python3 "$WIN/placement.py" judge "$OUT/placement"
}
