#!/usr/bin/env bash
# Stage library of bw-stack11-base.sh (sourced, never run alone). The component gate bodies are the stack10 component windows'
# own gates (the codex/stack10/gates_table.py extractions in wt/stack10b) copied verbatim but for paths and the stack11 test names;
# a gate runs in a subshell under set -e with OUT=$OUT/<dir>, pinned to $CPUS. Every kept negative control must fail on purpose:
# a control that passes fails its gate (rc 8), a control that cannot be shown to bite is rc 5.
# Needs from the caller: ROOT W EVAL OUT SHA SHORT VISIBLE STARTED COMPONENTS CPUS CCCL BUILDGATE flags S10 S10_LIB S10_SHA S10_REPO
# CAND_LIB MASTER_LIB WIN.

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
    echo "bw-stack11: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || { echo "bw-stack11: $(fail_tag "$rc") $name rc=$rc"; finish_fail "$rc"; }
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
    echo "bw-stack11: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    if [[ $rc != 0 ]]; then
        FAILS+=("$name")
        STAGE_RCS+=("$rc")
        echo "bw-stack11: $(fail_tag "$rc") $name rc=$rc"
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
    echo "bw-stack11: $(fail_tag "$worst") gates=[${FAILS[*]:-}] rc=$worst; nothing frozen, nothing rendered"
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

# Test binaries of the stack (all exist in this tree; preflight checks the targets).
targets() { echo llama-server test-backend-ops test-alloc test-recurrent-state-rollback test-meta-input-stage test-mmvq-params test-topk-batched; }
build() {
    mkdir -p /mnt/scratch/tmp-build
    TMPDIR=/mnt/scratch/tmp-build nice -n 19 "$BUILDGATE" cmake -S . -B build-cuda "${flags[@]}"
    # shellcheck disable=SC2046
    TMPDIR=/mnt/scratch/tmp-build nice -n 19 "$BUILDGATE" cmake --build build-cuda -j4 --target $(targets)
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}

stamps() {
    local n lib
    for lib in libggml-cuda.so libggml-base.so libllama.so libllama-common.so; do [[ -s build-cuda/bin/$lib ]]; done
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK) || true
    echo "DeviceTopK_count_M=$n"
    [[ $n -gt 0 ]]
    [[ $(strings build-cuda/bin/libggml-base.so | grep -cx "$SHORT") -ge 1 ]]
    [[ $(grep -ac 'backend busy at a skipped sync' build-cuda/bin/libllama.so) -ge 1 ]]
    [[ $(grep -ac 'gdn_joint_mutant: engaged' build-cuda/bin/libggml-cuda.so) -ge 1 ]]
    [[ $(grep -ac 'GGML_CUDA_MMQ_SK_MUTANT' build-cuda/bin/libggml-cuda.so) -ge 1 ]]
    [[ $(grep -ac 'fattn_kv_max' build-cuda/bin/libggml-cuda.so) == 0 ]]
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    (cd build-cuda/bin && sha256sum round-trace llama-server test-backend-ops ./*.so) > "$OUT/build-sha256.txt"
}

# The stack10 crash: both slots prefilled at 8k, joint replay, 3 repetitions must survive; the pre-fix mutant must die.
np2_smoke() {
    [[ -n ${PORT:-} ]]
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$ROOT" PORT="$PORT" XID_SINCE="$STARTED" bash "$WIN/np2_smoke.sh" "$OUT/np2-smoke"
}

# --- base: alpha gate fusion (bw-stack10-cand.sh ops) -----------------------------------------------------------------------------
base_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev
    for dev in CUDA0 CUDA1; do
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o ADD_SOFTPLUS_MUL -j 1 > "$OUT/alpha-$dev.log" 2>&1
        grep -q 'same-backend byte identity and changing-input replay PASS' "$OUT/alpha-$dev.log"
        grep -q 'alpha_gate_fusion:' "$OUT/alpha-$dev.log"
        if env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_ALPHA_GATE_MUTANT=1 timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o ADD_SOFTPLUS_MUL -j 1 > "$OUT/alpha-mutant-$dev.log" 2>&1; then
            echo 'FAIL alpha mutant escaped'; return 1
        fi
        grep -q 'FAIL ADD_SOFTPLUS_MUL byte identity' "$OUT/alpha-mutant-$dev.log"
        if env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_ALPHA_ALIAS_MUTANT=1 timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o ADD_SOFTPLUS_MUL -j 1 > "$OUT/alpha-alias-mutant-$dev.log" 2>&1; then
            echo 'FAIL alpha alias mutant escaped'; return 1
        fi
        grep -E 'FAIL ADD_SOFTPLUS_MUL byte identity:.*layout=[23]' "$OUT/alpha-alias-mutant-$dev.log" >/dev/null
    done
}
g_base() { in_dir base base_ops; }

# --- FA: stack11-fa PORT.md T1-T7, grouped q8_0 decode engagement and its off control --------------------------------------------
FA_T=(
    'T1 hsk=256,hsv=256,nh=4,nr23=\[6,[12]\],kv=[0-9]+,nb=[1-8],.*type_K=q8_0'
    'T2 hsk=256,hsv=256,nh=2,nr23=\[8,1\],kv=1024,nb=5,'
    'T3 kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0'
    'T4 hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=(1024|16384),nb=1,.*type_K=q8_0'
    'T5 hsk=576,hsv=512,nh=1,nr23=\[8,1\],kv=(113|1024),nb=1,.*type_K=q8_0'
    'T6 type_K=q8_0'
)
Q8_DECODE_RE='hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0'
fa_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev row tag
    for dev in CUDA0 CUDA1; do
        for row in "${FA_T[@]}"; do
            tag=${row%% *}
            tb "$OUT/$tag-$dev.log" "$dev" 1800 -- -o FLASH_ATTN_EXT -p "${row#* }"
            all_passed "$OUT/$tag-$dev.log"
        done
        # the f16-copy arm of the in-kernel q8_0 verify (GGML_CUDA_FA_MMA_Q8_0=0) keeps passing the same shapes; no log line, recorded
        tb "$OUT/T1-mma-q8-off-$dev.log" "$dev" 1800 GGML_CUDA_FA_MMA_Q8_0=0 -- -o FLASH_ATTN_EXT -p "${FA_T[0]#* }"
        all_passed "$OUT/T1-mma-q8-off-$dev.log"
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p "$Q8_DECODE_RE" -j 1 > "$OUT/q8-$dev.log" 2>&1
        grep -q 'fattn_q8_decode:' "$OUT/q8-$dev.log"
        grep -E '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/q8-$dev.log" >/dev/null
        env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_Q8_DECODE=0 timeout 600 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p "$Q8_DECODE_RE" -j 1 > "$OUT/q8-control-$dev.log" 2>&1
        if grep -q 'fattn_q8_decode:' "$OUT/q8-control-$dev.log"; then
            echo 'FAIL grouped q8 off control engaged'; return 1
        fi
        # T7: the whole op; a failing case is excused only by frozen/stack10 failing it 3 of 3
        tbo_sql build-cuda/bin/test-backend-ops 5400s "$OUT/T7-$dev.sql" "$OUT/T7-$dev.err" -b "$dev" -o FLASH_ATTN_EXT
        python3 "$WIN/full_checks.py" ops-check "$OUT/T7-$dev.sql" "$dev" "$SHA" in FLASH_ATTN_EXT
        excuse_retry "$OUT/T7-$dev.sql" "$dev"
    done
}
g_fa() { in_dir fa fa_ops; }

# --- K7: RMS_NORM_SCALE_PAIR (codex/k7/bw-k7.sh ops) ------------------------------------------------------------------------------
k7_run_pair() {
    local name=$1 dev=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o RMS_NORM_SCALE_PAIR -j 1 > "$OUT/$name-$dev.log" 2>&1
}
k7_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev mask msg op
    for dev in CUDA0 CUDA1; do
        k7_run_pair pair "$dev"
        grep -c 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$OUT/pair-$dev.log" > /dev/null
        grep -c 'rms_norm_scale_pair_fusion:' "$OUT/pair-$dev.log" > /dev/null
        k7_run_pair pair-eager "$dev" GGML_CUDA_DISABLE_GRAPHS=1
        grep -c 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$OUT/pair-eager-$dev.log" > /dev/null
        grep -c 'rms_norm_scale_pair_fusion:' "$OUT/pair-eager-$dev.log" > /dev/null
        k7_run_pair pair-off "$dev" GGML_CUDA_RMS_NORM_SCALE_PAIR=0
        grep -c 'RMS_NORM_SCALE_PAIR: same-backend byte identity, matcher guards and changing-input replay PASS' "$OUT/pair-off-$dev.log" > /dev/null
        if grep -c 'rms_norm_scale_pair_fusion:' "$OUT/pair-off-$dev.log" > /dev/null; then
            echo 'FAIL pair off switch engaged'; return 1
        fi
        for mask in 1 2; do
            if k7_run_pair "mutant-$mask" "$dev" GGML_CUDA_RMS_NORM_SCALE_PAIR_MUTANT=$mask; then
                echo "FAIL pair mutant $mask escaped"; return 1
            fi
            grep -c 'FAIL RMS_NORM_SCALE_PAIR byte identity' "$OUT/mutant-$mask-$dev.log" > /dev/null
            grep -c 'rms_norm_scale_pair_fusion:' "$OUT/mutant-$mask-$dev.log" > /dev/null
        done
        for mask in 1 2 4 8; do
            case $mask in
                1) msg='first output over the second input' ;;
                2) msg='second output over the first input' ;;
                4) msg='outputs overlap' ;;
                8) msg='second input is a view of the first output' ;;
            esac
            if k7_run_pair "alias-mutant-$mask" "$dev" GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT=$mask; then
                echo "FAIL alias mutant $mask escaped"; return 1
            fi
            grep -c "FAIL RMS_NORM_SCALE_PAIR matcher accepted: $msg" "$OUT/alias-mutant-$mask-$dev.log" > /dev/null
        done
        for op in RMS_NORM RMS_NORM_MUL_ADD RMS_NORM_MUL_ROPE SCALE; do
            env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o "$op" -j 1 > "$OUT/neighbour-$op-$dev.log" 2>&1
            grep -E '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/neighbour-$op-$dev.log" > /dev/null
        done
    done
}
g_k7() { in_dir k7 k7_ops; }

# --- K4: SSM_CONV_RS (codex/k4/bw-k4.sh ops + model) ------------------------------------------------------------------------------
K4_MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
k4_tbo() {
    local dev=$1 log=$2
    shift 2
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o SSM_CONV_RS -j 1 > "$log" 2>&1 || RC=$?
}
k4_arm() {
    local dev=$1 name=$2 expect=$3 ref=$4
    shift 4
    local cmp=(--getter-logits "$OUT/model/$dev-$name.bin")
    [[ -z $ref ]] || cmp=(--getter-reference "$OUT/model/$dev-$ref.bin")
    RC=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-recurrent-state-rollback --getter-gate "${cmp[@]}" \
        --conv-expect "$expect" -m "$K4_MODEL" -c 4096 -t 4 -ngl 99 -dev "$dev" -lv 5 > "$OUT/model/$dev-$name.log" 2>&1 || RC=$?
}
k4_model() {
    mkdir -p "$OUT/model"
    local dev m
    for dev in CUDA0 CUDA1; do
        k4_arm "$dev" gathered gathered '' GGML_SSM_CONV_RS_OFF=1
        [[ $RC == 0 ]]
        k4_arm "$dev" rs rs gathered GGML_SSM_CONV_RS_LOG=1
        [[ $RC == 0 ]]
        grep -aq 'ssm_conv_rs engaged' "$OUT/model/$dev-rs.log"
        for m in 1 2; do
            k4_arm "$dev" mutant$m rs gathered GGML_SSM_CONV_RS_MUTANT=$m
            if [[ $RC == 0 ]]; then echo "FAIL model mutant $m escaped on $dev"; return 1; fi
            grep -aq 'logits differ' "$OUT/model/$dev-mutant$m.log"
        done
    done
}
k4_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev m
    for dev in CUDA0 CUDA1; do
        k4_tbo "$dev" "$OUT/ops-$dev.log" GGML_SSM_CONV_RS_LOG=1
        [[ $RC == 0 ]]
        grep -aq 'byte identity with the gathered path and changing-id replay PASS cases=[1-9][0-9]* unsupported=0\b' "$OUT/ops-$dev.log"
        grep -aq 'ssm_conv_rs engaged:.* silu=1' "$OUT/ops-$dev.log"
        grep -aq 'ssm_conv_rs engaged:.* silu=0' "$OUT/ops-$dev.log"
        grep -aEq '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/ops-$dev.log"
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o SSM_CONV -j 1 > "$OUT/ssm-conv-$dev.log" 2>&1
        for m in 1 2; do
            k4_tbo "$dev" "$OUT/ops-mutant$m-$dev.log" GGML_SSM_CONV_RS_MUTANT=$m
            if [[ $RC == 0 ]]; then echo "FAIL ssm_conv_rs mutant $m escaped on $dev"; return 1; fi
            grep -aq 'FAIL SSM_CONV_RS byte identity' "$OUT/ops-mutant$m-$dev.log"
        done
        k4_tbo "$dev" "$OUT/ops-off-$dev.log" GGML_SSM_CONV_RS_OFF=1 GGML_SSM_CONV_RS_LOG=1
        [[ $RC == 0 ]]
        grep -aq 'PASS cases=0 ' "$OUT/ops-off-$dev.log"
        if grep -aq 'ssm_conv_rs engaged' "$OUT/ops-off-$dev.log"; then echo "FAIL off control engaged on $dev"; return 1; fi
    done
}
g_k4() { in_dir k4 k4_ops k4_model; }

# --- meta subgraph fix (pkg 86): test-alloc on CPU devices; the pre-fix mutant is recorded only (no ASAN here: its read of the freed
# arena need not fault; the must-fail controls are the Mac ASAN stage and the np2 smoke's mutant arm) ------------------------------
mfix_run() {
    local rc=0
    timeout --kill-after=10s 300 env -u GGML_META_SUBGRAPH_MUTANT CUDA_VISIBLE_DEVICES='' build-cuda/bin/test-alloc > "$OUT/test-alloc.txt" 2>&1
    grep -aq 'test_meta_subgraph_realloc_count PASSED' "$OUT/test-alloc.txt"
    grep -aq 'test_meta_subgraph_realloc_size PASSED' "$OUT/test-alloc.txt"
    timeout --kill-after=10s 300 env GGML_META_SUBGRAPH_MUTANT=1 CUDA_VISIBLE_DEVICES='' build-cuda/bin/test-alloc > "$OUT/test-alloc-mutant.txt" 2>&1 || rc=$?
    echo "recorded: pre-fix mutant test-alloc rc_M=$rc (nonzero expected)"
}
g_mfix() { in_dir mfix mfix_run; }

# --- K6a / K6b (codex/k6/bw-s10-k6.sh ops, split as in stack10) ------------------------------------------------------------------
k6_run() {
    local log=$1
    shift
    local envs=()
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ${envs[@]+"${envs[@]}"} timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test "$@" -j 1 > "$log" 2>&1
}
k6_count() { grep -ac -- "$1" "$2" || true; }
k6a_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev n
    for dev in CUDA0 CUDA1; do
        k6_run "$OUT/shared-$dev.log" -- -b "$dev" -o MUL_MAT_SHARED
        grep -aq 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS' "$OUT/shared-$dev.log"
        grep -aq 'q8_reuse:.* rows=12 skipped=[1-9]' "$OUT/shared-$dev.log"
        grep -aq 'q8_reuse:.* rows=16 skipped=[1-9]' "$OUT/shared-$dev.log"
        n=$(k6_count 'q8_reuse:.* rows=17 ' "$OUT/shared-$dev.log"); [[ $n == 0 ]] || { echo "FAIL 17-row input engaged reuse ($n)"; return 1; }
        k6_run "$OUT/shared-off-$dev.log" GGML_CUDA_Q8_REUSE=0 -- -b "$dev" -o MUL_MAT_SHARED
        grep -aq 'MUL_MAT_SHARED: same-backend byte identity and changing-input replay PASS' "$OUT/shared-off-$dev.log"
        n=$(k6_count 'q8_reuse:' "$OUT/shared-off-$dev.log"); [[ $n == 0 ]] || { echo "FAIL reuse off still engaged ($n)"; return 1; }
        if k6_run "$OUT/shared-stale-$dev.log" GGML_CUDA_Q8_REUSE_STALE_MUTANT=1 -- -b "$dev" -o MUL_MAT_SHARED; then
            echo 'FAIL stale-cache mutant escaped'; return 1
        fi
        grep -aq 'FAIL MUL_MAT_SHARED byte identity' "$OUT/shared-stale-$dev.log"
    done
}
k6b_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev n
    for dev in CUDA0 CUDA1; do
        k6_run "$OUT/addrms-$dev.log" -- -b "$dev" -o RMS_NORM_MUL_ADD,ADD_RMS_NORM
        grep -aq 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS' "$OUT/addrms-$dev.log"
        grep -aq 'add_rms_norm: .* rows=12 ' "$OUT/addrms-$dev.log"
        grep -aq 'add_rms_norm: .* rows=16 ' "$OUT/addrms-$dev.log"
        n=$(k6_count 'add_rms_norm: .* rows=\(17\|33\) ' "$OUT/addrms-$dev.log"); [[ $n == 0 ]] || { echo "FAIL rows above the cap fused ($n)"; return 1; }
        k6_run "$OUT/addrms-off-$dev.log" GGML_CUDA_ADD_RMS_NORM=0 -- -b "$dev" -o RMS_NORM_MUL_ADD,ADD_RMS_NORM
        grep -aq 'ADD_RMS_NORM: same-backend byte identity and changing-input replay PASS' "$OUT/addrms-off-$dev.log"
        n=$(k6_count 'add_rms_norm:' "$OUT/addrms-off-$dev.log"); [[ $n == 0 ]] || { echo "FAIL residual fusion off still engaged ($n)"; return 1; }
        if k6_run "$OUT/addrms-mutant-$dev.log" GGML_CUDA_ADD_RMS_NORM_MUTANT=1 -- -b "$dev" -o RMS_NORM_MUL_ADD,ADD_RMS_NORM; then
            echo 'FAIL residual numerical mutant escaped'; return 1
        fi
        grep -aq 'FAIL ADD_RMS_NORM byte identity' "$OUT/addrms-mutant-$dev.log"
    done
}
k6_memcheck() {
    [[ $VISIBLE == GPU-* ]] || return 1
    # CUDA graphs off: a refused cudaGraphExecUpdate (error 910) is handled by re-instantiation (ggml-cuda.cu) but memcheck
    # counts every one as an error; the kernels under test run the same without graphs.
    env CUDA_VISIBLE_DEVICES="${VISIBLE%%,*}" GGML_CUDA_DISABLE_GRAPHS=1 timeout 1800 nice -n 19 /usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 9 \
        build-cuda/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_SHARED,RMS_NORM_MUL_ADD,ADD_RMS_NORM -j 1 > "$OUT/memcheck.log" 2>&1
    grep -aq 'ERROR SUMMARY: 0 errors' "$OUT/memcheck.log"
}
g_k6a() { in_dir k6a k6a_ops; }
g_k6b() { in_dir k6b k6b_ops; }
g_k6_memcheck() { in_dir k6 k6_memcheck; }

# --- gdn4 + joint multi-sequence GDN (wt/np2-kernels codex/np2/cuda/cuda-checks.sh, both GPUs) ------------------------------------
GDN_ENGAGED=('seqs=2 indexed=1 gdn=1 fused=1\b' 'seqs=4 indexed=1 gdn=1 fused=1\b' 'seqs=5 indexed=1 gdn=1 fused=0\b')
gdn_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev re i caught
    for dev in CUDA0 CUDA1; do
        tb "$OUT/indexed-fusion-$dev.log" "$dev" 900 GGML_CUDA_DEBUG_GDN_STATE=1 -- -o GATED_DELTA_NET_INDEXED_CACHE_FUSION
        all_passed "$OUT/indexed-fusion-$dev.log"
        for re in "${GDN_ENGAGED[@]}"; do
            grep -aEq "gdn_cache_fusion: graph device=[01] graph=[0-9]+ tokens=[0-9]+ $re" "$OUT/indexed-fusion-$dev.log" || { echo "FAIL no /$re/ on $dev"; return 1; }
        done
        for op in GATED_DELTA_NET_CACHE_FUSION GATED_DELTA_NET GATED_DELTA_NET_INDEXED; do
            tb "$OUT/$op-$dev.log" "$dev" 900 -- -o "$op"
            all_passed "$OUT/$op-$dev.log"
        done
        # the same build with each sequence in its own blocks must race on the swapped ids
        caught=0
        for i in 1 2 3 4 5; do
            tb "$OUT/mutant-$i-$dev.log" "$dev" 900 GGML_CUDA_DEBUG_GDN_STATE=1 GGML_CUDA_GDN_JOINT_MUTANT=1 -- -o GATED_DELTA_NET_INDEXED_CACHE_FUSION -p 'head_count=1024,'
            grep -aq 'gdn_joint_mutant: engaged' "$OUT/mutant-$i-$dev.log" || { echo "mutant not engaged on $dev"; exit 6; }
            if ! all_passed "$OUT/mutant-$i-$dev.log" > /dev/null; then caught=$i; break; fi
        done
        (( caught )) || { echo "void control: joint-kernel mutant passed 5 times on $dev"; exit 5; }
        echo "$dev: joint mutant caught on try $caught"
    done
}
g_gdn4() { in_dir gdn4 gdn_ops; }

# --- K10: batched top-k (codex/k10/bw-k10.sh ops + controls) and the CPU emulator unit test -------------------------------------
k10_tbo() {
    local dev=$1 out=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o TOP_K -j 1 > "$out" 2>&1
}
k10_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev
    for dev in CUDA0 CUDA1; do
        k10_tbo "$dev" "$OUT/topk-$dev.log"
        grep -aE '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/topk-$dev.log" > /dev/null
        python3 "$WIN/k10_perf.py" engaged "$OUT/topk-$dev.log"
        k10_tbo "$dev" "$OUT/topk-log-$dev.log" GGML_CUDA_TOPK_LOG=1
        [[ $(grep -ac 'top_k_batched: engaged' "$OUT/topk-log-$dev.log") -gt 0 ]]
    done
}
k10_controls() {
    local dev arm rc envs
    for dev in CUDA0 CUDA1; do
        for arm in off m1 m2 m3 m4 m5 m6; do
            case $arm in
                off) envs=(GGML_CUDA_TOPK_BATCHED=0) ;;
                m*) envs=(GGML_CUDA_TOPK_MUTANT="${arm#m}") ;;
            esac
            rc=0
            k10_tbo "$dev" "$OUT/control-$arm-$dev.log" "${envs[@]}" || rc=$?
            [[ $rc != 0 ]] || { echo "FAIL control $arm escaped on $dev"; return 1; }
            python3 "$WIN/k10_perf.py" failures "$OUT/control-$arm-$dev.log" "$arm"
        done
    done
}
k10_unit() {
    local rc=0
    env CUDA_VISIBLE_DEVICES='' timeout 900 nice -n 19 build-cuda/bin/test-topk-batched > "$OUT/unit.log" 2>&1
    grep -aEq '^PASS: [1-9][0-9]* cases, 0 failed' "$OUT/unit.log"
    env CUDA_VISIBLE_DEVICES='' TOPK_MUTANT=1 timeout 900 nice -n 19 build-cuda/bin/test-topk-batched > "$OUT/unit-mutant1.log" 2>&1 || rc=$?
    [[ $rc == 1 ]] && grep -aEq '^FAIL: [1-9][0-9]* cases, [1-9][0-9]* failed' "$OUT/unit-mutant1.log" || { echo "FAIL emulator tie-order mutant escaped rc=$rc"; return 1; }
}
g_k10() { in_dir k10 k10_ops k10_controls k10_unit; }

# --- p55: MMVQ launch parameters (stack11-mmvq PORT.md "GPU window"); no switch exists, the CPU table controls are in PORT.md -----
mmvq_ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev q n
    env CUDA_VISIBLE_DEVICES='' timeout 600 nice -n 19 build-cuda/bin/test-mmvq-params > "$OUT/mmvq-params.log" 2>&1
    grep -aEq '^M: [1-9][0-9]* configurations passed' "$OUT/mmvq-params.log"
    for dev in CUDA0 CUDA1; do
        tb "$OUT/mmvq-$dev.log" "$dev" 1800 -- -o MUL_MAT -p '^type_a=(q8_0|q4_K|q5_K|q6_K|iq4_xs),type_b=f32,m=[0-9]+,n=[1-8],k='
        all_passed "$OUT/mmvq-$dev.log" 60  # D: the m 5-7 x n 5-8 loop alone is 5 types x 12 cases (stack11-mmvq PORT.md)
        for q in q8_0 q4_K q5_K q6_K iq4_xs; do
            for n in 5 6 7 8; do
                grep -aEq "MUL_MAT\(type_a=$q,type_b=f32,m=[0-9]+,n=$n,[^)]*\).*OK" "$OUT/mmvq-$dev.log" || { echo "FAIL no passing $q n=$n case on $dev"; return 1; }
            done
        done
    done
}
g_mmvq() { in_dir mmvq mmvq_ops; }

# --- K8: its own k8-gpu.sh (bitwise reference = the same build's GGML_CUDA_MMQ_TILES=default; auto-drop and the stream-k mutant must
# fail) -----------------------------------------------------------------------------------------------------------------------------
g_k8() {
    mkdir -p "$OUT/k8"
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$ROOT" COMMIT="$SHA" OUT="$OUT/k8/checks-canary" BUDGET_S=600 bash "$WIN/k8-gpu.sh" checks --canary
    env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$ROOT" COMMIT="$SHA" OUT="$OUT/k8/checks" BUDGET_S=9000 bash "$WIN/k8-gpu.sh" checks --run
}

# --- K11b: input staging (codex/k11b/bw-s10-k11b.sh ops) --------------------------------------------------------------------------
K11B_T=build-cuda/bin/test-meta-input-stage
k11b_run_test() {  # LOG VISIBLE [ENV=VAL...]
    local log=$1 vis=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$vis" INPUT_STAGE_REQUIRE_CUDA=1 "$@" timeout 900 nice -n 19 "$K11B_T" > "$log" 2>&1
}
k11b_ops() {
    [[ $VISIBLE == GPU-*,GPU-* ]] || return 1
    local rev=${VISIBLE#*,},${VISIBLE%%,*}
    k11b_run_test "$OUT/stage-fwd-vis.log" "$VISIBLE"
    grep -aq 'cuda input stage: OK' "$OUT/stage-fwd-vis.log"
    grep -aq 'test-meta-input-stage: PASS' "$OUT/stage-fwd-vis.log"
    grep -aq 'slowest host link first: CUDA1 ' "$OUT/stage-fwd-vis.log"
    k11b_run_test "$OUT/stage-rev-vis.log" "$rev"
    grep -aq 'cuda input stage: OK' "$OUT/stage-rev-vis.log"
    grep -aq 'slowest host link first: CUDA0 ' "$OUT/stage-rev-vis.log"
    k11b_run_test "$OUT/stage-order-fwd.log" "$VISIBLE" GGML_CUDA_INPUT_STAGE_ORDER=fwd
    grep -aq 'slowest host link first: CUDA0 ' "$OUT/stage-order-fwd.log"
    k11b_run_test "$OUT/stage-order-rev.log" "$VISIBLE" GGML_CUDA_INPUT_STAGE_ORDER=rev
    grep -aq 'slowest host link first: CUDA1 ' "$OUT/stage-order-rev.log"
    if k11b_run_test "$OUT/stage-mutant-cuda.log" "$VISIBLE" GGML_CUDA_INPUT_STAGE_MUTANT=1; then
        echo 'FAIL CUDA staging mutant escaped'; return 1
    fi
    grep -aq 'tensor [AB] differs' "$OUT/stage-mutant-cuda.log"
    # The glue mutant makes fake_stage_set abort() (rc 134, buffered stdout lost); "fake/staged: ranks differ" (rc 1) is the fallback.
    local rc=0
    k11b_run_test "$OUT/stage-mutant-glue.log" "$VISIBLE" GGML_META_INPUT_STAGE_MUTANT=1 || rc=$?
    echo "glue mutant rc=$rc"
    if [[ $rc == 0 ]]; then echo 'FAIL meta glue mutant escaped'; return 1; fi
    if [[ $rc != 134 ]] && ! { [[ $rc == 1 ]] && grep -aq 'fake/staged: ranks differ' "$OUT/stage-mutant-glue.log"; }; then
        echo "FAIL meta glue mutant: rc=$rc is neither SIGABRT (134) nor the fake/staged content check"; return 1
    fi
    if grep -aq 'test-meta-input-stage: PASS' "$OUT/stage-mutant-glue.log"; then echo 'FAIL glue mutant log has a PASS line'; return 1; fi
}
g_k11b() { in_dir k11b k11b_ops; }

# --- K11a: sync-count capture (three server boots under nsys), last and only on a clean sheet ------------------------------------
g_k11a() {
    mkdir -p "$OUT/k11a"
    [[ -n ${PORT:-} ]]
    env CUDA_VISIBLE_DEVICES="$VISIBLE" DRY_RUN=0 ROOT="$ROOT" OUT="$OUT/k11a/sync" PORT="$PORT" XID_SINCE="$STARTED" BUDGET_S=3000 \
        bash "$WIN/k11a/sync_count.sh"
    [[ -s $OUT/k11a/sync/sync-count.json ]]
}

# --- fafix: SASS of the multi-query q8_0 verify FA kernels vs frozen/stack10, spill relative to the master-anchored f16 sibling;
# the regressed stack10-cand library and the spill mutants must fail ------------------------------------------------------------------
g_sass() {
    mkdir -p "$OUT/sass"
    python3 "$WIN/sass_gate.py" "$OUT/sass" "$S10_LIB/libggml-cuda.so" build-cuda/bin/libggml-cuda.so --master "$MASTER_LIB/libggml-cuda.so" \
        --control "$CAND_LIB/libggml-cuda.so"
}

# --- fafix: FA output digests of this build vs frozen/stack10, each driver built against the headers of the libraries it loads --------
fa_runid() { # DEVICE GROUP TAG EXE LIBDIR [ENV...]
    local dev=$1 group=$2 tag=$3 exe=$4 lib=$5
    shift 5
    env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" "$@" timeout 900 "$exe" "$dev" "$group" \
        > "$OUT/id-$dev-$group-$tag.txt" 2> "$OUT/id-$dev-$group-$tag.err"
}
fa_same() { cmp -s "$OUT/id-$1-$2-$3.txt" "$OUT/id-$1-$2-$4.txt"; }
s10_headers() { # DEST: frozen/stack10's public headers from its commit (wt/stack10b holds the objects)
    mkdir -p "$1"
    git -C "$S10_REPO" archive --format=tar "$S10_SHA" ggml/include include common | tar -x -C "$1"
}
fa_drivers() {
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -Iggml/include "$WIN/fa-identity.cpp" -L build-cuda/bin -lggml -lggml-base -o "$OUT/fa-identity"
    s10_headers "$OUT/s10-src"
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -I"$OUT/s10-src/ggml/include" "$WIN/fa-identity.cpp" -L "$S10_LIB" -lggml -lggml-base -o "$OUT/fa-identity-s10"
}
fafix_identity() {
    [[ $VISIBLE == GPU-* ]] || return 1
    fa_drivers
    local dev group f
    for dev in CUDA0 CUDA1; do
        for group in decode verify other; do
            fa_runid "$dev" "$group" new "$OUT/fa-identity" "$ROOT/build-cuda/bin"
            fa_runid "$dev" "$group" s10 "$OUT/fa-identity-s10" "$S10_LIB"
            f=$OUT/id-$dev-$group-new.txt
            [[ $(wc -l < "$f") -ge 10 && $(grep -c UNSUPPORTED "$f") == 0 && $(grep -c 'nonfinite=[1-9]' "$f") == 0 ]] || { echo "FAIL vacuous or non-finite $f"; return 1; }
            echo "$dev $group cases=$(wc -l < "$f") new==s10:$(fa_same "$dev" "$group" new s10 && echo yes || echo no)"
        done
        fa_same "$dev" decode new s10 || { echo "FAIL $dev decode bytes differ from frozen/stack10"; return 1; }
        fa_same "$dev" verify new s10 || { echo "FAIL $dev verify bytes differ from frozen/stack10"; return 1; }
        # master-owned paths (f16 KV, prefill shapes) may move with the new base: recorded, not judged
        echo "$dev other (informational) new==s10: $(fa_same "$dev" other new s10 && echo yes || echo no), differing cases $(diff "$OUT/id-$dev-other-new.txt" "$OUT/id-$dev-other-s10.txt" | grep -c '^>' || true)"
        [[ $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-new.err") -gt 0 && $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-s10.err") -gt 0 ]] || { echo "FAIL $dev decode not engaged in the driver"; return 1; }
        fa_runid "$dev" decode q8off "$OUT/fa-identity" "$ROOT/build-cuda/bin" GGML_CUDA_Q8_DECODE=0
        [[ $(grep -ac 'fattn_q8_decode:' "$OUT/id-$dev-decode-q8off.err") == 0 ]] || { echo "FAIL $dev switch-off control engaged"; return 1; }
        echo "$dev decode on-vs-off differing cases (informational): $(diff "$OUT/id-$dev-decode-new.txt" "$OUT/id-$dev-decode-q8off.txt" | grep -c '^>' || true)"
        fa_runid "$dev" verify f16copy "$OUT/fa-identity" "$ROOT/build-cuda/bin" GGML_CUDA_FA_MMA_Q8_0=0
        echo "$dev verify q8-in-kernel vs f16-copy differing cases (informational): $(diff "$OUT/id-$dev-verify-new.txt" "$OUT/id-$dev-verify-f16copy.txt" | grep -c '^>' || true)"
    done
    f=$OUT/id-CUDA0-decode-new.txt
    sed '1s/ [0-9a-f]\{16\} / 0000000000000000 /' "$f" > "$OUT/mutated.txt"
    if cmp -s "$f" "$OUT/mutated.txt"; then
        echo 'FAIL digest mutation not seen'; return 1
    fi
}
g_fafix() { in_dir fafix fafix_identity; }

# --- FA q8_0 timing vs frozen/stack10 at the served verify shapes (the SASS gate's spill bound is master-relative, so the cost of
# the higher absolute spill is measured): one GPU, arms alternated ABAB, FAPERF_REPS runs each; fa_q8_perf.py judges the medians
# per shape (stack11 <= stack10 x 1.03) and writes OUT/fa-q8-perf.tsv.
FAPERF_DEV=CUDA0
FAPERF_REPS=5
faperf_ab() {
    [[ $VISIBLE == GPU-* ]] || return 1
    fa_drivers
    local rep
    for rep in $(seq 1 "$FAPERF_REPS"); do
        fa_runid "$FAPERF_DEV" perf "new-r$rep" "$OUT/fa-identity" "$ROOT/build-cuda/bin"
        fa_runid "$FAPERF_DEV" perf "s10-r$rep" "$OUT/fa-identity-s10" "$S10_LIB"
    done
    python3 "$WIN/fa_q8_perf.py" judge "${OUT%/faperf}/fa-q8-perf.tsv" "$OUT"/id-"$FAPERF_DEV"-perf-*.txt
}
g_faperf() { in_dir faperf faperf_ab; }

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
    re=$(python3 "$WIN/full_checks.py" --regex "$params")
    tbo_sql "$bin" 300s "$tag.sql" "$tag.err" -b "$dev" -o "$op" -p "$re" || return 9
    python3 "$WIN/full_checks.py" --rerun "$tag.sql" "$op" "$dev" "$commit" "$params" || rc=$?
    return "$rc"
}

# A failing row is excused only by evidence: scope out (a case no component's launch config reaches) passes 3 fresh reruns of this
# build; or frozen/stack10's binary fails the same case 3 of 3 (pre-existing, not a regression). Scope in has only the second way.
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
        rerun_case "$S10_LIB/test-backend-ops" "$S10_SHA" "$op" "$dev" "$params" "$tag.base$a" || rc=$?
        [[ $rc == 2 ]] || { echo "NOT excused ($scope, frozen/stack10 rerun $a rc=$rc): $op $params"; return 1; }
    done
    echo "excused, fails 3 of 3 on frozen/stack10 too: $op $params"
}
excuse_retry() { # SQL DEV: every row of SQL.retry must be excused
    local scope op params
    while IFS=$'\t' read -r -u 3 scope op params; do
        [[ -n $params ]] || continue
        excused "$scope" "$op" "$2" "$params" "$1.re$((++NRE))"
    done 3< "$1.retry"
}

# --- every op the stack touches, on both GPUs: each must have passing rows and no unexcused failure (scope out) ------------------------
TOUCHED=GATED_DELTA_NET,GATED_DELTA_NET_INDEXED,GATED_DELTA_NET_CACHE_FUSION,GATED_DELTA_NET_INDEXED_CACHE_FUSION,SSM_CONV,SSM_CONV_RS,SSM_CONV_STATE_FUSION,SSM_CONV_BIAS_SILU,TOP_K,RMS_NORM,RMS_NORM_MUL_ADD,RMS_NORM_MUL_ROPE,RMS_NORM_SCALE_PAIR,ADD_RMS_NORM,SCALE,CONCAT,ADD_SOFTPLUS_MUL,MUL_MAT_SHARED,MUL_MAT_ID_SHARED,GET_ROWS,MUL_MAT_ID
touched_ops() {
    local dev
    for dev in CUDA0 CUDA1; do
        tbo_sql build-cuda/bin/test-backend-ops 5400s "$OUT/touched-$dev.sql" "$OUT/touched-$dev.err" -b "$dev" -o "$TOUCHED"
        # the K7 self-test is skipped after any failing case: its marker also comes from a run that selects no case (as in g_full)
        tbo_sql build-cuda/bin/test-backend-ops 1800s "$OUT/touched-$dev.self.sql" "$OUT/touched-$dev.self.err" -b "$dev" -o RMS_NORM_SCALE_PAIR -p no-such-case
        python3 "$WIN/full_checks.py" ops-check "$OUT/touched-$dev.sql" "$dev" "$SHA" out "$TOUCHED" "$OUT/touched-$dev.self.sql"
        excuse_retry "$OUT/touched-$dev.sql" "$dev"
    done
}
g_touched() { in_dir touched touched_ops; }

# The only gate that runs every case of the suite: byte-identity markers of the base and selected components, no FAIL line, per-op
# coverage, and flake handling after p55_checks.py. Covers p55's MMVQ coverage (changed quants, n=5..8).
g_full() {
    local dev base csv
    mkdir -p "$OUT/full"
    python3 "$WIN/full_checks.py" --fixtures
    csv=base,$(echo "$COMPONENTS" | tr ' ' ',')
    for dev in CUDA0 CUDA1; do
        base=$OUT/full/$dev
        tbo_sql build-cuda/bin/test-backend-ops 12600s "$base.sql" "$base.err" -b "$dev"
        # One failing case skips the K7 self-test (chained after `n_ok == tests_run` with &&), so its marker comes from a run that
        # selects no case.
        tbo_sql build-cuda/bin/test-backend-ops 1800s "$base.self.sql" "$base.self.err" -b "$dev" -o RMS_NORM_SCALE_PAIR -p no-such-case
        nice -n 19 python3 "$WIN/full_checks.py" check "$base.sql" "$dev" "$SHA" "$csv" "$base.self.sql"
        excuse_retry "$base.sql" "$dev"
    done
}

# --- served-flag health boots: each profile's exact serve.sh argv/env on this build's round-trace ---------------------------------
g_health() {
    local p
    mkdir -p "$OUT/health"
    [[ -n ${PORT:-} ]]
    for p in swift15-j5q swift15-j5q-duo swift15-iq4 swift15-iq4-duo; do
        env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$ROOT" PORT="$PORT" XID_SINCE="$STARTED" \
            python3 "$WIN/health.py" "$p" "$OUT/health/$p"
    done
}

freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    [[ ! -e $W/frozen/stack11-base ]]
    local desc="stack11-base: master abee0c847 + p55 K7 K4 (+ meta subgraph fix) K6a K6b K10 K8 K11a K11b gdn4 ($COMPONENTS), no K2/K9; CUDA gates, np2 smoke and served-flag health boots passed on both GPUs; CUDA 13.2 120a CCCL343 NCCL OFF"
    nice -n 19 "$W/bin/wt" freeze "$ROOT" stack11-base --build build-cuda --desc "$desc"
    [[ $(head -n 1 "$W/frozen/stack11-base/SOURCE") == "$SHA" ]]
    {
        printf 'commit %s\nconfigure: cmake -S . -B build-cuda ' "$SHA"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: TMPDIR=/mnt/scratch/tmp-build bin/buildgate cmake --build build-cuda -j4 --target %s\n' "$(targets)"
        printf 'round-trace = copy of llama-server; gates ran in codex/window/%s (components: %s)\n' "${OUT##*/}" "$COMPONENTS"
    } > "$W/frozen/stack11-base/RECIPE.txt"
}

# kld_dump against each pair member's own headers and libraries (ABI), for the paired KLD items.
kld_build() {
    nice -n 19 bash "$WIN/kld/build_dump.sh" "$WIN/kld/gen" "$W/frozen/stack11-base" "$ROOT" "$S10" "$S10_REPO"
}

items() {
    nice -n 19 python3 "$WIN/render_items.py" --out "$WIN/rendered" --components "$COMPONENTS"
}
