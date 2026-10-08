#!/usr/bin/env bash
# GDN four-columns-per-warp (#30087) on stack8, test-backend-ops only: no model, no server.
#   cuda-checks.sh --preflight | --canary | --run     env: OUT (required), COMMIT (frozen candidate SOURCE sha), BUDGET_S (default 2400)
# Binaries: cand = frozen/gdn-4col-s8, base = stack8 source + the same tests (work/gdn-4col-s8-variants/base),
#           mutant = cand with the attn-score reduction halved (work/gdn-4col-s8-variants/mutant), the negative control.
# rc: 0 pass | 3 Xid | 4 GPU missing or busy | 5 mutant not caught (control inconclusive) | 6 integrity (bytes, libs, symbols, GPU model)
#     7 budget | 8 a correctness case or perf run fails (on cand or base), or perf summary
set -euo pipefail
export GIT_NO_LAZY_FETCH=1
W=/home/xanewok/repos/qwen38-eval-worktrees
FZ=$W/frozen/gdn-4col-s8
VAR=$W/work/gdn-4col-s8-variants
HERE=$(cd "$(dirname "$0")" && pwd)
declare -A BIN=([cand]=$FZ/build/bin [base]=$VAR/base/bin [mutant]=$VAR/mutant/bin)
GPUS=(GPU-ee9a3956-297c-604b-b4c1-f07930d27e46 GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f)
MODE=${1:?usage: cuda-checks.sh --preflight|--canary|--run}
OUT=${OUT:?OUT required}
COMMIT=${COMMIT:?COMMIT required}
BUDGET_S=${BUDGET_S:-2400}
START=$SECONDS
CUTOFF=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
die() { local rc=$1; shift; echo "cuda-checks: FAIL rc=$rc $*" | tee -a "$SUMMARY" >&2; exit "$rc"; }
note() { echo "cuda-checks: $*" | tee -a "$SUMMARY"; }

KERNEL4='gated_delta_net_cudaILi128ELb[01]ELb[01]ELi4E'  # cols_per_warp = 4 instantiation in the mangled name
verify_arm() { # ARM
    local a=$1 d=${BIN[$1]}
    [[ -x $d/test-backend-ops ]] || die 6 "no test-backend-ops in $d"
    (cd "$d" && sha256sum -c --quiet SHA256SUMS) || die 6 "bytes changed in $d"
    [[ -z $(ldd "$d/test-backend-ops" | grep 'not found' || true) ]] || die 6 "unresolved libraries in $d"
    local stray
    stray=$(ldd "$d/test-backend-ops" | awk '/libggml/ {print $3}' | grep -v "^$d/" || true)
    [[ -z $stray ]] || die 6 "ggml libs of $a resolve outside $d: $stray"
    if [[ $a == base ]]; then
        if grep -aEq "$KERNEL4" "$d/libggml-cuda.so"; then die 6 "base carries the four-column kernel"; fi
        grep -aEq 'gated_delta_net_cudaILi128ELb[01]ELb[01]EEv' "$d/libggml-cuda.so" || die 6 "base lacks the stack8 GDN kernel"
    else
        grep -aEq "$KERNEL4" "$d/libggml-cuda.so" || die 6 "$a lacks the four-column kernel"
    fi
}
verify_all() {
    [[ $(head -1 "$FZ/SOURCE" | cut -d' ' -f1) == "$COMMIT" ]] || die 6 "frozen SOURCE is not $COMMIT"
    for a in cand base mutant; do verify_arm "$a"; done
}
guard() {
    if journalctl -k -b --since "$CUTOFF" --no-pager 2>/dev/null | grep -Ec '\bXid\b|fallen off the bus' > /dev/null; then die 3 "Xid since $CUTOFF"; fi
    (( SECONDS - START < BUDGET_S )) || die 7 "budget expired"
}
check_gpu() { # UUID
    local name mem
    name=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v u="$1" '$1 == u {print $2}')
    [[ -n $name ]] || die 4 "GPU $1 missing"
    [[ $name == *"RTX 5060 Ti"* ]] || die 6 "GPU $1 is not an RTX 5060 Ti: $name"
    for _ in $(seq 45); do
        mem=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u="$1" '$1 == u {print $2}')
        (( mem <= 512 )) && return 0
        sleep 1
    done
    die 4 "GPU $1 memory did not release"
}
RC=0
tbo() { # TAG ARM GPU_INDEX -- ARGS...
    local tag=$1 arm=$2 gpu=${GPUS[$3]}; shift 4
    guard
    RC=0
    env CUDA_VISIBLE_DEVICES="$gpu" timeout 600 nice -n 5 "${BIN[$arm]}/test-backend-ops" -b CUDA0 "$@" > "$OUT/$tag.log" 2>&1 || RC=$?
    (( RC != 124 )) || die 8 "$tag hung (600 s)"
    printf '%s\t%s\n' "$tag" "$RC" >> "$OUT/runs.tsv"
}
passed() { # TAG: rc 0, a non-empty "N/N tests passed" and no FAIL line
    [[ $RC == 0 ]] && grep -Eq '^ +([1-9][0-9]*)/\1 tests passed' "$OUT/$1.log" && ! grep -q 'FAIL' "$OUT/$1.log"
}
OPS=GATED_DELTA_NET,GATED_DELTA_NET_INDEXED,GATED_DELTA_NET_CACHE_FUSION,GATED_DELTA_NET_INDEXED_CACHE_FUSION

case $MODE in
--preflight)
    verify_all
    note "preflight ok ($COMMIT)"
    exit 0 ;;
--canary)
    verify_all; check_gpu "${GPUS[0]}"
    tbo canary-cand cand 0 -- -o GATED_DELTA_NET -p 'head_count=4,'
    passed canary-cand || die 8 "canary: candidate fails (see $OUT/canary-cand.log)"
    tbo canary-mutant mutant 0 -- -o GATED_DELTA_NET -p 'head_count=4,'
    if passed canary-mutant; then die 5 "canary: mutant passed; the NMSE check cannot see a wrong reduction"; fi
    note "canary pass: candidate passes, mutant fails on GATED_DELTA_NET head_count=4"
    exit 0 ;;
--run) ;;
*) die 8 "unknown mode $MODE" ;;
esac

verify_all
# 1. correctness against the CPU backend (NMSE tolerance; the reduction order changed, so bitwise equality with stack8 is not a gate)
for g in 0 1; do
    check_gpu "${GPUS[$g]}"
    tbo "cand-gpu$g" cand "$g" -- -o "$OPS"
    passed "cand-gpu$g" || die 8 "candidate fails on GPU $g (see $OUT/cand-gpu$g.log)"
done
tbo base-gpu0 base 0 -- -o "$OPS"
passed base-gpu0 || die 8 "base (stack8) fails the same cases (see $OUT/base-gpu0.log): the case list is not a valid gate"
note "correctness pass: $(grep -Eh 'tests passed' "$OUT"/cand-gpu0.log "$OUT"/cand-gpu1.log "$OUT"/base-gpu0.log | tr -s ' ' | tr '\n' ';')"

# 2. negative control: a wrong attn-score reduction must fail the same cases
tbo mutant-gpu0 mutant 0 -- -o "$OPS"
if passed mutant-gpu0; then die 5 "mutant passed all cases; the NMSE check is not a gate for this lever"; fi
note "mutant caught: $(grep -m1 -E 'FAIL|ERR' "$OUT/mutant-gpu0.log")"

# 3. perf, per GPU, base and cand in ABBA order over 4 rounds: the base-vs-base spread is the noise floor.
#    A: split 48-head shapes (head_count=24, head_size 128, n_seq_tokens 1 and 6, n_seqs 1 and 2); B: every GATED_DELTA_NET shape (prefill 32/4 heads).
for g in 0 1; do
    check_gpu "${GPUS[$g]}"
    for r in 1 2 3 4; do
        if (( r == 1 || r == 4 )); then order=(base cand); else order=(cand base); fi
        for arm in "${order[@]}"; do
            tbo "perfA-gpu$g-$arm-$r" "$arm" "$g" -- perf -o "$OPS" -p 'head_count=24,'
            [[ $RC == 0 ]] || die 8 "perf A $arm gpu$g round $r (see $OUT/perfA-gpu$g-$arm-$r.log)"
            tbo "perfB-gpu$g-$arm-$r" "$arm" "$g" -- perf -o GATED_DELTA_NET
            [[ $RC == 0 ]] || die 8 "perf B $arm gpu$g round $r (see $OUT/perfB-gpu$g-$arm-$r.log)"
        done
    done
done
python3 -P "$HERE/perf_summary.py" "$OUT" > "$OUT/perf.tsv" || die 8 "perf summary (see $OUT/perf.tsv)"
[[ -s $OUT/perf.tsv ]] || die 8 "no perf lines"
note "perf table in $OUT/perf.tsv (cols: gpu, case, base us, cand us, delta %, base spread %, cand spread %)"
guard
note "PASS $COMMIT"
