#!/usr/bin/env bash
# CUDA checks of the joint-sequence fused GDN kernel on one RTX 5060 Ti: test-backend-ops of frozen/np2-getter, no model, no server.
#   cuda-checks.sh --preflight | --canary | --run     env: OUT (required), COMMIT (frozen SOURCE sha), BUDGET_S (default 1200)
# rc: 0 pass | 3 Xid | 4 GPU missing or busy | 5 mutant not caught in 5 tries (negative control inconclusive)
#     6 integrity (frozen bytes, libs, engagement) | 7 budget | 8 the new build fails a correctness case
set -euo pipefail
export GIT_NO_LAZY_FETCH=1
FZ=/home/xanewok/repos/qwen38-eval-worktrees/frozen/np2-getter
BIN=$FZ/build/bin
GPU=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46
MODE=${1:?usage: cuda-checks.sh --preflight|--canary|--run}
OUT=${OUT:?OUT required}
COMMIT=${COMMIT:?COMMIT required}
BUDGET_S=${BUDGET_S:-1200}
START=$SECONDS
CUTOFF=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
die() { local rc=$1; shift; echo "cuda-checks: FAIL rc=$rc $*" | tee -a "$SUMMARY" >&2; exit "$rc"; }
note() { echo "cuda-checks: $*" | tee -a "$SUMMARY"; }

verify_frozen() {
    (cd "$BIN" && sha256sum -c --quiet SHA256SUMS) || die 6 "frozen bytes changed"
    [[ $(head -1 "$FZ/SOURCE" | cut -d' ' -f1) == "$COMMIT" ]] || die 6 "frozen bytes changed: SOURCE is not $COMMIT"
    [[ -x $BIN/test-backend-ops ]] || die 6 "no test-backend-ops in $BIN"
    [[ -z $(ldd "$BIN/test-backend-ops" | grep 'not found' || true) ]] || die 6 "unresolved libraries"
    local stray
    stray=$(ldd "$BIN/test-backend-ops" | awk '/libggml/ {print $3}' | grep -v "^$BIN/" || true)
    [[ -z $stray ]] || die 6 "ggml libs resolve outside the frozen dir: $stray"
    strings -a "$BIN/libggml-cuda.so" | grep -q 'gdn_joint_mutant: engaged' || die 6 "libggml-cuda lacks the joint kernel mutant switch"
}
guard() {
    if journalctl -k -b --since "$CUTOFF" --no-pager 2>/dev/null | grep -Eq '\bXid\b|fallen off the bus'; then die 3 "Xid since $CUTOFF"; fi
    (( SECONDS - START < BUDGET_S )) || die 7 "budget expired"
}
check_gpu() {
    local name mem
    name=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v u="$GPU" '$1 == u {print $2}')
    [[ -n $name ]] || die 4 "GPU $GPU missing"
    [[ $name == *"RTX 5060 Ti"* ]] || die 6 "GPU $GPU is not an RTX 5060 Ti: $name"
    for _ in $(seq 45); do
        mem=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u="$GPU" '$1 == u {print $2}')
        (( mem <= 512 )) && return 0
        sleep 1
    done
    die 4 "GPU $GPU memory did not release"
}
RC=0
tbo() { # TAG [ENV=VAL ...] -- ARGS...
    local tag=$1; shift
    local envs=()
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    guard
    RC=0
    env CUDA_VISIBLE_DEVICES="$GPU" GGML_CUDA_DEBUG_GDN_STATE=1 "${envs[@]}" \
        timeout 600 nice -n 5 "$BIN/test-backend-ops" -b CUDA0 "$@" > "$OUT/$tag.log" 2>&1 || RC=$?
    (( RC != 124 )) || die 8 "$tag hung (600 s)"
    printf '%s\t%s\n' "$tag" "$RC" >> "$OUT/runs.tsv"
}
passed() { # TAG: rc 0, a non-empty "N/N tests passed" and no FAIL line
    [[ $RC == 0 ]] && grep -Eq '^ +([1-9][0-9]*)/\1 tests passed' "$OUT/$1.log" && ! grep -q 'FAIL' "$OUT/$1.log"
}
engaged() { # TAG REGEX...
    local tag=$1; shift
    for re in "$@"; do grep -Eq "$re" "$OUT/$tag.log" || die 6 "engagement: no /$re/ in $OUT/$tag.log"; done
}
JOINT2='gdn_cache_fusion: graph device=0 graph=[0-9]+ tokens=[0-9]+ seqs=2 indexed=1 gdn=1 fused=1\b'

case $MODE in
--preflight)
    verify_frozen
    note "preflight ok ($COMMIT)"
    exit 0 ;;
--canary)
    verify_frozen; check_gpu
    tbo canary -- -o GATED_DELTA_NET_INDEXED_CACHE_FUSION -p 'n_seqs=2,'
    passed canary || die 8 "canary cases fail (see $OUT/canary.log)"
    engaged canary "$JOINT2"
    note "canary pass: two indexed sequences fused in one launch"
    exit 0 ;;
--run) ;;
*) die 8 "unknown mode $MODE" ;;
esac

verify_frozen; check_gpu
# 1. correctness: joint kernel (2..4 sequences), refusal above 4, and the unchanged one-sequence paths
tbo indexed-fusion -- -o GATED_DELTA_NET_INDEXED_CACHE_FUSION
passed indexed-fusion || die 8 "indexed fusion cases (see $OUT/indexed-fusion.log)"
engaged indexed-fusion "$JOINT2" 'seqs=4 indexed=1 gdn=1 fused=1\b' 'seqs=5 indexed=1 gdn=1 fused=0\b'
tbo plain-fusion -- -o GATED_DELTA_NET_CACHE_FUSION
passed plain-fusion || die 8 "plain fusion cases (see $OUT/plain-fusion.log)"
tbo gdn -- -o GATED_DELTA_NET
passed gdn || die 8 "GATED_DELTA_NET cases (see $OUT/gdn.log)"
note "correctness pass: $(grep -Eh 'tests passed' "$OUT"/{indexed-fusion,plain-fusion,gdn}.log | tr -s ' ' | tr '\n' ';')"

# 2. negative control: the same build with each sequence in its own blocks must race on the swapped ids
caught=0
for i in 1 2 3 4 5; do
    tbo "mutant-$i" GGML_CUDA_GDN_JOINT_MUTANT=1 -- -o GATED_DELTA_NET_INDEXED_CACHE_FUSION -p 'head_count=1024,'
    engaged "mutant-$i" 'gdn_joint_mutant: engaged'
    if ! passed "mutant-$i"; then caught=$i; break; fi
done
(( caught )) || die 5 "mutant (per-sequence blocks) passed 5 times; the race amplifier did not expose it"
note "mutant caught on try $caught: $(grep -m1 -E 'FAIL|ERR' "$OUT/mutant-$caught.log")"

# 3. perf at the split 48-head verify shape: gathered (per-sequence blocks) vs indexed (joint), and the racy per-sequence layout
for r in 1 2 3; do
    tbo "perf-$r" -- perf -o GATED_DELTA_NET_CACHE_FUSION,GATED_DELTA_NET_INDEXED_CACHE_FUSION -p 'head_count=24,'
    [[ $RC == 0 ]] || die 8 "perf run $r (see $OUT/perf-$r.log)"
    tbo "perf-mutant-$r" GGML_CUDA_GDN_JOINT_MUTANT=1 -- perf -o GATED_DELTA_NET_INDEXED_CACHE_FUSION -p 'head_count=24,'
    [[ $RC == 0 ]] || die 8 "perf mutant run $r (see $OUT/perf-mutant-$r.log)"
done
grep -Eh '(CACHE_FUSION)\(.*us/run' "$OUT"/perf-*.log | sed -E 's/^ +//' | sort > "$OUT/perf.txt" || true
[[ -s $OUT/perf.txt ]] || die 8 "no perf lines"
note "perf lines in $OUT/perf.txt (joint kernel: indexed n_seqs=2 rows of perf-N.log; per-sequence blocks: perf-mutant-N.log)"
guard
note "PASS $COMMIT"
