#!/usr/bin/env bash
# L1 shape-gated small-I MMQ tiles: test-backend-ops of frozen/l1-mmq-smallj on both RTX 5060 Ti. No model, no server.
#   l1-gpu.sh checks|perf --preflight|--canary|--run     env: OUT, COMMIT (frozen SOURCE sha), BUDGET_S
# rc: 0 pass | 3 Xid | 4 GPU missing or busy | 5 negative control void | 6 integrity (frozen bytes, engagement)
#     7 budget | 8 a correctness or determinism case fails
set -euo pipefail
export GIT_NO_LAZY_FETCH=1
FZ=/home/xanewok/repos/qwen38-eval-worktrees/frozen/l1-mmq-smallj
BIN=$FZ/build/bin
STOCK=/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack8/build/bin   # L1 base 41d2954a2, same libggml soname
EXE=$BIN
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
GPUS=(GPU-ee9a3956-297c-604b-b4c1-f07930d27e46 GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f)
KIND=${1:?usage: l1-gpu.sh checks|perf --preflight|--canary|--run}
MODE=${2:?usage: l1-gpu.sh checks|perf --preflight|--canary|--run}
OUT=${OUT:?OUT required}
COMMIT=${COMMIT:?COMMIT required}
BUDGET_S=${BUDGET_S:-3000}
START=$SECONDS
CUTOFF=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
die() { local rc=$1; shift; echo "l1-gpu: FAIL rc=$rc $*" | tee -a "$SUMMARY" >&2; exit "$rc"; }
note() { echo "l1-gpu: $*" | tee -a "$SUMMARY"; }

verify_frozen() {
    (cd "$BIN" && sha256sum -c --quiet SHA256SUMS) || die 6 "frozen bytes changed"
    [[ $(head -1 "$FZ/SOURCE" | cut -d' ' -f1) == "$COMMIT" ]] || die 6 "frozen bytes changed: SOURCE is not $COMMIT"
    [[ -x $BIN/test-backend-ops ]] || die 6 "no test-backend-ops in $BIN"
    [[ -z $(ldd "$BIN/test-backend-ops" | grep 'not found' || true) ]] || die 6 "unresolved libraries"
    local stray
    stray=$(ldd "$BIN/test-backend-ops" | awk '/libggml/ {print $3}' | grep -v "^$BIN/" || true)
    [[ -z $stray ]] || die 6 "ggml libs resolve outside the frozen dir: $stray"
    grep -aq 'mmq_tiles: type=' "$BIN/libggml-cuda.so" || die 6 "libggml-cuda lacks the mmq_tiles log (not the L1 build)"
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
N=0
tbo() { # TAG GPU_INDEX TIMEOUT [ENV=VAL ...] -- ARGS...   stdout and stderr -> $OUT/TAG.log; stdout alone -> $OUT/TAG.out
    local tag=$1 gpu=${GPUS[$2]} tmo=$3; shift 3
    local envs=()
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    guard
    RC=0
    env CUDA_VISIBLE_DEVICES="$gpu" "${envs[@]}" \
        timeout "$tmo" nice -n 5 "$EXE/test-backend-ops" "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err" || RC=$?
    cat "$OUT/$tag.out" "$OUT/$tag.err" > "$OUT/$tag.log"
    (( RC != 124 )) || die 8 "$tag hung ($tmo s)"
    printf '%s\t%s\n' "$tag" "$RC" >> "$OUT/runs.tsv"
}
passed() { # TAG [MIN_CASES]: rc 0, "N/N tests passed" with N >= MIN_CASES, no FAIL line
    local n
    [[ $RC == 0 ]] || return 1
    ! grep -q 'FAIL' "$OUT/$1.log" || return 1
    n=$(grep -Eo '^ +[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/$1.log" | awk -F'[ /]+' 'NR == 1 {print ($2 == $3) ? $2 : 0}' || true)
    (( ${n:-0} >= ${2:-1} ))
}
engaged() { # TAG REGEX...
    local tag=$1; shift
    for re in "$@"; do grep -Eq "$re" "$OUT/$tag.log" || die 6 "engagement: no /$re/ in $OUT/$tag.log"; done
}
absent() { # TAG REGEX: negative control
    grep -Eq "$2" "$OUT/$1.log" && die 6 "engagement: unexpected /$2/ in $OUT/$1.log" || true
}

# Gate prediction at nsm=36 (D): ntiles = ceil(m/64) * ceil(cols/J); small-I wins only at >= 90 % of whole SM waves.
ENG_ON=('mmq_tiles: type=q6_K m=8704 cols=12 z=1 J=16 ntiles=136 nsm=36 small_i=1'
        'mmq_tiles: type=iq4_xs m=9216 cols=24 z=1 J=24 ntiles=144 nsm=36 small_i=1'
        'mmq_tiles: type=q5_K m=124160 cols=12 z=1 J=16 ntiles=1940 nsm=36 small_i=1'
        'mmq_tiles: type=q4_K m=8704 cols=16 z=1 J=16 ntiles=136 nsm=36 small_i=1')
ENG_OFF=('mmq_tiles: type=q6_K m=3072 cols=12 z=1 J=16 ntiles=48 nsm=36 small_i=0'
         'mmq_tiles: type=q4_K m=3072 cols=24 z=1 J=24 ntiles=48 nsm=36 small_i=0')
P_MM='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|8704|9216|124160),n=(9|12|16|17|24|25),k=(1024|256),'
P_ID='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,n_mats=4,n_used=2,b=0,m=8704,n=(12|24),k=512,'
MIN_MM=100   # 5 types x (3 m x 6 n + 2 n at m=124160); the binary has no m=11520 MUL_MAT case, so the regex needs no 11520
MIN_ID=10

eval_pair() { # TAG GPU_INDEX [ENV...]: the new MUL_MAT and MUL_MAT_ID cases
    local tag=$1 g=$2; shift 2
    tbo "$tag-mm" "$g" 900 "$@" -- -b CUDA0 -o MUL_MAT -p "$P_MM"
    passed "$tag-mm" "$MIN_MM" || die 8 "$tag MUL_MAT cases fail or fewer than $MIN_MM ran (see $OUT/$tag-mm.log)"
    tbo "$tag-id" "$g" 900 "$@" -- -b CUDA0 -o MUL_MAT_ID -p "$P_ID"
    passed "$tag-id" "$MIN_ID" || die 8 "$tag MUL_MAT_ID cases fail or fewer than $MIN_ID ran (see $OUT/$tag-id.log)"
}

must_fail() { # TAG: negative control, MUL_MAT_Q cases ran and some failed; q8_0 (no small-I tile) never launches a dropped kernel
    local ran failed
    read -r ran failed < <(awk '/^ +[0-9]+\/[0-9]+ tests passed/ {split($1, a, "/"); print a[2], a[2] - a[1]; exit}' "$OUT/$1.log") || true
    (( ${ran:-0} >= MIN_MM )) || die 5 "negative control void: $1 ran ${ran:-0} cases, fewer than $MIN_MM (see $OUT/$1.log)"
    (( ${failed:-0} > 0 )) || die 5 "negative control void: $1 dropped the small-I launches and nothing failed"
    ! grep -Eq '(^|[[:space:]])MUL_MAT\(type_a=q8_0.*FAIL' "$OUT/$1.log" || die 5 "negative control void: $1 fails q8_0 cases, which never take a dropped launch"
    note "$1: ${failed} of $ran cases fail with small-I launches dropped, as required"
}
hash_keys() { cut -d' ' -f1-4 "$1"; }
hash_diff() { # A B: "type m k n" whose hash differs in B
    awk 'NR == FNR {h[$1" "$2" "$3" "$4] = $5; next} h[$1" "$2" "$3" "$4] != $5 {print $1, $2, $3, $4}' "$1" "$2" | sort
}
hash_gates() { # G
    local g=$1 d=$OUT/hash-g$1 mode
    for mode in default-2 auto auto-drop small; do
        cmp -s <(hash_keys "$d-default.out") <(hash_keys "$d-$mode.out") || die 8 "g$g: $mode hashes cover different cases than default"
    done
    cmp -s "$d-default.out" "$d-default-2.out" || die 5 "g$g: default tiles not deterministic run to run; hash comparison void"
    hash_diff "$d-default.out" "$d-auto.out" > "$d-auto.keys"
    hash_diff "$d-default.out" "$d-auto-drop.out" > "$d-auto-drop.keys"
    [[ -s $d-auto-drop.keys ]] || die 5 "g$g: negative control void: auto-drop equals default bitwise"
    [[ -z $(comm -23 "$d-auto.keys" "$d-auto-drop.keys") ]] || die 8 "g$g: auto differs from default where auto-drop does not (see $d-auto.keys, $d-auto-drop.keys)"
}
stock_setup() { # copy of the L1 test binary beside symlinks to the stack8 libs: same code, stock MMQ
    local d=$OUT/stock-bin f
    (cd "$STOCK" && grep -a ' libggml' SHA256SUMS | sha256sum -c --quiet) || die 6 "stack8 libggml bytes changed"
    [[ $(head -1 "$STOCK/../../SOURCE" | cut -d' ' -f1) == 41d2954a21106c9a65c8dd2a0697e7d33ae4f22f ]] || die 6 "stack8 is not 41d2954a2"
    rm -rf "$d"; mkdir -p "$d"
    cp "$BIN/test-backend-ops" "$d/"
    for f in "$STOCK"/libggml*.so*; do ln -s "$(readlink -f "$f")" "$d/$(basename "$f")"; done
    local libs stray
    libs=$(ldd "$d/test-backend-ops")
    [[ -z $(grep 'not found' <<< "$libs" || true) ]] || die 5 "stock arm: unresolved libraries in $d"
    [[ $(grep -c "libggml.* => $d/" <<< "$libs" || true) == 4 ]] || die 5 "stock arm: the 4 libggml* do not all resolve to $d"
    stray=$(awk '/libggml/ {print $3}' <<< "$libs" | grep -v "^$d/" || true)
    [[ -z $stray ]] || die 5 "stock arm: ggml libs resolve outside $d: $stray"
    STOCK_EXE=$d
}

mode_checks() {
    local g tag mode
    case $MODE in
    --canary)
        check_gpu "${GPUS[0]}"
        tbo canary 0 300 GGML_CUDA_MMQ_TILES_LOG=1 -- -b CUDA0 -o MUL_MAT -p 'type_a=q6_K,type_b=f32,m=(3072|8704),n=12,k=1024,'
        passed canary 2 || die 8 "canary cases fail (see $OUT/canary.log)"
        engaged canary "${ENG_ON[0]}" 'mmq_tiles: type=q6_K m=3072 cols=12 z=1 J=16 ntiles=48 nsm=36 small_i=0'
        tbo canary-default 0 300 GGML_CUDA_MMQ_TILES=default GGML_CUDA_MMQ_TILES_LOG=1 -- -b CUDA0 -o MUL_MAT -p 'type_a=q6_K,type_b=f32,m=8704,n=12,k=1024,'
        passed canary-default || die 8 "canary default-tiles case fails"
        absent canary-default 'mmq_tiles:'
        note "canary pass: m=8704 takes small-I tiles, m=3072 does not, GGML_CUDA_MMQ_TILES=default takes none"
        return ;;
    --run) ;;
    *) die 8 "unknown mode $MODE" ;;
    esac

    for g in 0 1; do
        check_gpu "${GPUS[$g]}"
        # 1. auto (the shipped policy): correctness plus the gate decision per shape
        eval_pair "g$g-auto" "$g" GGML_CUDA_MMQ_TILES_LOG=1
        engaged "g$g-auto-mm" "${ENG_ON[@]}" "${ENG_OFF[@]}"
        absent "g$g-auto-mm" 'mmq_tiles: type=q8_0'
        # 2. small-I kernels everywhere they exist, including the shapes the gate rejects
        eval_pair "g$g-small" "$g" GGML_CUDA_MMQ_TILES=small GGML_CUDA_MMQ_TILES_LOG=1
        engaged "g$g-small-mm" 'mmq_tiles: type=q6_K m=3072 cols=12 z=1 J=16 ntiles=48 nsm=36 small_i=1'
        # 3. stock tiles only: the switch must really turn the path off
        eval_pair "g$g-default" "$g" GGML_CUDA_MMQ_TILES=default GGML_CUDA_MMQ_TILES_LOG=1
        absent "g$g-default-mm" 'mmq_tiles:'
        note "g$g correctness pass: $(grep -Eh 'tests passed' "$OUT"/g$g-{auto,small,default}-{mm,id}.log | tr -s ' ' | tr '\n' ';')"
    done

    # 4. bitwise comparison per (type, m, k, n) across settings; auto-drop skips the small-I launch (dst unwritten): the control
    stock_setup
    for g in 0 1; do
        for mode in default default auto auto-drop small; do
            tag="hash-g$g-$mode"; [[ -e $OUT/$tag.out ]] && tag=$tag-2
            tbo "$tag" "$g" 1500 GGML_CUDA_MMQ_TILES=$mode -- mmq-hash CUDA0
            [[ $RC == 0 && -s $OUT/$tag.out ]] || die 8 "$tag failed (see $OUT/$tag.log)"
        done
        hash_gates "$g"
        EXE=$STOCK_EXE
        tbo "hash-g$g-stock" "$g" 1500 -- mmq-hash CUDA0
        EXE=$BIN
        [[ $RC == 0 && -s $OUT/hash-g$g-stock.out ]] || die 8 "hash-g$g-stock failed (see $OUT/hash-g$g-stock.log)"
        cmp -s "$OUT/hash-g$g-default.out" "$OUT/hash-g$g-stock.out" || die 8 "g$g: L1 default path differs from stock MMQ (see hash-g$g-stock.out)"
        tbo "g$g-drop-mm" "$g" 900 GGML_CUDA_MMQ_TILES=auto-drop -- -b CUDA0 -o MUL_MAT -p "$P_MM"
        must_fail "g$g-drop-mm"
        for mode in auto small; do
            diff "$OUT/hash-g$g-default.out" "$OUT/hash-g$g-$mode.out" | grep -c '^>' > "$OUT/hash-g$g-$mode.ndiff" || true
            diff "$OUT/hash-g$g-default.out" "$OUT/hash-g$g-$mode.out" | grep '^>' > "$OUT/hash-g$g-$mode.diff" || true
            note "g$g bitwise vs default tiles, mode $mode: $(cat "$OUT/hash-g$g-$mode.ndiff") of $(wc -l < "$OUT/hash-g$g-default.out") cases differ (list: hash-g$g-$mode.diff)"
        done
    done
    cmp -s "$OUT/hash-g0-auto.out" "$OUT/hash-g1-auto.out" || note "auto hashes differ between the two GPUs"

    # 5. the whole existing suites on the new build (default/auto dispatch of every other shape and type)
    check_gpu "${GPUS[0]}"
    tbo suite-mm 0 2400 -- -b CUDA0 -o MUL_MAT
    passed suite-mm 500 || die 8 "full MUL_MAT suite (see $OUT/suite-mm.log)"
    tbo suite-id 0 2400 -- -b CUDA0 -o MUL_MAT_ID
    passed suite-id 100 || die 8 "full MUL_MAT_ID suite (see $OUT/suite-id.log)"
    guard
    note "PASS $COMMIT"
}

mode_perf() {
    local rep mode
    case $MODE in
    --canary)
        check_gpu "${GPUS[0]}"
        tbo canary 0 600 GGML_CUDA_MMQ_TILES_LOG=1 GGML_TEST_PERF_COLD_MB=64 -- perf -b CUDA0 -o MUL_MAT -j 1 --output sql -p 'type_a=q6_K,type_b=f32,m=8704,n=12,k=5120,'
        [[ $RC == 0 ]] || die 8 "perf canary failed"
        python3 "$HERE/perf_summary.py" "$OUT" canary || die 8 "perf canary unreadable"
        engaged canary "${ENG_ON[0]}"
        note "perf canary pass"
        return ;;
    --run) ;;
    *) die 8 "unknown mode $MODE" ;;
    esac
    local base=(perf -b CUDA0 -o MUL_MAT -j 1 --output sql)
    local hot='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|5120|6144|8704|9216|17408|124160),n=(12|24),k=(5120|8704),'
    local cold='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|5120|6144|8704|9216|17408|124160),n=12,k=(5120|8704),'
    # A, B, C, A again per repetition: default tiles / auto policy / auto-drop (no GEMM at gated shapes: floor, not a null) / default
    local order=(default auto auto-drop default)
    for rep in 1 2; do
        check_gpu "${GPUS[0]}"
        for mode in "${order[@]}"; do
            tbo "p-g0-hot-r$rep-$mode-$((++N))" 0 1800 GGML_CUDA_MMQ_TILES=$mode -- "${base[@]}" -p "$hot"
            [[ $RC == 0 ]] || die 8 "perf run failed (see $OUT/p-g0-hot-r$rep-$mode-$N.log)"
        done
    done
    for mode in default auto default auto-drop; do
        tbo "p-g0-cold-$mode-$((++N))" 0 3600 GGML_CUDA_MMQ_TILES=$mode GGML_TEST_PERF_COLD_MB=64 -- "${base[@]}" -p "$cold"
        [[ $RC == 0 ]] || die 8 "cold perf run failed (see $OUT/p-g0-cold-$mode-$N.log)"
    done
    check_gpu "${GPUS[1]}"
    for mode in default auto default; do
        tbo "p-g1-hot-r1-$mode-$((++N))" 1 1800 GGML_CUDA_MMQ_TILES=$mode -- "${base[@]}" -p "$hot"
        [[ $RC == 0 ]] || die 8 "GPU1 perf run failed (see $OUT/p-g1-hot-r1-$mode-$N.log)"
    done
    python3 "$HERE/perf_summary.py" "$OUT" run > "$OUT/perf.tsv" || die 8 "perf summary failed"
    note "perf table: $OUT/perf.tsv (columns: gpu temp type m k n default_us auto_us autodrop_us auto-default aa_default_spread autodrop-default)"
    guard
    note "PASS $COMMIT"
}

case $MODE in
--preflight)
    verify_frozen
    [[ $KIND == checks || $KIND == perf ]] || die 6 "unknown kind $KIND"
    note "preflight ok ($KIND $COMMIT)"
    exit 0 ;;
esac
verify_frozen
case $KIND in
checks) mode_checks ;;
perf) mode_perf ;;
*) die 8 "unknown kind $KIND" ;;
esac
