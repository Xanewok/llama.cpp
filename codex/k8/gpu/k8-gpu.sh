#!/usr/bin/env bash
# K8: MMQ small-I tiles partitioned like the default tile (stream-k included). test-backend-ops of the window's build on both RTX 5060 Ti.
#   k8-gpu.sh checks|perf --preflight|--canary|--run     env: OUT, COMMIT (build SOURCE sha), BUDGET_S
# rc: 0 pass | 3 Xid | 4 GPU missing or busy | 5 negative control void | 6 integrity (build bytes, engagement)
#     7 budget | 8 a correctness or bitwise gate fails
set -euo pipefail
export GIT_NO_LAZY_FETCH=1
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
BIN=$ROOT/k8-bin   # test-backend-ops + libs of build-cuda, RUNPATH $ORIGIN (bw-k8.sh stage 'bin')
STOCK=/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack8/build/bin   # base 41d2954a2 of L1 and K8, same libggml soname, stock MMQ
EXE=$BIN
GPUS=(GPU-ee9a3956-297c-604b-b4c1-f07930d27e46 GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f)
KIND=${1:?usage: k8-gpu.sh checks|perf --preflight|--canary|--run}
MODE=${2:?usage: k8-gpu.sh checks|perf --preflight|--canary|--run}
OUT=${OUT:?OUT required}
COMMIT=${COMMIT:?COMMIT required}
BUDGET_S=${BUDGET_S:-4800}
START=$SECONDS
CUTOFF=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
die() { local rc=$1; shift; echo "k8-gpu: FAIL rc=$rc $*" | tee -a "$SUMMARY" >&2; exit "$rc"; }
note() { echo "k8-gpu: $*" | tee -a "$SUMMARY"; }

verify_build() {
    (cd "$BIN" && sha256sum -c --quiet SHA256SUMS) || die 6 "build bytes changed"
    [[ $(cut -d' ' -f1 "$BIN/SOURCE") == "$COMMIT" ]] || die 6 "build bytes changed: SOURCE is not $COMMIT"
    [[ -x $BIN/test-backend-ops ]] || die 6 "no test-backend-ops in $BIN"
    [[ -z $(ldd "$BIN/test-backend-ops" | grep 'not found' || true) ]] || die 6 "unresolved libraries"
    local stray
    stray=$(ldd "$BIN/test-backend-ops" | awk '/libggml/ {print $3}' | grep -v "^$BIN/" || true)
    [[ -z $stray ]] || die 6 "ggml libs resolve outside $BIN: $stray"
    grep -aq 'GGML_CUDA_MMQ_SK_MUTANT' "$BIN/libggml-cuda.so" || die 6 "libggml-cuda lacks the stream-k mutant (not the K8 build)"
    grep -aq 'units=%d parts=%d blocks=%d' "$BIN/libggml-cuda.so" || die 6 "libggml-cuda lacks the K8 tile log (not the K8 build)"
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
    ! grep -aq 'FAIL' "$OUT/$1.log" || return 1
    n=$(grep -aEo '^ +[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/$1.log" | awk -F'[ /]+' 'NR == 1 {print ($2 == $3) ? $2 : 0}' || true)
    (( ${n:-0} >= ${2:-1} ))
}
engaged() { # TAG FIXED_STRING...
    local tag=$1; shift
    for s in "$@"; do grep -aqF -- "$s" "$OUT/$tag.log" || die 6 "engagement: no '$s' in $OUT/$tag.log"; done
}
absent() { # TAG FIXED_STRING: negative control
    if grep -aqF -- "$2" "$OUT/$1.log"; then die 6 "engagement: unexpected '$2' in $OUT/$1.log"; fi
}

# Tile log at nsm=36 (D): ntiles counts 64-row tiles, units the default 128-row tiles; plain tiling parts == units, else stream-k parts = nsm.
L='mmq_tiles: type='
ENG_PLAIN=("${L}q6_K m=8704 cols=12 z=1 J=16 ntiles=136 nsm=36 small_i=1 units=68 parts=68 blocks=136"
           "${L}iq4_xs m=9216 cols=24 z=1 J=24 ntiles=144 nsm=36 small_i=1 units=72 parts=72 blocks=144"
           "${L}q5_K m=124160 cols=12 z=1 J=16 ntiles=1940 nsm=36 small_i=1 units=970 parts=970 blocks=1940"
           "${L}q4_K m=8704 cols=16 z=1 J=16 ntiles=136 nsm=36 small_i=1 units=68 parts=68 blocks=136")
ENG_SK_ON=("${L}q6_K m=3072 cols=12 z=1 J=16 ntiles=48 nsm=36 small_i=1 units=24 parts=36 blocks=72"
           "${L}q4_K m=3072 cols=24 z=1 J=24 ntiles=48 nsm=36 small_i=1 units=24 parts=36 blocks=72"
           "${L}iq4_xs m=5120 cols=12 z=1 J=16 ntiles=80 nsm=36 small_i=1 units=40 parts=36 blocks=72"
           "${L}q6_K m=6144 cols=12 z=1 J=16 ntiles=96 nsm=36 small_i=1 units=48 parts=36 blocks=72")
ENG_SK_OFF=("${L}q6_K m=3072 cols=12 z=1 J=16 ntiles=48 nsm=36 small_i=0 units=24 parts=36 blocks=72"
            "${L}iq4_xs m=5120 cols=12 z=1 J=16 ntiles=80 nsm=36 small_i=0 units=40 parts=36 blocks=72"
            "${L}q6_K m=6144 cols=12 z=1 J=16 ntiles=96 nsm=36 small_i=0 units=48 parts=36 blocks=72")
P_MM='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|5120|6144|8704|9216|124160),n=(9|12|16|17|24|25),k=(1024|256),'
P_ID='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,n_mats=4,n_used=2,b=0,m=(3072|5120|8704),n=(12|24|48),k=(512|1024),'
MIN_MM=160   # 5 types x (5 m x 6 n + 2 n at m=124160)
MIN_ID=30    # 5 types x (2 n at m=8704 + 2 m x 2 n)

eval_pair() { # TAG GPU_INDEX [ENV...]: the MUL_MAT and MUL_MAT_ID cases
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
    ! grep -aEq '^ +MUL_MAT\(type_a=q8_0.*FAIL' "$OUT/$1.log" || die 5 "negative control void: $1 fails q8_0 cases, which never take a dropped launch"
    # stream-k shapes must be among the failures: the dropped launch there is a K8 launch
    grep -aEq 'MUL_MAT\(type_a=(iq4_xs|q4_K|q5_K|q6_K),type_b=f32,m=(3072|5120|6144),n=(9|12|16|17|24),k=1024.*FAIL' "$OUT/$1.log" ||
        die 5 "negative control void: $1 fails no stream-k shape (m=3072|5120|6144), so the K8 launches are not taken"
    note "$1: ${failed} of $ran cases fail with small-I launches dropped, as required"
}
hash_keys() { cut -d' ' -f1-4 "$1"; }
hash_diff() { # A B: "type m k n" whose hash differs in B
    awk 'NR == FNR {h[$1" "$2" "$3" "$4] = $5; next} h[$1" "$2" "$3" "$4] != $5 {print $1, $2, $3, $4}' "$1" "$2" | sort
}
# np2 model shapes (D, from the partition model codex/k8/mmq_sk_sim.cpp at nsm=36): default tile runs stream-k / plain tiling
SK_KEYS=()
PLAIN_KEYS=()
for t in iq4_xs q4_K q5_K q6_K; do
    for n in 12 24; do
        for mk in "3072 5120" "5120 5120" "6144 5120" "5120 8704"; do
            read -r m k <<< "$mk"; SK_KEYS+=("$t m=$m k=$k n=$n")
        done
        for mk in "8704 5120" "9216 5120" "17408 5120" "8704 8704"; do
            read -r m k <<< "$mk"; PLAIN_KEYS+=("$t m=$m k=$k n=$n")
        done
    done
done
PLAIN_KEYS+=("q6_K m=124160 k=512 n=12" "q6_K m=124160 k=512 n=24")
keys_in() { # DIFFFILE KEY...: every key differs
    local f=$1 key; shift
    for key in "$@"; do grep -aqxF -- "$key" "$f" || { echo "$key"; return 1; }; done
}
keys_out() { # DIFFFILE KEY...: no key differs
    local f=$1 key; shift
    for key in "$@"; do ! grep -aqxF -- "$key" "$f" || { echo "$key"; return 1; }; done
}
stock_setup() { # copy of the K8 test binary beside symlinks to the stack8 libs: same code, stock MMQ
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

hash_gates() { # G: bitwise gates of the K8 claim, plus the controls that prove they can fail
    local g=$1 d=$OUT/hash-g$1 mode miss
    for mode in default-2 auto auto-l1 auto-drop mutant; do
        cmp -s <(hash_keys "$d-default.out") <(hash_keys "$d-$mode.out") || die 8 "g$g: $mode hashes cover different cases than default"
    done
    cmp -s "$d-default.out" "$d-default-2.out" || die 5 "g$g: default tiles not deterministic run to run; hash comparison void"
    cmp -s "$d-default.out" "$d-stock.out" || die 8 "g$g: K8 default path differs from stock MMQ (see $d-stock.out)"
    for mode in auto auto-l1 mutant auto-drop; do hash_diff "$d-default.out" "$d-$mode.out" > "$d-$mode.keys"; done
    # the claim: narrow tiles, stream-k or plain, are bitwise the default tile
    [[ ! -s $d-auto.keys ]] || die 8 "g$g: auto (SMALL_I incl. stream-k) differs from the default tile on $(wc -l < "$d-auto.keys") cases (see $d-auto.keys)"
    [[ ! -s $d-auto-l1.keys ]] || die 8 "g$g: auto-l1 differs from the default tile on $(wc -l < "$d-auto-l1.keys") cases (see $d-auto-l1.keys)"
    # controls: the gate must be able to fail where K8 launches
    miss=$(keys_in "$d-auto-drop.keys" "${SK_KEYS[@]}" "q6_K m=8704 k=5120 n=12" "q6_K m=124160 k=512 n=12") ||
        die 5 "g$g: negative control void: auto-drop leaves '$miss' equal to default, so no SMALL_I launch happens there"
    miss=$(keys_in "$d-mutant.keys" "${SK_KEYS[@]}") ||
        die 5 "g$g: negative control void: the shifted stream-k partition leaves '$miss' equal to default"
    miss=$(keys_out "$d-mutant.keys" "${PLAIN_KEYS[@]}") ||
        die 5 "g$g: negative control void: the stream-k mutant changed plain-tiling case '$miss'"
    [[ -z $(comm -23 "$d-mutant.keys" "$d-auto-drop.keys") ]] || die 5 "g$g: negative control void: the mutant differs where no SMALL_I launch happens"
    note "g$g bitwise: auto and auto-l1 equal default on all $(wc -l < "$d-default.out") cases; controls differ: auto-drop $(wc -l < "$d-auto-drop.keys"), stream-k mutant $(wc -l < "$d-mutant.keys") ($(grep -ac '^id_' "$d-mutant.keys") expert-path)"
    note "g$g expert path: auto-drop differs on $(grep -ac '^id_' "$d-auto-drop.keys") of $(grep -ac '^id_' "$d-default.out") id cases (0 means no SMALL_I launch there: engagement unverified)"
}

mode_checks() {
    local g tag mode
    case $MODE in
    --canary)
        check_gpu "${GPUS[0]}"
        tbo canary 0 300 GGML_CUDA_MMQ_TILES_LOG=1 -- -b CUDA0 -o MUL_MAT -p 'type_a=q6_K,type_b=f32,m=(3072|8704),n=12,k=1024,'
        passed canary 2 || die 8 "canary cases fail (see $OUT/canary.log)"
        engaged canary "${ENG_PLAIN[0]}" "${ENG_SK_ON[0]}"
        tbo canary-l1 0 300 GGML_CUDA_MMQ_TILES=auto-l1 GGML_CUDA_MMQ_TILES_LOG=1 -- -b CUDA0 -o MUL_MAT -p 'type_a=q6_K,type_b=f32,m=(3072|8704),n=12,k=1024,'
        passed canary-l1 2 || die 8 "canary auto-l1 cases fail"
        engaged canary-l1 "${ENG_PLAIN[0]}" "${ENG_SK_OFF[0]}"
        tbo canary-default 0 300 GGML_CUDA_MMQ_TILES=default GGML_CUDA_MMQ_TILES_LOG=1 -- -b CUDA0 -o MUL_MAT -p 'type_a=q6_K,type_b=f32,m=(3072|8704),n=12,k=1024,'
        passed canary-default 2 || die 8 "canary default-tiles cases fail"
        absent canary-default "$L"
        note "canary pass: auto takes small-I on m=8704 (plain, 136 blocks) and m=3072 (stream-k, 72 blocks), auto-l1 only on m=8704, default none"
        return ;;
    --run) ;;
    *) die 8 "unknown mode $MODE" ;;
    esac

    for g in 0 1; do
        check_gpu "${GPUS[$g]}"
        # 1. correctness vs CPU: the K8 policy, the L1 policy, stock tiles
        eval_pair "g$g-auto" "$g" GGML_CUDA_MMQ_TILES_LOG=1
        engaged "g$g-auto-mm" "${ENG_PLAIN[@]}" "${ENG_SK_ON[@]}"
        absent "g$g-auto-mm" "${L}q8_0"
        eval_pair "g$g-autol1" "$g" GGML_CUDA_MMQ_TILES=auto-l1 GGML_CUDA_MMQ_TILES_LOG=1
        engaged "g$g-autol1-mm" "${ENG_PLAIN[@]}" "${ENG_SK_OFF[@]}"
        eval_pair "g$g-default" "$g" GGML_CUDA_MMQ_TILES=default GGML_CUDA_MMQ_TILES_LOG=1
        absent "g$g-default-mm" "$L"
        note "g$g correctness pass: $(grep -aEh 'tests passed' "$OUT"/g$g-{auto,autol1,default}-{mm,id}.log | tr -s ' ' | tr '\n' ';')"
    done

    # 2. bitwise: every setting against the default tile; auto-drop and the stream-k mutant are the controls
    stock_setup
    for g in 0 1; do
        check_gpu "${GPUS[$g]}"
        for mode in default default-2 auto auto-l1 auto-drop mutant; do
            tag="hash-g$g-$mode"
            case $mode in
                default-2) setting=(GGML_CUDA_MMQ_TILES=default) ;;
                mutant)    setting=(GGML_CUDA_MMQ_TILES=auto GGML_CUDA_MMQ_SK_MUTANT=1) ;;
                *)         setting=(GGML_CUDA_MMQ_TILES=$mode) ;;
            esac
            tbo "$tag" "$g" 1500 "${setting[@]}" -- mmq-hash CUDA0
            [[ $RC == 0 && -s $OUT/$tag.out ]] || die 8 "$tag failed (see $OUT/$tag.log)"
        done
        EXE=$STOCK_EXE
        tbo "hash-g$g-stock" "$g" 1500 -- mmq-hash CUDA0
        EXE=$BIN
        [[ $RC == 0 && -s $OUT/hash-g$g-stock.out ]] || die 8 "hash-g$g-stock failed (see $OUT/hash-g$g-stock.log)"
        hash_gates "$g"
        tbo "g$g-drop-mm" "$g" 900 GGML_CUDA_MMQ_TILES=auto-drop -- -b CUDA0 -o MUL_MAT -p "$P_MM"
        must_fail "g$g-drop-mm"
    done
    cmp -s "$OUT/hash-g0-auto.out" "$OUT/hash-g1-auto.out" || note "auto hashes differ between the two GPUs"

    # 3. the whole existing suites on the new build (default/auto dispatch of every other shape and type)
    check_gpu "${GPUS[0]}"
    tbo suite-mm 0 2400 -- -b CUDA0 -o MUL_MAT
    passed suite-mm 500 || die 8 "full MUL_MAT suite (see $OUT/suite-mm.log)"
    tbo suite-id 0 2400 -- -b CUDA0 -o MUL_MAT_ID
    passed suite-id 100 || die 8 "full MUL_MAT_ID suite (see $OUT/suite-id.log)"
    guard
    note "PASS $COMMIT"
}

# Hot weights fit the L2 cache and understate the weight-read cost; cold rotates the weights over >= 64 MiB.
# Shapes (m,k): gate/up, down, attention q, GDN qkv, z, out_proj, plus 9216/17408 (L1) and the lm_head; n = columns.
shapes() { # N_REGEX
    echo "(m=8704,n=$1,k=5120|m=5120,n=$1,k=8704|m=3072,n=$1,k=5120|m=6144,n=$1,k=5120|m=5120,n=$1,k=3072|m=5120,n=$1,k=5120|m=9216,n=$1,k=5120|m=17408,n=$1,k=5120|m=124160,n=$1,k=5120),"
}
mode_perf() {
    local rep mode
    case $MODE in
    --canary)
        check_gpu "${GPUS[0]}"
        tbo canary 0 600 GGML_CUDA_MMQ_TILES_LOG=1 GGML_TEST_PERF_COLD_MB=64 -- perf -b CUDA0 -o MUL_MAT -j 1 --output sql -p 'type_a=q6_K,type_b=f32,m=3072,n=12,k=5120,'
        [[ $RC == 0 ]] || die 8 "perf canary failed"
        python3 "$HERE/perf_summary.py" "$OUT" canary || die 8 "perf canary unreadable"
        engaged canary "${ENG_SK_ON[0]}"
        note "perf canary pass"
        return ;;
    --run) ;;
    *) die 8 "unknown mode $MODE" ;;
    esac
    local base=(perf -b CUDA0 -o MUL_MAT -j 1 --output sql)
    local ty='type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,'
    local hot="$ty$(shapes '(12|24)')" cold="$ty$(shapes 12)"
    # A, B, C, D, A again per repetition: default tile / L1 policy / K8 policy / auto-drop (no SMALL_I GEMM: a floor, not a null) / default
    local order=(default auto-l1 auto auto-drop default)
    for rep in 1 2; do
        check_gpu "${GPUS[0]}"
        for mode in "${order[@]}"; do
            tbo "p-g0-hot-r$rep-$mode-$((++N))" 0 1800 GGML_CUDA_MMQ_TILES=$mode -- "${base[@]}" -p "$hot"
            [[ $RC == 0 ]] || die 8 "perf run failed (see $OUT/p-g0-hot-r$rep-$mode-$N.log)"
        done
    done
    for mode in default auto-l1 auto default auto-drop; do
        tbo "p-g0-cold-$mode-$((++N))" 0 3600 GGML_CUDA_MMQ_TILES=$mode GGML_TEST_PERF_COLD_MB=64 -- "${base[@]}" -p "$cold"
        [[ $RC == 0 ]] || die 8 "cold perf run failed (see $OUT/p-g0-cold-$mode-$N.log)"
    done
    check_gpu "${GPUS[1]}"
    for mode in default auto-l1 auto default; do
        tbo "p-g1-hot-r1-$mode-$((++N))" 1 1800 GGML_CUDA_MMQ_TILES=$mode -- "${base[@]}" -p "$hot"
        [[ $RC == 0 ]] || die 8 "GPU1 perf run failed (see $OUT/p-g1-hot-r1-$mode-$N.log)"
    done
    python3 "$HERE/perf_summary.py" "$OUT" run > "$OUT/perf.tsv" || die 8 "perf summary failed"
    note "perf table: $OUT/perf.tsv"
    guard
    note "PASS $COMMIT"
}

case $MODE in
--preflight)
    verify_build
    [[ $KIND == checks || $KIND == perf ]] || die 6 "unknown kind $KIND"
    note "preflight ok ($KIND $COMMIT)"
    exit 0 ;;
esac
verify_build
case $KIND in
checks) mode_checks ;;
perf) mode_perf ;;
*) die 8 "unknown kind $KIND" ;;
esac
