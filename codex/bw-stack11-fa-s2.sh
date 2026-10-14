#!/usr/bin/env bash
# stack11-fa-s2 window (harness/bw-run.sh payload): build this tree = frozen/stack11-fa (8eda87ffb, pkg 93/0001) + the frozen/stack11-s2-nok
# commits (7e7b03962..b782906ee: 87/88/91/90, K7/K10 dropped), then the union of both parents' gates, then freeze frozen/stack11-fa-s2.
#   stack11-fa gates:     FLASH_ATTN_EXT full test-backend-ops on CUDA0 and CUDA1, served-flag health boots.
#   stack11-s2-nok gates: CPU-device unit tests, test-host-mapped bitwise + NO_PINNED control, GET_ROWS/TOP_K/norm ops vs the
#                         frozen/stack11-s2-nok binary, health boots with the p90 CUDA_Host_Mapped engagement line gating, K7/K10 lines absent.
# Health boots: all four served profiles. Contract: nothing is frozen unless every gate passed.
# usage: OUT | --preflight | --selftest (CPU only, recorded outputs in codex/stack11-fa-s2/fixtures). rc: gpuq map 0 3 4 5 6 7 8.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WIN=$ROOT/codex/stack11-s2/window
FIX=$ROOT/codex/stack11-fa-s2/fixtures
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BUILDGATE=$W/bin/buildgate
SERVE=$HOME/.claude/skills/local-qwen-server/bin/serve.sh
NAME=stack11-fa-s2
BRANCH=topic/stack11-fa-s2
FA_SHA=8eda87ffb7787c660fd6789e4f89d52483897833      # frozen/stack11-fa
NOK_SHA=b782906ee8ed04e3fed10e57eb774257e9f65ed4     # frozen/stack11-s2-nok
BASE_SHA=7e7b039628ada54d3b978dda866a45804388fcee    # stack11-base, parent of both
BASE_FROZEN=$W/frozen/stack11-s2-nok                 # ops reference: same tests source, no 93/0001
VISIBLE=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
CPUS=0-3,12-15
MIN_FA=4089   # stack11-fa-gates a1 (M): 4089/4089 per GPU; 87/88/91/90 change no FLASH_ATTN_EXT test
PROFILES='swift15-j5q swift15-j5q-duo swift15-iq4 swift15-iq4-duo'
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10 CUDA_VISIBLE_DEVICES=''
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
# gates.sh needs these names; the K tree is not built here.
KROOT=$ROOT SHA='' KSHA='' SHORT='' KSHORT='' STARTED='' COMPONENTS=p90 OUT=${OUT:-/nonexistent}
# shellcheck source=stack11-s2/window/gates.sh
source "$WIN/gates.sh"

flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)

# --- judges (pure: log files in, rc out; the selftest drives them on recorded outputs) ----------------------------------------------
# LOG: FLASH_ATTN_EXT run complete: no FAIL line, every "N/M tests passed" has N == M, at least MIN_FA cases.
fa_judge() {
    local n
    if grep -aq 'FAIL' "$1"; then echo "FAIL line in $1"; return 8; fi
    n=$(grep -aEo '[0-9]+/[0-9]+ tests passed' "$1" | awk -F'[ /]' '{ if ($1 != $2) bad = 1; s += $1 } END { print bad ? -1 : s + 0 }') || true
    (( ${n:-0} >= MIN_FA )) || { echo "FA cases ${n:-0} < $MIN_FA (or a partial N/M): $1"; return 8; }
    echo "pass: $(basename "$1") FA cases=$n"
}
# NEW BASE: op logs of this build and of the reference: no FAIL line in NEW absent from BASE, same complete case count, iq4_xs GET_ROWS ran.
ops_judge() {
    local n m
    n=$(grep -aEo '[0-9]+/[0-9]+ tests passed' "$1" | awk -F'[ /]' '{ if ($1 != $2) bad = 1; s += $1 } END { print bad ? -1 : s + 0 }') || true
    m=$(grep -aEo '[0-9]+/[0-9]+ tests passed' "$2" | awk -F'[ /]' '{ s += $2 } END { print s + 0 }') || true
    (( ${n:-0} >= 1000 && n == m )) || { echo "ops cases new=${n:-0} reference=${m:-0}: $1"; return 8; }
    if [[ -n $(comm -23 <(grep -a 'FAIL' "$1" | sort -u) <(grep -a 'FAIL' "$2" | sort -u)) ]]; then echo "new FAIL lines in $1"; return 8; fi
    grep -aEq 'GET_ROWS\(type=iq4_xs,n=[0-9]+,.*\): .*OK' "$1" || { echo "no GET_ROWS iq4_xs row: $1"; return 8; }
    echo "pass: $(basename "$1") ops cases=$n"
}
# ALLOC SPEC: the Mac gate's unit tests (91 Meta cache, 90 host-mapped weight, 88 uses_prompt).
unit_judge() {
    local t
    for t in test_meta_subgraph_realloc_count test_meta_subgraph_realloc_size test_meta_subgraph_reuse_two_splits \
             test_meta_subgraph_reuse_ten_splits test_meta_subgraph_reuse_same_graph test_meta_host_mapped_weight; do
        grep -aq "^$t PASSED" "$1" || { echo "missing: $t PASSED"; return 8; }
    done
    if grep -aq 'FAIL' "$1" "$2"; then echo 'FAIL line in a unit log'; return 8; fi
    grep -aq 'uses_prompt' "$2" || { echo 'missing: uses_prompt'; return 8; }
}
# LOG PROFILE: served-flag boot log judged by health.py (engagement lines of every kept component incl. p90; K7/K10 compiled out).
health_judge() {
    ENGAGEMENT_DROP='k7 k10' python3 - "$WIN" "$1" "$2" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import health
r = health.judge(open(sys.argv[2], errors='replace').read(), sys.argv[3])
if 'p90' not in r['gating'] or {'k7', 'k10'} & set(r['gating']):
    raise SystemExit(f'gating set {r["gating"]}')
if r['missing'] or r['fatal']:
    raise SystemExit(f'missing {r["missing"]} fatal {r["fatal"]}')
PY
}

selftest() {
    local d rc
    d=$(mktemp -d)
    trap 'rm -rf "$d"' RETURN
    must() { "$@" > /dev/null 2>&1 || { echo "selftest: good input failed: $*"; exit 8; }; }
    mustnt() { if "$@" > /dev/null 2>&1; then echo "selftest: broken input passed: $*"; exit 8; fi; }
    drop() { grep -av "$1" "$2" > "$3" || true; }  # PATTERN IN OUT; an empty OUT is a valid broken input
    gunzip -c "$FIX/fa-CUDA0.log.gz" > "$d/fa"
    gunzip -c "$FIX/ops-CUDA0.txt.gz" > "$d/ops0"
    gunzip -c "$FIX/ops-CUDA1.txt.gz" > "$d/ops1"
    gunzip -c "$FIX/server-j5q-duo.log.gz" > "$d/server"
    must fa_judge "$d/fa"
    sed '0,/FLASH_ATTN_EXT(.*OK/s//&\n  FLASH_ATTN_EXT(x): FAIL/' "$d/fa" > "$d/fa-fail"; mustnt fa_judge "$d/fa-fail"
    sed 's#4089/4089 tests passed#4088/4089 tests passed#' "$d/fa" > "$d/fa-part"; mustnt fa_judge "$d/fa-part"
    drop 'tests passed' "$d/fa" "$d/fa-none"; mustnt fa_judge "$d/fa-none"
    must ops_judge "$d/ops0" "$d/ops1"
    sed '0,/GET_ROWS(.*OK/s/OK/FAIL/' "$d/ops0" > "$d/ops-fail"; mustnt ops_judge "$d/ops-fail" "$d/ops1"
    sed 's#1131/1131 tests passed#900/900 tests passed#' "$d/ops0" > "$d/ops-few"; mustnt ops_judge "$d/ops-few" "$d/ops1"
    drop 'GET_ROWS(type=iq4_xs' "$d/ops0" "$d/ops-noiq4"; mustnt ops_judge "$d/ops-noiq4" "$d/ops1"
    RC=0
    must unit_judge "$FIX/test-alloc.txt" "$FIX/test-speculative.txt"
    drop '^test_meta_host_mapped_weight PASSED' "$FIX/test-alloc.txt" "$d/alloc-drop"; mustnt unit_judge "$d/alloc-drop" "$FIX/test-speculative.txt"
    drop 'uses_prompt' "$FIX/test-speculative.txt" "$d/spec-drop"; mustnt unit_judge "$FIX/test-alloc.txt" "$d/spec-drop"
    must gate_mapped "$FIX/thm-zc.log" 0
    sed '0,/differing=0\//s//differing=1\//' "$FIX/thm-zc.log" > "$d/thm-diff"; mustnt gate_mapped "$d/thm-diff" 0
    mustnt gate_mapped "$FIX/thm-zc.log" 1
    must gate_nopin "$FIX/thm-nopin.log" 0
    drop 'dev=CUDA1 fallback=1 get_rows=CPU' "$FIX/thm-nopin.log" "$d/nopin-drop"; mustnt gate_nopin "$d/nopin-drop" 0
    must health_judge "$d/server" swift15-j5q-duo
    drop 'CUDA_Host_Mapped model buffer' "$d/server" "$d/server-nop90"; mustnt health_judge "$d/server-nop90" swift15-j5q-duo
    { cat "$d/server"; echo 'CUDA error: an illegal memory access was encountered'; } > "$d/server-fatal"; mustnt health_judge "$d/server-fatal" swift15-j5q-duo
    rc=0; python3 "$WIN/health.py" --selftest > /dev/null || rc=$?
    [[ $rc == 0 ]] || { echo 'selftest: health.py --selftest'; exit 8; }
    rc=0; python3 "$WIN/engagement.py" selftest > /dev/null || rc=$?
    [[ $rc == 0 ]] || { echo 'selftest: engagement.py selftest'; exit 8; }
    echo 'bw-fa-s2: PASS selftest (fa/ops/unit/host-mapped/health judges pass recorded outputs and fail 13 broken copies)'
}

preflight() {
    local tool p
    for tool in cmake git python3 nice ccache strings timeout taskset g++ nvidia-smi journalctl curl awk sed comm gunzip patchelf; do
        command -v "$tool" > /dev/null || { echo "missing tool $tool"; return 1; }
    done
    [[ $(git -C "$ROOT" branch --show-current) == "$BRANCH" ]] || { echo "branch is not $BRANCH"; return 1; }
    [[ -z $(git -C "$ROOT" status --porcelain --untracked-files=no) ]] || { echo 'tracked tree is dirty'; return 1; }
    [[ ! -e $W/frozen/$NAME ]] || { echo "frozen/$NAME exists"; return 1; }
    git -C "$ROOT" merge-base --is-ancestor "$FA_SHA" HEAD || { echo "stack11-fa $FA_SHA is not an ancestor"; return 1; }
    # HEAD = stack11-s2-nok + the 93/0001 diff, nothing else outside codex/ (blob ids ignored); a failed git diff must not compare equal
    [[ $(git -C "$ROOT" diff --name-only "$BASE_SHA" "$FA_SHA" | wc -l) == 2 ]] || { echo '93/0001 diff unreadable'; return 1; }
    cmp -s <(git -C "$ROOT" diff "$BASE_SHA" "$FA_SHA" -- . ':!codex' | grep -v '^index ') \
        <(git -C "$ROOT" diff "$NOK_SHA" HEAD -- . ':!codex' | grep -v '^index ') \
        || { echo 'HEAD differs from stack11-s2-nok by more than the 93/0001 diff'; return 1; }
    [[ $(head -n 1 "$W/frozen/stack11-fa/SOURCE") == "$FA_SHA" && $(head -n 1 "$BASE_FROZEN/SOURCE") == "$NOK_SHA" ]] \
        || { echo 'frozen parents are not the expected commits'; return 1; }
    for p in "$CCCL" "$BUILDGATE" "$W/bin/wt" /usr/local/cuda-13.2/bin/nvcc "$SERVE" /mnt/scratch "$BASE_FROZEN/build/bin/test-backend-ops" \
             "$FIX/fa-CUDA0.log.gz" "$FIX/server-j5q-duo.log.gz"; do
        [[ -e $p ]] || { echo "missing $p"; return 1; }
    done
    [[ $(grep -c '_Host_Mapped' "$ROOT/ggml/src/ggml-cuda/ggml-cuda.cu") -gt 0 && -s $ROOT/tests/test-host-mapped.cpp ]] || { echo 'no pkg 90 source'; return 1; }
    [[ ! -e $ROOT/tests/test-topk-batched.cpp && ! -e $ROOT/ggml/src/ggml-rms-norm-scale-pair.h ]] || { echo 'K7/K10 sources present'; return 1; }
    bash -n "$WIN/gates.sh" || { echo 'gates.sh syntax'; return 1; }
    python3 "$WIN/health.py" --parse 18099 > /dev/null || { echo 'serve.sh launch lines'; return 1; }
    return 0
}

# --- stages --------------------------------------------------------------------------------------------------------------------------
build() {
    cd "$ROOT"
    mkdir -p /mnt/scratch/tmp-build
    TMPDIR=/mnt/scratch/tmp-build nice -n 19 "$BUILDGATE" cmake -S . -B build-cuda "${flags[@]}"
    # shellcheck disable=SC2046
    TMPDIR=/mnt/scratch/tmp-build nice -n 19 "$BUILDGATE" cmake --build build-cuda -j4 --target $(targets)
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}

stamps_fa_s2() {
    local so=$ROOT/build-cuda/bin lib n m
    for lib in libggml-cuda.so libggml-base.so libllama.so libllama-common.so; do [[ -s $so/$lib ]]; done
    [[ $(strings "$so/libggml-base.so" | grep -cx "$SHORT") -ge 1 ]]
    [[ $(grep -ac 'CUDA_Host_Mapped' "$so/libggml-cuda.so") -ge 1 ]]
    [[ $(grep -ac 'fattn_kv_max' "$so/libggml-cuda.so") == 0 ]]
    n=$(grep -ac 'rms_norm_scale_pair_fusion: device' "$so/libggml-cuda.so") || true
    m=$(grep -ac 'top_k_batched: engaged' "$so/libggml-cuda.so") || true
    echo "rms_norm_scale_pair_fusion_count_M=$n top_k_batched_count_M=$m (both must be 0)"
    [[ $n == 0 && $m == 0 ]]
    [[ $(git -C "$ROOT" rev-parse HEAD) == "$SHA" && -z $(git -C "$ROOT" status --porcelain --untracked-files=no) ]]
    (cd "$so" && sha256sum round-trace llama-server test-backend-ops test-host-mapped test-alloc ./*.so) > "$OUT/build-sha256.txt"
}

unit_run() {
    local d=$ROOT/build-cuda/bin
    timeout --kill-after=10s 600 env CUDA_VISIBLE_DEVICES='' nice -n 19 "$d/test-alloc" > "$OUT/test-alloc.txt" 2>&1
    timeout --kill-after=10s 600 env CUDA_VISIBLE_DEVICES='' nice -n 19 "$d/test-speculative" > "$OUT/test-speculative.txt" 2>&1
    unit_judge "$OUT/test-alloc.txt" "$OUT/test-speculative.txt"
}
g_unit_fa_s2() { in_dir unit unit_run; }

g_mapped_fa_s2() { in_dir mapped mapped_nok; }

ops_run() {
    local dev who bin rc
    for dev in CUDA0 CUDA1; do
        for who in new ref; do
            if [[ $who == new ]]; then bin=$ROOT/build-cuda/bin/test-backend-ops; else bin=$BASE_FROZEN/build/bin/test-backend-ops; fi
            rc=0
            env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout --kill-after=30s 3600 nice -n 19 "$bin" test -b "$dev" -o "$OPS_COMMON" -j 1 \
                > "$OUT/ops-$dev-$who.txt" 2>&1 || rc=$?
            echo "ops $dev $who rc_M=$rc"
            [[ $rc == 0 ]] || return 8
        done
        ops_judge "$OUT/ops-$dev-new.txt" "$OUT/ops-$dev-ref.txt"
    done
}
g_ops_fa_s2() { in_dir ops ops_run; }

fa_run() {
    local dev rc
    for dev in CUDA0 CUDA1; do
        rc=0
        env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout --kill-after=30s 3600 nice -n 19 "$ROOT/build-cuda/bin/test-backend-ops" test -b "$dev" \
            -o FLASH_ATTN_EXT -j 1 > "$OUT/fa-$dev.log" 2>&1 || rc=$?
        echo "FLASH_ATTN_EXT $dev rc_M=$rc"
        [[ $rc == 0 ]] || return 8
        fa_judge "$OUT/fa-$dev.log"
    done
}
g_fa() { in_dir fa fa_run; }

g_health_fa_s2() {
    local p
    mkdir -p "$OUT/health"
    for p in $PROFILES; do
        env CUDA_VISIBLE_DEVICES="$VISIBLE" ROOT="$ROOT" PORT="$PORT" XID_SINCE="$STARTED" ENGAGEMENT_DROP='k7 k10' \
            python3 "$WIN/health.py" "$p" "$OUT/health/$p"
        health_judge "$OUT/health/$p/server.log" "$p"
    done
}

freeze_fa_s2() {
    [[ $(git -C "$ROOT" rev-parse HEAD) == "$SHA" && -z $(git -C "$ROOT" status --porcelain --untracked-files=no) && ! -e $W/frozen/$NAME ]]
    nice -n 19 "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda --desc "stack11-fa-s2: parents frozen/stack11-fa ($FA_SHA = stack11-base + pkg 93/0001) and frozen/stack11-s2-nok ($NOK_SHA; its commits $BASE_SHA..$NOK_SHA cherry-picked: 87 draft-state reset, 88 spec prompt on demand, 91 Meta per-uid cache tests, 90 CUDA_Host_Mapped token table; K7/K10 dropped); gates: unit tests, host-mapped bitwise, GET_ROWS/TOP_K/norm ops vs stack11-s2-nok, FLASH_ATTN_EXT full on both GPUs, served-flag health boots of all four profiles; CUDA 13.2 120a CCCL343 NCCL OFF"
    [[ $(head -n 1 "$W/frozen/$NAME/SOURCE") == "$SHA" ]]
    {
        printf 'commit %s\nconfigure: cmake -S . -B build-cuda ' "$SHA"
        printf '%q ' "${flags[@]}"
        printf '\nbuild: TMPDIR=/mnt/scratch/tmp-build bin/buildgate cmake --build build-cuda -j4 --target %s\n' "$(targets)"
        printf 'parents: frozen/stack11-fa %s; frozen/stack11-s2-nok %s (cherry-pick %s..%s onto %s)\n' "$FA_SHA" "$NOK_SHA" "$BASE_SHA" "$NOK_SHA" "$FA_SHA"
        printf 'round-trace = copy of llama-server; gates ran in %s\n' "$OUT"
    } > "$W/frozen/$NAME/RECIPE.txt"
}

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

case ${1:-} in
    --selftest) selftest; exit 0 ;;
    --preflight) if preflight; then echo 'bw-fa-s2: PASS preflight'; exit 0; fi; echo 'bw-fa-s2: FAIL preflight'; exit 8 ;;
    '') echo 'usage: OUT | --preflight | --selftest'; exit 8 ;;
esac
OUT=$1
mkdir -p "$OUT"
preflight || { echo 'bw-fa-s2: FAIL preflight'; exit 8; }
SHA=$(git -C "$ROOT" rev-parse HEAD)
SHORT=$(git -C "$ROOT" rev-parse --short=9 HEAD)
STARTED=$(date '+%Y-%m-%d %H:%M:%S')
PORT=${PORT:-$(free_port)}
echo "bw-fa-s2: commit=$SHA visible=$VISIBLE port=$PORT"

fatal 40 cuda-build build
fatal 42 stamps stamps_fa_s2
gate 50 unit-cpu g_unit_fa_s2
gate 51 host-mapped g_mapped_fa_s2
gate 52 ops g_ops_fa_s2
gate 53 fa-ops g_fa
if [[ ${#FAILS[@]} == 0 ]]; then
    gate 60 health-boots g_health_fa_s2
else
    echo 'bw-fa-s2: health boots skipped, an earlier gate failed'
fi
if [[ ${#FAILS[@]} != 0 ]]; then finish_fail; fi
fatal 70 freeze freeze_fa_s2
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], GPU_execution=True, stages=stages,
                                              frozen=['frozen/stack11-fa-s2']), indent=2)+'\n')
PY_RESULT
echo 'bw-fa-s2: PASS; frozen as stack11-fa-s2'
