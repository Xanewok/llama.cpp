#!/usr/bin/env bash
# CUDA window for the bounded-scratch q8_0 K/V flash attention (topic/fa-q8-bounded on top of the pkg54-fixed 2efa6f8dc).
# Runs after bw-s10c-fafix: it needs that window's frozen/stack10-cand2 as the base, which preflight does not require.
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
# the gpuq daemon PATH has no CUDA tools (cuobjdump, nvcc)
export PATH="$HOME/.local/bin:$PATH:/usr/local/cuda-13.2/bin" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BASE=2efa6f8dc
S9_LIB=$W/frozen/stack9/build/bin
CAND_LIB=$W/frozen/stack10-cand/build/bin
B2=$W/frozen/stack10-cand2
MODEL=$HOME/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf
DRAFT_VOCAB=$HOME/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json
preflight() {
    [[ $(git branch --show-current) == topic/fa-q8-bounded ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor "$BASE" HEAD || return 1
    local changed f
    mapfile -t changed < <(git diff --name-only "$BASE" HEAD)
    for f in "${changed[@]}"; do
        case $f in codex/*|ggml/src/ggml-cuda/fattn-common.cuh|ggml/src/ggml-cuda/fattn.cu|tests/test-backend-ops.cpp) ;; *) return 1 ;; esac
    done
    for p in "$CCCL" "$W/bin/buildgate" "$W/bin/wt" "$W/frozen/stack9/SOURCE" "$W/frozen/stack10-cand/SOURCE" \
             "$S9_LIB/libggml-cuda.so" "$CAND_LIB/libggml-cuda.so" /usr/local/cuda-13.2/bin/nvcc /usr/local/cuda-13.2/bin/cuobjdump \
             /usr/bin/g++ "$MODEL" "$DRAFT_VOCAB" codex/sass_gate.py codex/fa-window.cpp codex/fa_window_perf.py \
             codex/render_fa_q8_bounded_items.py codex/items/fa-q8-bounded-depth-j5q.toml.in; do
        [[ -e $p ]] || return 1
    done
    [[ ! -e $W/frozen/fa-q8-bounded ]] || return 1
    [[ $(sed -n '1p' "$W/frozen/stack9/SOURCE") == cd776e104dbd516c7b74506992d2e8efd7600155 ]] || return 1
    python3 - <<'PY_GATE' || return 1
from pathlib import Path
for tested in Path('codex/parser-harness/harness').glob('*.py'):
    assert tested.read_bytes() == (Path.home()/'repos/qwen38-eval/harness'/tested.name).read_bytes(), 'parent parser changed since Mac replay'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-fa-q8-bounded: FAIL preflight'; exit 8; }
    echo 'bw-fa-q8-bounded: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-fa-q8-bounded && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/fa-q8-bounded-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
# the base is the pkg54-fixed freeze of bw-s10c-fafix, check it before compiling for 3 minutes
[[ -e $B2/SOURCE && -e $B2/build/bin/libggml-cuda.so && -x $B2/build/bin/test-backend-ops ]] || { echo 'bw-fa-q8-bounded: FAIL no frozen/stack10-cand2 (bw-s10c-fafix did not freeze)'; exit 8; }
[[ $(git rev-parse "$BASE") == $(git merge-base "$BASE" "$(sed -n '1p' "$B2/SOURCE")") ]] || { echo 'bw-fa-q8-bounded: FAIL stack10-cand2 does not contain the base'; exit 8; }
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
FAIL_RC=8
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-fa-q8-bounded: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    if [[ $rc != 0 ]]; then
        echo "bw-fa-q8-bounded: FAIL $name"
        tail -n 40 "$OUT/$name.log" || true
        exit "$FAIL_RC"
    fi
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target llama-server test-backend-ops -j4
    grep -F 'set(CMAKE_CXX_COMPILER_ID "GNU")' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
    grep -E 'set\(CMAKE_CXX_COMPILER_VERSION "13\.' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
# Existing instances must be unchanged: the verify kernels against stack9 and the pkg54-fixed base, every other kernel against
# the base; the only difference is the fixup kernel (new template parameter). The control feeds stack10-cand (the measured
# verify regression) through the same gate and requires it to fail.
sass() {
    python3 codex/sass_gate.py "$OUT/sass" "$S9_LIB/libggml-cuda.so" "$B2/build/bin/libggml-cuda.so" build-cuda/bin/libggml-cuda.so \
        --control "$CAND_LIB/libggml-cuda.so"
}
stamps() {
    local n m
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    m=$(strings "$B2/build/bin/libggml-cuda.so" | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n base=$m"
    [[ $n == "$m" && $n -gt 0 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
all_passed() {
    local n m
    read -r n m < <(sed -n 's#.*[^0-9]\([0-9][0-9]*\)/\([0-9][0-9]*\) tests passed.*#\1 \2#p' "$1" | tail -n 1) || true
    echo "$1: passed=${n:-none} total=${m:-none}"
    [[ -n ${n:-} && ${n:-0} == "${m:-x}" && $n -gt 0 ]]
}
completed() { grep -aEq '[0-9]+/[0-9]+ tests passed' "$1"; }
# Failing case of a completed run: "N/M tests passed" with N < M and a FAIL on a FLASH_ATTN_EXT case. test-backend-ops colours
# FAIL (ESC[1;31mFAIL), so the case line is matched as OP(...) followed by FAIL, not ": FAIL".
mutant_detected() { # LOG
    local n m
    grep -aq 'FLASH_ATTN_EXT(.*FAIL' "$1" || return 1
    read -r n m < <(sed -n 's#.*[^0-9]\([0-9][0-9]*\)/\([0-9][0-9]*\) tests passed.*#\1 \2#p' "$1" | tail -n 1) || true
    [[ -n ${n:-} && -n ${m:-} && $n -lt $m ]]
}
# Failing-case identity: the OP(vars) text only; the ERR/NMSE numbers differ between runs and builds.
fail_ids() { sed 's/\x1b\[[0-9;]*m//g' "$1" | sed -E 's/^.*(FLASH_ATTN_EXT\([^)]*\)).*$/\1/' | LC_ALL=C sort -u; }
# Every FLASH_ATTN_EXT case must pass. A failing case is tolerated only if the pkg54-fixed base build fails the same case on
# this device (a pre-existing failure); a case the base does not have (the new window cases) is never tolerated.
fa_full() {
    local dev=$1 base=$B2/build/bin/test-backend-ops
    if env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 2700 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -j 1 > "$OUT/fa-$dev.log" 2>&1; then
        all_passed "$OUT/fa-$dev.log"
        return 0
    fi
    echo "fa-$dev: failing cases, comparing with the base build"
    completed "$OUT/fa-$dev.log" || { echo "FAIL fa-$dev: run did not complete (no tests-passed summary: crash or timeout)"; return 1; }
    grep -a 'FAIL' "$OUT/fa-$dev.log" > "$OUT/fa-$dev.fail.raw" || { echo "FAIL fa-$dev: no failing case line (crash or timeout)"; return 1; }
    fail_ids "$OUT/fa-$dev.fail.raw" > "$OUT/fa-$dev.fail"
    env CUDA_VISIBLE_DEVICES="$VISIBLE" timeout 2700 nice -n 19 "$base" test -b "$dev" -o FLASH_ATTN_EXT -j 1 > "$OUT/fa-base-$dev.log" 2>&1 || true
    grep -a 'FAIL' "$OUT/fa-base-$dev.log" > "$OUT/fa-base-$dev.fail.raw" || true
    fail_ids "$OUT/fa-base-$dev.fail.raw" > "$OUT/fa-base-$dev.fail"
    LC_ALL=C comm -23 "$OUT/fa-$dev.fail" "$OUT/fa-base-$dev.fail" > "$OUT/fa-$dev.new-failures"
    cat "$OUT/fa-$dev.new-failures"
    [[ ! -s $OUT/fa-$dev.new-failures ]]
}
# The window cases of the new test code (q8_0 K/V, long views); the CPU backend is the reference (NMSE 5e-4).
WINDOW_CASES='hsk=(128|256),hsv=(128|256),.*kv=(8192|8448|16384|20113|32768|90112),.*type_K=q8_0,type_V=q8_0'
wcases() { # DEV TAG [ENV...]: runs the window cases, the log must be passed 100 %
    local dev=$1 tag=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 2700 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p "$WINDOW_CASES" -j 1 > "$OUT/win-$tag-$dev.log" 2>&1
}
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev n
    for dev in CUDA0 CUDA1; do
        fa_full "$dev"
        # 8192 (default): several windows at 32768 and 90112; 1024: the smallest window, 8 windows at kv 8192, 2 at 1280..
        wcases "$dev" w8192 GGML_CUDA_FA_KV_WINDOW_LOG=1
        all_passed "$OUT/win-w8192-$dev.log"
        n=$(grep -ac 'fattn_kv_window: window 1/' "$OUT/win-w8192-$dev.log" || true); echo "$dev engaged at 8192: $n second-window logs"
        [[ $n -gt 0 ]]
        wcases "$dev" w1024 GGML_CUDA_FA_KV_WINDOW=1024 GGML_CUDA_FA_KV_WINDOW_LOG=1
        all_passed "$OUT/win-w1024-$dev.log"
        wcases "$dev" w16384 GGML_CUDA_FA_KV_WINDOW=16384
        all_passed "$OUT/win-w16384-$dev.log"
        # window 0: the one-pass conversion, no window log (the control for the engagement check above)
        wcases "$dev" off GGML_CUDA_FA_KV_WINDOW=0 GGML_CUDA_FA_KV_WINDOW_LOG=1
        all_passed "$OUT/win-off-$dev.log"
        n=$(grep -ac 'fattn_kv_window:' "$OUT/win-off-$dev.log" || true); echo "$dev window off: $n window logs"
        [[ $n == 0 ]]
    done
}
# The same cases with each device or host mutant must FAIL against the CPU (the gate sees a wrong merge).
mutants() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local m seen=0
    for m in 1 2 3 4 5; do
        if wcases CUDA0 "mut$m" GGML_CUDA_FA_KV_WINDOW=1024 GGML_CUDA_FA_KV_WINDOW_MUTANT=$m; then
            echo "mutant $m: test-backend-ops PASSED (control blind)"
        elif mutant_detected "$OUT/win-mut$m-CUDA0.log"; then
            echo "mutant $m: test-backend-ops failed as required ($(grep -ac 'FAIL' "$OUT/win-mut$m-CUDA0.log") failing lines)"
            seen=$((seen+1))
        else
            echo "mutant $m: nonzero rc without a completed run with a failing case (crash or timeout; not a detection)"
        fi
    done
    [[ $seen == 5 ]]
}
# fa-window: windowed against one-pass outputs on the same device (raw f32 bytes, public ggml API only). The windows change the
# order of the f32 sums and the f16 rounding of P (different running maxima, different block splits), so equality is not
# expected beyond a single window; the gate is NMSE <= 1e-5 (G; the CPU gate of test-backend-ops is 5e-4), single window bitwise.
nmse_max() { sed -n 's/.* nmse=\([0-9.e+-]*\) .*/\1/p' "$1" | sort -g | tail -n 1; }
nmse_ok() { # FILE LIMIT
    python3 - "$1" "$2" <<'PY_NMSE'
import re, sys
rows = [l for l in open(sys.argv[1]) if ' nmse=' in l]
assert len(rows) >= 10, f'{len(rows)} cases'
worst = max(float(re.search(r' nmse=(\S+)', l).group(1)) for l in rows)
assert all('nonfinite=0' in l for l in rows), 'non-finite output'
print(f'cases={len(rows)} max_nmse={worst:.3e} limit={sys.argv[2]}')
sys.exit(0 if worst <= float(sys.argv[2]) else 1)
PY_NMSE
}
nmse() {
    [[ $VISIBLE == GPU-* ]] || return 1
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -Iggml/include codex/fa-window.cpp -L build-cuda/bin -lggml -lggml-base -o "$OUT/fa-window"
    local dev w m f want got
    for dev in CUDA0 CUDA1; do
        mkdir -p "$OUT/ref-$dev"
        env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$ROOT/build-cuda/bin" GGML_CUDA_FA_KV_WINDOW=0 timeout 1800 "$OUT/fa-window" "$dev" save "$OUT/ref-$dev" > "$OUT/win-save-$dev.txt"
        for w in 8192 16384 4096 1024; do
            f=$OUT/win-cmp-$dev-w$w.txt
            env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$ROOT/build-cuda/bin" GGML_CUDA_FA_KV_WINDOW=$w timeout 1800 "$OUT/fa-window" "$dev" cmp "$OUT/ref-$dev" > "$f"
            echo "$dev window $w: bitwise cases=$(grep -c 'bitwise=1' "$f") of $(grep -c nmse= "$f")"
            nmse_ok "$f" 1e-5
        done
        # one window (kv 8192 at window 8192) takes the unwindowed path: bytes must be the one-pass bytes
        # expected = number of kv=8192 cases of the case list (fa-window.cpp cases(): exactly 1, read back from the save run so a
        # list change is followed); every such row of the cmp run must be bitwise
        want=$(grep -ac ',kv=8192,.* saved nonfinite=' "$OUT/win-save-$dev.txt" || true)
        got=$(grep -ac ',kv=8192,.*bitwise=1' "$OUT/win-cmp-$dev-w8192.txt" || true)
        echo "$dev kv=8192 bitwise rows: $got of $want expected"
        [[ $want -ge 1 && $got == "$want" ]] || { echo "FAIL $dev single window not bitwise"; return 1; }
        # the other kv sizes (several windows) must have engaged the merge: not bitwise-identical ones exist (informational)
        echo "$dev window 8192 not bitwise: $(grep -c 'bitwise=0' "$OUT/win-cmp-$dev-w8192.txt") cases"
        for m in 1 2 3 4 5; do
            f=$OUT/win-cmp-$dev-mut$m.txt
            env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$ROOT/build-cuda/bin" GGML_CUDA_FA_KV_WINDOW=1024 GGML_CUDA_FA_KV_WINDOW_MUTANT=$m timeout 1800 "$OUT/fa-window" "$dev" cmp "$OUT/ref-$dev" > "$f" || true
            echo "$dev mutant $m: max nmse $(nmse_max "$f") (must exceed 1e-5)"
            if nmse_ok "$f" 1e-5; then echo "FAIL $dev mutant $m not seen by the NMSE gate"; return 1; fi
        done
    done
}
freeze() {
    nice -n 19 "$W/bin/wt" freeze "$ROOT" fa-q8-bounded --build build-cuda --desc 'stack10-cand2 + q8_0 K/V f16 scratch converted in 8192-cell windows merged by log-sum-exp (local commit: window switch, mutants); CUDA gates passed'
    [[ $(sed -n '1p' "$W/frozen/fa-q8-bounded/SOURCE") == "$SHA" ]]
}
items() { nice -n 19 python3 codex/render_fa_q8_bounded_items.py --out "$ROOT/codex/items/rendered"; }
# Reserve of the served J5q np2 shape (2 x 102400, q8_0 KV, f16 MTP draft KV): boot to the first listening state, read the
# sched_reserve lines, stop that PID. Main and draft "compute buffer size" lines per arm and -ub.
boot_reserve() { # LIBDIR TAG UB
    local lib=$1 tag=$2 ub=$3 log=$OUT/reserve-$2-ub$3.log pid i
    env CUDA_VISIBLE_DEVICES="$VISIBLE" LD_LIBRARY_PATH="$lib" GGML_SCHED_INPUT_STAGING=1 LLAMA_ARG_KV_PAD=4096 LLAMA_MTP_CHAIN=1 \
        nice -n 19 "$lib/round-trace" -m "$MODEL" -c 204800 -np 2 -fa on -ctk q8_0 -ctv q8_0 --spec-draft-type-k f16 --spec-draft-type-v f16 \
        -sm tensor --tensor-split 50,50 -ngl 99 -b 2048 -ub "$ub" --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-n-min 0 \
        --spec-draft-p-min 0 --spec-draft-sampling probabilistic --spec-draft-backend-sampling --backend-sampling \
        --spec-draft-vocab "$DRAFT_VOCAB" --cache-ram 0 --metrics -t 8 -tb 8 --host 127.0.0.1 --port "${GPUQ_PORT:?}" > "$log" 2>&1 &
    pid=$!
    for ((i = 0; i < 100; i++)); do
        grep -aq 'server is listening\|all slots are idle' "$log" && break
        kill -0 "$pid" 2> /dev/null || break
        sleep 3
    done
    local up=0
    grep -aq 'server is listening\|all slots are idle' "$log" && up=1
    echo "$up" > "$OUT/reserve-$tag-ub$ub.up"
    kill -TERM "$pid" 2> /dev/null || true
    for ((i = 0; i < 30; i++)); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
    kill -KILL "$pid" 2> /dev/null || true
    wait "$pid" 2> /dev/null || true
    echo "$tag -ub $ub: reached listening=$up"
}
reserve() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local ub
    for ub in 128 512; do
        boot_reserve "$B2/build/bin" base "$ub"
        boot_reserve "$ROOT/build-cuda/bin" bnd "$ub"
    done
    python3 - "$OUT" <<'PY_RESERVE'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
res, bad = {}, []
for ub in (128, 512):
    for tag in ('base', 'bnd'):
        text = (out/f'reserve-{tag}-ub{ub}.log').read_text(errors='replace')
        up = (out/f'reserve-{tag}-ub{ub}.up').read_text().strip() == '1'
        # the base may not fit at -ub 512 (np2 duo has ~450-560 MiB free per GPU): its reserve lines still come before the OOM
        if not up and not (tag == 'base' and ub == 512):
            bad.append(f'{tag} ub{ub}: server did not reach listening')
        lines = [m.group(0) for m in re.finditer(r'sched_reserve:\s+\S+ compute buffer size =\s+[0-9.]+ MiB', text)]
        dev = [float(re.search(r'= +([0-9.]+)', l).group(1)) for l in lines if 'CUDA_Host' not in l]
        res[f'{tag}-ub{ub}'] = dict(lines=lines, device_mib=dev)
        print(f'{tag} -ub {ub}: ' + ' | '.join(l.split('sched_reserve:')[1].strip() for l in lines))
        if up and len(dev) < 2:
            bad.append(f'{tag} ub{ub}: fewer than two device compute buffers (main, draft)')
    b, n = res[f'base-ub{ub}']['device_mib'], res[f'bnd-ub{ub}']['device_mib']
    if not (b and n):
        bad.append(f'ub{ub}: no device compute buffer line')
    else:
        print(f'-ub {ub}: main {b[0]:.1f} -> {n[0]:.1f} MiB (D expected <= ~100), draft {b[1] if len(b) > 1 else None} -> {n[1] if len(n) > 1 else None}')
        if not n[0] <= 0.25*b[0]:
            bad.append(f'ub{ub}: main buffer {n[0]} MiB is not <= 25 % of the base {b[0]} MiB')
(out/'reserve.json').write_text(json.dumps(res, indent=1))
for b in bad:
    print('RESERVE GATE VIOLATION:', b)
sys.exit(1 if bad else 0)
PY_RESERVE
}
PERF_CASES='hsk=(128|256),.*kv=(8192|32768|90112),.*type_K=q8_0,type_V=q8_0'
# Window-size ABBA (palindromic order), one-pass conversion at both ends as the A/A reference; 8 runs per device.
perf() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev w i n
    for dev in CUDA0 CUDA1; do
        i=0
        for w in 0 8192 16384 4096 4096 16384 8192 0; do
            env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_FA_KV_WINDOW=$w timeout 1500 nice -n 19 build-cuda/bin/test-backend-ops perf -b "$dev" -o FLASH_ATTN_EXT -p "$PERF_CASES" > "$OUT/perf-$dev-0$i-w$w.log" 2>&1
            n=$(grep -ac 'FLASH_ATTN_EXT(' "$OUT/perf-$dev-0$i-w$w.log" || true); echo "$dev run $i window $w: $n cases"
            [[ $n -ge 20 ]]
            i=$((i+1))
        done
    done
    python3 codex/fa_window_perf.py "$OUT" CUDA0 CUDA1
}
stage 40 cuda-build build
FAIL_RC=1 stage 41 sass-gate sass
FAIL_RC=8
stage 42 cuda-stamps stamps
stage 43 cuda-fa-ops ops
stage 44 window-mutants-cpu-gate mutants
stage 45 window-nmse-gpu-vs-gpu nmse
stage 46 freeze-fa-q8-bounded freeze
stage 47 render-and-check-items items
stage 48 reserve-boots reserve
stage 49 perf-abba perf
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True,items='codex/items/rendered; checked, NOT submitted'),indent=2)+'\n')
PY_RESULT
echo 'bw-fa-q8-bounded: PASS; frozen and item checked, NOT submitted'
