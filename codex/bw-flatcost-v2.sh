#!/usr/bin/env bash
# Build-window payload (gpuq filler item; never submits anything). Usage: bw-flatcost-v2.sh [--preflight | OUT_DIR]
#
# BW_CPU_ONLY=1 (Mac, bin/bw-mac): CPU build of test-backend-ops (clang), then on the CPU backend only:
#          a. test mode on the q6_K/iq4_xs MUL_MAT eval cases (builds and runs every new case: tails, short rows, broadcast,
#             strided view, permuted operands);
#          b. support mode on the GGML_TEST_FLATCOST case lists: 260 cases for =1, 137 for =q6_k and for =iq4_xs.
# Stage 2  CUDA build (stack6 RECIPE flags, tests on) of test-backend-ops + llama-server, CCCL check, ggml stamp check,
#          mmvq-mma register/stack report, wt freeze flatcost-v2 + RECIPE.txt; base arm work/flatcost-v2/base-bin
#          (this test-backend-ops with RUNPATH frozen/stack6/build/bin).
# Stage 3  Copy the memcheck/perf items into the drafts and run gpuq submit --check only.
#
# Exit codes: 0 all pass | 2 CPU case failure | 8 CPU build or preflight | 9 CUDA build/freeze/CCCL/base arm | 10 item check.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/.." || exit 8
ROOT=$PWD
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=flatcost-v2
BASE_BIN=$W/work/flatcost-v2/base-bin
ITEMS=(flatcost-v2-memcheck flatcost-v2-perf)
log() { echo "bw-$NAME: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    paths=("${BW_BUILDGATE:-$W/bin/buildgate}" "${BW_GATE:-$W/bin/cpugate}")
    if [[ ${BW_CPU_ONLY:-0} != 1 ]]; then
        paths+=("$W/bin/wt" "$CCCL" "$W/frozen/stack6/build/bin/libggml-cuda.so.0.25.3" "$W/work/flatcost-v2/run.sh"
                "$EVAL/orchestration/gpuq/items/drafts" /usr/bin/patchelf /usr/local/cuda-13.2/bin/cuobjdump)
        for item in "${ITEMS[@]}"; do paths+=("$ROOT/codex/$item.toml"); done
    fi
    for p in "${paths[@]}"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    [[ ${BW_CPU_ONLY:-0} != 1 ]] || exit 0
    command -v gpuq >/dev/null || die 8 "preflight: gpuq not on PATH"
    [[ ! -e $W/frozen/$NAME ]] || die 8 "preflight: $W/frozen/$NAME exists already"
    [[ ! -e $BASE_BIN ]] || die 8 "preflight: $BASE_BIN exists already"
    for item in "${ITEMS[@]}"; do
        [[ ! -e $EVAL/orchestration/gpuq/items/drafts/$item.toml ]] || die 8 "preflight: drafts/$item.toml exists already"
    done
    exit 0
fi
if [[ $(uname) == Darwin ]]; then
    OUT=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${1:-$ROOT/codex/bw-results}")
else
    OUT=$(readlink -m "${1:-$ROOT/codex/bw-results}")
fi
mkdir -p "$OUT"

[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "tracked changes present; commit before the build window (wt freeze needs a clean tree)"
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}
CPUGATE=${BW_GATE:-$W/bin/cpugate}
RESULTS=$OUT/RESULTS.txt
: > "$RESULTS"
note() { echo "$*" | tee -a "$RESULTS"; }

# ---------------------------------------------------------------- CPU (Mac): test-backend-ops cases
if [[ ${BW_CPU_ONLY:-0} == 1 ]]; then
    cpu_extra=()
    # BW_CPU_EXTRA is a whitespace-separated list of cmake arguments; no shell expansion.
    read -r -a cpu_extra <<< "${BW_CPU_EXTRA:-}"
    {
        "$BUILDGATE" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF \
            -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF "${cpu_extra[@]}" &&
        "$BUILDGATE" cmake --build build-cpu --target test-backend-ops -j3  # Mac rule: at most 3 build jobs
    } > "$OUT/build-cpu.log" 2>&1 || die 8 "CPU build (see $OUT/build-cpu.log)"
    note "CPU build ok, HEAD $SHA, $(uname -sm), $(c++ --version | head -1)"
    TBO=build-cpu/bin/test-backend-ops
    "$CPUGATE" "$TBO" test -o MUL_MAT -b CPU -p 'type_a=(q6_K|iq4_xs),' --output sql > "$OUT/cpu-eval.sql" 2> "$OUT/cpu-eval.err" ||
        die 2 "CPU eval cases (see $OUT/cpu-eval.err)"
    n=$(grep -c "^INSERT" "$OUT/cpu-eval.sql")
    bad=$(grep "^INSERT" "$OUT/cpu-eval.sql" | grep -vc "'test', '1', '1'" || true)
    [[ $n -ge 146 && $bad -eq 0 ]] || die 2 "CPU eval cases: $n rows (want >= 146), $bad not supported+passed"
    note "CPU eval q6_K|iq4_xs MUL_MAT: $n cases ran and passed (CPU vs CPU: graph build and execution, not numerics)"
    for sel in 1:260 q6_k:137 iq4_xs:137; do
        GGML_TEST_FLATCOST=${sel%%:*} "$CPUGATE" "$TBO" support -o MUL_MAT -b CPU --output sql > "$OUT/cpu-flat-${sel%%:*}.sql" 2>&1 ||
            die 2 "flatcost support list ${sel%%:*}"
        n=$(grep -c "^INSERT" "$OUT/cpu-flat-${sel%%:*}.sql")
        [[ $n == "${sel##*:}" ]] || die 2 "GGML_TEST_FLATCOST=${sel%%:*}: $n cases, want ${sel##*:}"
        note "GGML_TEST_FLATCOST=${sel%%:*}: $n cases"
    done
    note "CPU PASS"
    exit 0
fi

# ---------------------------------------------------------------- stage 2: CUDA build, freeze, base arm
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_stage() {
    export CUDA_HOME=/usr/local/cuda-13.2
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}" || return 1
    "$BUILDGATE" cmake --build build-cuda --target test-backend-ops llama-server -j4 || return 1
    [[ $(git rev-parse HEAD) == "$SHA" ]] || { echo "HEAD moved during the build"; return 1; }
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 1
    local n; n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "CCCL DeviceTopK strings: $n"
    [[ $n == 25 ]] || { echo "expected 25 DeviceTopK strings (CCCL v3.4.3)"; return 1; }
    local short; short=$(git rev-parse --short HEAD)
    strings build-cuda/bin/libggml-base.so | grep -x "$short" >/dev/null || { echo "ggml stamp $short not in libggml-base.so"; return 1; }
    grep -aq 'mmvq_mma: device' build-cuda/bin/libggml-cuda.so || { echo "libggml-cuda.so lacks mmvq-mma"; return 1; }
    # registers/stack of the new kernels (launch bound: 128 threads, 4 blocks/SM -> <= 128 registers); report, not a gate
    /usr/local/cuda-13.2/bin/cuobjdump --dump-resource-usage build-cuda/bin/libggml-cuda.so > "$OUT/resource-usage.txt" 2>&1 || return 1
    python3 - "$OUT/resource-usage.txt" > "$OUT/mmvq-mma-resources.txt" <<'EOF' || return 1
import re, sys
lines = open(sys.argv[1], errors='replace').read().splitlines()
found = 0
for i, l in enumerate(lines):
    if 'mul_mat_vec_q_mma' in l and i + 1 < len(lines):
        found += 1
        print(l.strip()[:160])
        print('   ', lines[i + 1].strip())
if not found:
    sys.exit('no mul_mat_vec_q_mma entries in the resource usage dump')
EOF
    cat "$OUT/mmvq-mma-resources.txt"
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack6 ac7465885 + mmvq-mma (few-column q6_K/iq4_xs on int8 tensor cores) + local knobs/test cases; CUDA 13.2, CCCL343, NCCL OFF, tests on; round-trace = cp of llama-server" || return 1
    {
        printf 'name %s\n' "$NAME"
        printf 'commit %s (base stack6 ac7465885dc9f13de396ccd91b4be5eaffde3730, branch %s)\n' "$SHA" "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
        printf 'worktree %s, build dir build-cuda\n' "$ROOT"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf ' (CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES="")\n'
        printf 'build: bin/buildgate cmake --build build-cuda --target test-backend-ops llama-server -j4\n'
        printf 'round-trace = cp of llama-server (harness arms must not execute a file named llama-server); listed in SHA256SUMS\n'
        printf 'base arm for items: work/flatcost-v2/base-bin/test-backend-ops = this test-backend-ops with RUNPATH frozen/stack6/build/bin\n'
        printf 'DeviceTopK strings %s; no CUDA binary executed; mmvq-mma resources %s\n' "$n" "$OUT/mmvq-mma-resources.txt"
        printf 'ggml stamp %s\n' "$short"
    } > "$W/frozen/$NAME/RECIPE.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE" | cut -d' ' -f1) == "$SHA" ]] || return 1
    mkdir -p "$BASE_BIN" || return 1
    cp "$W/frozen/$NAME/build/bin/test-backend-ops" "$BASE_BIN/" || return 1
    patchelf --set-rpath "$W/frozen/stack6/build/bin" "$BASE_BIN/test-backend-ops" || return 1
    local stray; stray=$(ldd "$BASE_BIN/test-backend-ops" | awk '/libggml/ {print $3}' | grep -v "^$W/frozen/stack6/build/bin/" || true)
    [[ -z $stray ]] || { echo "base arm resolves ggml outside frozen/stack6: $stray"; return 1; }
    (cd "$BASE_BIN" && sha256sum test-backend-ops > SHA256SUMS) || return 1
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL/base arm (see $OUT/cuda.log)"
note "CUDA build frozen at $W/frozen/$NAME; base arm $BASE_BIN"
tee -a "$RESULTS" < "$OUT/mmvq-mma-resources.txt"

# ---------------------------------------------------------------- stage 3: items (check only, never submit)
item_stage() {
    local item dst
    for item in "${ITEMS[@]}"; do
        dst=$EVAL/orchestration/gpuq/items/drafts/$item.toml
        [[ ! -e $dst ]] || { echo "$dst exists; not overwriting"; return 1; }
        cp "codex/$item.toml" "$dst" || return 1
        gpuq submit --check "$dst" || return 1
        echo "item checked, NOT submitted: $dst"
    done
}
item_stage > "$OUT/items.log" 2>&1 || die 10 "item check (see $OUT/items.log)"
tee -a "$RESULTS" < "$OUT/items.log"
log "done"
exit 0
