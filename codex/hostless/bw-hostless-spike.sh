#!/usr/bin/env bash
# Parent queue only: x86 np2 hostless identity cells (L1/L2/L3 vs L0), CUDA build of the spike without visible GPUs (the
# frozen/chain-m2-s10 recipe), CPU test-llama-archs qwen35, freeze frozen/hostless-spike, then render/check the A/B and trace
# items (never submits).
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GGML_CCACHE=ON
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda-13.2
export TMPDIR=/mnt/scratch/tmp-build
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
NAME=hostless-spike
BASE=eedee42c50f928e48a28a030a73238e4e25efa64
# the last source commit; commits after it may touch codex/ only (this window and its items live there)
EXPECT=6c0646065e1c4ec462dbc24a2aee8344e75a8f97
RECIPE=$W/frozen/chain-m2-s10/RECIPE.txt
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
# frozen/chain-m2-s10 RECIPE line 3 targets
targets=(llama-server test-backend-ops test-alloc test-recurrent-state-rollback test-fattn-kv-max-cache test-meta-input-stage test-llama-archs)
cd "$ROOT"
configure_line() { printf 'configure: cmake -S . -B build-cuda '; printf '%q ' "${flags[@]}"; }
preflight() {
    [[ $(git branch --show-current) == topic/hostless-spike ]] || return 6
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 6
    git merge-base --is-ancestor "$EXPECT" HEAD || return 6
    git diff --quiet "$EXPECT" HEAD -- . ':(exclude)codex' || return 6
    git merge-base --is-ancestor "$BASE" HEAD || return 6
    git grep -c 'LLAMA_HOSTLESS_VERIFY' HEAD -- src/llama-context.cpp > /dev/null || return 6
    [[ $(configure_line) == "$(sed -n 2p "$RECIPE")" ]] || return 6
    [[ $(sed -n 3p "$RECIPE") == "build: bin/buildgate cmake --build build-cuda -j4 --target ${targets[*]}" ]] || return 6
    [[ ! -e $W/frozen/$NAME && ! -e build-cuda && ! -e build-cpu ]] || return 6
    if git show-ref --verify --quiet "refs/tags/frozen/$NAME"; then return 6; fi
    [[ -d $CCCL && -x /usr/local/cuda-13.2/bin/nvcc && -d $TMPDIR && -w $TMPDIR && -f $MODEL ]] || return 6
    for tool in cmake git python3 nice gpuq ccache strings patchelf taskset; do command -v "$tool" > /dev/null || return 6; done
    for path in "$W/bin/buildgate" "$W/bin/cpugate" "$W/bin/wt" "$W/work/aa-load/ccx01.sh" "$W/frozen/chain-m2-s10/SOURCE"; do
        [[ -f $path ]] || return 6
    done
    bash -n codex/hostless/bw-hostless-spike.sh codex/hostless/cells.sh codex/hostless/trace_hostless.sh \
        codex/hostless/postrun_proofs.sh || return 6
    bash codex/hostless/postrun_proofs.sh --selftest > /dev/null || return 6
    python3 - <<'PY' || return 6
from pathlib import Path
for p in Path('codex/hostless').glob('*.py'):
    compile(p.read_text(), str(p), 'exec')
PY
}
# pass: rc 0, at least one OK line and no FAIL line
gate_test_log() {
    [[ $2 == 0 ]] || return 8
    grep -aq 'OK' "$1" || return 8
    if grep -aq 'FAIL' "$1"; then return 8; fi
}
selftest() {
    local d
    d=$(mktemp -d)
    trap 'rm -rf "$d"' RETURN
    printf 'qwen35: \033[1;32mOK\033[0m\n' > "$d/good"
    printf 'qwen35: \033[1;31mFAIL\033[0m\n' > "$d/fail"
    printf 'main: nothing ran\n' > "$d/missing"
    cat "$d/good" "$d/fail" > "$d/mixed"
    gate_test_log "$d/good" 0 || return 1
    if gate_test_log "$d/good" 1; then return 1; fi
    for bad in fail missing mixed; do
        if gate_test_log "$d/$bad" 0; then return 1; fi
    done
}
if [[ ${1:-} == --selftest ]]; then
    selftest || { echo 'bw-hostless-spike: FAIL selftest'; exit 8; }
    echo 'bw-hostless-spike: PASS selftest (good log passes; rc 1, FAIL, missing and mixed logs fail)'; exit 0
fi
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-hostless-spike: FAIL preflight'; exit 6; }
    echo 'bw-hostless-spike: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-hostless-spike && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/hostless/window/a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
stage() {
    local code=$1 name=$2 rc=0 began=$SECONDS
    shift 2
    (set -e; "$@") > "$OUT/$name.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-began))" >> "$OUT/stages.tsv"
    echo "bw-hostless-spike: $name rc=$rc wall_s_M=$((SECONDS-began))"
    [[ $rc == 0 ]] || { echo "bw-hostless-spike: FAIL $name"; case $rc in 3|4|5|6|7|8) exit "$rc" ;; *) exit 8 ;; esac; }
}
# x86 counterpart of the Mac cells: every level vs L0 at np2 (plain, HOST_DEFER, Meta(CPU,CPU)), floor, slotshift mutant;
# bitwise tokens/counters/draft ids, |dq| <= 2e-7, engagement strings and LLAMA_HOSTLESS_VERIFY
cpu_gate() {
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake -S codex/stack7-test -B build-cpu -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
        -DFETCHCONTENT_UPDATES_DISCONNECTED=ON || return 8
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake --build build-cpu --target bw-np2-driver -j4 || return 8
    local rc=0
    HL_GATE=$W/bin/cpugate taskset -c 6-11,18-23 bash codex/hostless/cells.sh "$ROOT/build-cpu/bin/bw-np2-driver" "$MODEL" \
        "$OUT/cpu-gate" all || rc=$?
    [[ $rc == 75 ]] && return 4
    [[ $rc == 0 ]] || return 8
    python3 codex/hostless/judge.py "$OUT/cpu-gate" || return 8
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
}
build() {
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake -S . -B build-cuda "${flags[@]}" || return 8
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake --build build-cuda -j4 --target "${targets[@]}" || return 8
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 8
}
stamps() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null || return 6
    strings build-cuda/bin/libllama-common.so | grep -F -- 'MTP chain: k=%d seqs=%d' > /dev/null || return 6
    strings build-cuda/bin/libllama-common.so | grep -F -- 'hostless: chain drafts deferred k=%d seqs=%d' > /dev/null || return 6
    strings build-cuda/bin/libllama.so | grep -F -- 'hostless: level=%d mutant=%s' > /dev/null || return 6
    strings build-cuda/bin/libllama.so | grep -F -- 'hostless verify: MISMATCH' > /dev/null || return 6
    strings build-cuda/bin/libllama-server-impl.so | grep -F -- 'hostless: output lag engaged' > /dev/null || return 6
    grep -ax 'GGML_CUDA:BOOL=ON' build-cuda/CMakeCache.txt || return 6
    grep -ax 'GGML_CUDA_NCCL:BOOL=OFF' build-cuda/CMakeCache.txt || return 6
    grep -aEx 'CMAKE_CUDA_ARCHITECTURES:(STRING|UNINITIALIZED)=120a' build-cuda/CMakeCache.txt || return 6
    cp build-cuda/CMakeCache.txt "$OUT/CMakeCache.txt" || return 6
    printf 'source_M=%s\nbase_D=%s\nCUDA_D=13.2\nCCCL_D=3.4.3\narch_D=120a\nNCCL_D=OFF\n' "$SHA" "$BASE" > "$OUT/STAMPS.txt"
}
# CPU device only: tiny random qwen35 model (L0: the spike hooks must not change the plain path)
cpu_tests() {
    local s rc
    for s in 1 2 3; do
        rc=0
        nice -n 19 taskset -c 6-11,18-23 build-cuda/bin/test-llama-archs -a qwen35 -b CPU -s "$s" > "$OUT/test-llama-archs-s$s.log" 2>&1 || rc=$?
        printf '%s\t%s\n' "$s" "$rc" >> "$OUT/test-llama-archs.tsv"
        gate_test_log "$OUT/test-llama-archs-s$s.log" "$rc" || return 8
    done
}
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    nice -n 19 "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda --desc "hostless spike on chain-m2-s10 $BASE: LLAMA_HOSTLESS=1 resident token-embedding table, 2 device-fed V draft ids, 3 one-round output lag; CUDA 13.2 120a CCCL343 NCCL OFF; frozen/chain-m2-s10 recipe; no GPU execution in build window" || return 6
    [[ $(sed -n '1p' "$W/frozen/$NAME/SOURCE") == "$SHA" ]] || return 6
    {
        printf 'commit %s\n' "$SHA"
        configure_line
        printf '\nbuild: bin/buildgate cmake --build build-cuda -j4 --target %s\n' "${targets[*]}"
        printf 'base: %s (frozen/chain-m2-s10)\nCPU gates: hostless np2 cells (x86) and test-llama-archs -a qwen35 -b CPU -s 1..3, logs in %s\nGPU execution: none\ntag: refs/tags/frozen/%s\n' "$BASE" "$OUT" "$NAME"
    } > "$W/frozen/$NAME/RECIPE.txt" || return 6
    cp "$OUT/STAMPS.txt" "$W/frozen/$NAME/STAMPS.txt" || return 6
    git tag "frozen/$NAME" "$SHA" || return 6
}
items() {
    nice -n 19 taskset -c 6-11,18-23 python3 codex/hostless/render_hostless.py
}
stage 40 cpu-gate cpu_gate
stage 41 cuda-build build
stage 42 cuda-stamps stamps
stage 43 cpu-tests cpu_tests
stage 48 freeze freeze
stage 49 render-and-check items
python3 - "$OUT" "$SHA" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1], 'summary.json').write_text(json.dumps({'source': sys.argv[2], 'frozen': 'hostless-spike', 'gpu_executed': False, 'items_submitted': False}, indent=2)+'\n')
PY
