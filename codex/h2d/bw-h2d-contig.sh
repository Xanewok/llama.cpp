#!/usr/bin/env bash
# Parent queue only: CUDA build of h2d-contig (STEP fix 3) with the frozen/stack10 recipe, byte identity of the packed upload on
# both GPUs (per rank and shared; the scatter mutant must fail), freeze, then render/check the GPU items (never submits).
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GGML_CCACHE=ON
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda-13.2
export TMPDIR=/mnt/scratch/tmp-build
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=h2d-contig
BASE=a588f2a44382a361df3e2a456be0c80ca10dd1d6
# the Mac CPU gate of these exact ggml/tests trees
MAC=$W/work/build-window-1007/h2d-contig-mac-a3
RECIPE=$W/frozen/stack10/RECIPE.txt
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
# frozen/stack10 RECIPE line 3 targets
targets=(llama-server test-backend-ops test-alloc test-recurrent-state-rollback test-fattn-kv-max-cache test-meta-input-stage)
cd "$ROOT"
configure_line() { printf 'configure: cmake -S . -B build-cuda '; printf '%q ' "${flags[@]}"; }
preflight() {
    [[ $(git branch --show-current) == topic/h2d-contig ]] || return 6
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 6
    git merge-base --is-ancestor "$BASE" HEAD || return 6
    git grep -c 'ggml_cuda_input_scatter(args, max_size, ctx->input_stream)' HEAD -- ggml/src/ggml-cuda/ggml-cuda.cu > /dev/null || return 6
    [[ $(configure_line) == "$(sed -n 2p "$RECIPE")" ]] || return 6
    [[ $(sed -n 3p "$RECIPE") == "build: bin/buildgate cmake --build build-cuda -j4 --target ${targets[*]}" ]] || return 6
    [[ ! -e $W/frozen/$NAME && ! -e build-cuda ]] || return 6
    if git show-ref --verify --quiet "refs/tags/frozen/$NAME"; then return 6; fi
    [[ -d $CCCL && -x /usr/local/cuda-13.2/bin/nvcc && -d $TMPDIR && -w $TMPDIR ]] || return 6
    for tool in cmake git python3 nice gpuq ccache strings patchelf taskset timeout; do command -v "$tool" > /dev/null || return 6; done
    for path in "$W/bin/buildgate" "$W/bin/wt" "$W/work/aa-load/ccx01.sh" "$W/frozen/stack10/SOURCE"; do
        [[ -f $path ]] || return 6
    done
    # the Mac gate passed on the same ggml and tests trees
    grep -aq 'EXIT_CODE=0' "$MAC/mac-run.txt" || return 6
    local mac
    mac=$(sed -n 's/^HEAD=//p' "$MAC/mac-run.txt")
    [[ $(git rev-parse "$mac:ggml") == $(git rev-parse HEAD:ggml) && $(git rev-parse "$mac:tests") == $(git rev-parse HEAD:tests) ]] || return 6
    bash -n codex/h2d/bw-h2d-contig.sh codex/h2d/postrun_proofs.sh || return 6
    bash codex/h2d/postrun_proofs.sh --selftest > /dev/null || return 6
    python3 -c "compile(open('codex/h2d/render_h2d.py').read(), 'render_h2d.py', 'exec')" || return 6
}
# pass: rc 0 and every required line present
gate_log() {  # LOG RC PATTERN...
    local log=$1 rc=$2 p
    shift 2
    [[ $rc == 0 ]] || return 8
    for p in "$@"; do grep -aEq "$p" "$log" || { echo "missing: $p"; return 8; }; done
}
selftest() {
    local d
    d=$(mktemp -d)
    trap 'rm -rf "$d"' RETURN
    printf 'cuda scatter per rank: OK (48 rounds)\ntest-meta-input-stage: PASS\n' > "$d/good"
    gate_log "$d/good" 0 'cuda scatter per rank: OK' 'test-meta-input-stage: PASS' || return 1
    if gate_log "$d/good" 1 'cuda scatter per rank: OK'; then return 1; fi
    if gate_log "$d/good" 0 'cuda scatter shared: OK'; then return 1; fi
}
if [[ ${1:-} == --selftest ]]; then
    selftest || { echo 'bw-h2d-contig: FAIL selftest'; exit 8; }
    echo 'bw-h2d-contig: PASS selftest (good log passes; rc 1 and a missing line fail)'; exit 0
fi
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-h2d-contig: FAIL preflight'; exit 6; }
    echo 'bw-h2d-contig: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-h2d-contig && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/h2d/window/a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-h2d-contig: $name rc=$rc wall_s_M=$((SECONDS-began))"
    [[ $rc == 0 ]] || { echo "bw-h2d-contig: FAIL $name"; case $rc in 3|4|5|6|7|8) exit "$rc" ;; *) exit 8 ;; esac; }
}
build() {
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake -S . -B build-cuda "${flags[@]}" || return 8
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake --build build-cuda -j4 --target "${targets[@]}" || return 8
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 8
}
stamps() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null || return 6
    strings build-cuda/bin/libggml-cuda.so | grep -F 'input_scatter: engaged device=' > /dev/null || return 6
    strings build-cuda/bin/libggml-cuda.so | grep -F 'k_input_scatter' > /dev/null || return 6
    grep -ax 'GGML_CUDA:BOOL=ON' build-cuda/CMakeCache.txt || return 6
    grep -ax 'GGML_CUDA_NCCL:BOOL=OFF' build-cuda/CMakeCache.txt || return 6
    grep -aEx 'CMAKE_CUDA_ARCHITECTURES:(STRING|UNINITIALIZED)=120a' build-cuda/CMakeCache.txt || return 6
    cp build-cuda/CMakeCache.txt "$OUT/CMakeCache.txt" || return 6
    printf 'source_M=%s\nbase_D=%s\nCUDA_D=13.2\nCCCL_D=3.4.3\narch_D=120a\nNCCL_D=OFF\n' "$SHA" "$BASE" > "$OUT/STAMPS.txt"
}
# GPU execution, both devices: Part 2 (overlapping writes: ordered path, slowest link first) and Part 3 (disjoint writes:
# packed copy + scatter, per rank and shared, byte-identical). The mutant shifts every scattered entry by one byte: Part 3 must
# fail while Part 2 (which never scatters) still passes.
gpu_tests() {
    [[ $VISIBLE == GPU-*,GPU-* ]] || return 6
    local t=build-cuda/bin/test-meta-input-stage rc=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" INPUT_STAGE_REQUIRE_CUDA=1 timeout 600 nice -n 19 "$t" > "$OUT/stage.log" 2>&1 || rc=$?
    gate_log "$OUT/stage.log" "$rc" 'cuda input stage: OK' 'cuda scatter per rank: OK' 'cuda scatter shared: OK' \
        'input_scatter: engaged device=0 .*mutant=0' 'input_scatter: engaged device=1 .*mutant=0' \
        'slowest host link first: CUDA1 ' 'test-meta-input-stage: PASS' || return 8
    rc=0
    env CUDA_VISIBLE_DEVICES="$VISIBLE" INPUT_STAGE_REQUIRE_CUDA=1 GGML_CUDA_INPUT_SCATTER_MUTANT=1 timeout 600 nice -n 19 "$t" \
        > "$OUT/stage-mutant.log" 2>&1 || rc=$?
    [[ $rc != 0 ]] || { echo 'FAIL scatter mutant escaped'; return 8; }
    grep -aEq 'cuda scatter per rank: CUDA[01] round [0-9]+ entry [0-9]+ differs' "$OUT/stage-mutant.log" || return 8
    grep -aq 'cuda input stage: OK' "$OUT/stage-mutant.log" || return 8
    grep -aq 'input_scatter: engaged device=0 .*mutant=1' "$OUT/stage-mutant.log" || return 8
}
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    nice -n 19 "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda --desc "STEP fix 3 on stack10 $BASE: staged inputs as one H2D copy + device scatter per upload (general 8893df82c + local log/mutant); CUDA 13.2 120a CCCL343 NCCL OFF; frozen/stack10 recipe; byte identity on both GPUs in the build window" || return 6
    [[ $(sed -n '1p' "$W/frozen/$NAME/SOURCE") == "$SHA" ]] || return 6
    {
        printf 'commit %s\n' "$SHA"
        configure_line
        printf '\nbuild: bin/buildgate cmake --build build-cuda -j4 --target %s\n' "${targets[*]}"
        printf 'base: %s (frozen/stack10)\nCPU gate: %s\nGPU gate: test-meta-input-stage on both GPUs + scatter mutant, logs in %s\ntag: refs/tags/frozen/%s\n' "$BASE" "$MAC" "$OUT" "$NAME"
    } > "$W/frozen/$NAME/RECIPE.txt" || return 6
    cp "$OUT/STAMPS.txt" "$W/frozen/$NAME/STAMPS.txt" || return 6
    git tag "frozen/$NAME" "$SHA" || return 6
}
items() {
    nice -n 19 taskset -c 6-11,18-23 python3 codex/h2d/render_h2d.py
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 gpu-identity gpu_tests
stage 48 freeze freeze
stage 49 render-and-check items
python3 - "$OUT" "$SHA" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1], 'summary.json').write_text(json.dumps({'source': sys.argv[2], 'frozen': 'h2d-contig', 'gpu_executed': True, 'items_submitted': False}, indent=2)+'\n')
PY
echo 'bw-h2d-contig: PASS'
