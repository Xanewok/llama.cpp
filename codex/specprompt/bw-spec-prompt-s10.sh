#!/usr/bin/env bash
# Parent queue only: CUDA build of spec-prompt-s10 (STEP fix 5) with the frozen/stack10 recipe, test-speculative on the CPU
# of this host (n-gram drafters declare the prompt they read), freeze, then render/check the GPU items (never submits).
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GGML_CCACHE=ON
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda-13.2
export TMPDIR=/mnt/scratch/tmp-build
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=spec-prompt-s10
BASE=a588f2a44382a361df3e2a456be0c80ca10dd1d6
# the Mac CPU gate of these exact ggml/tests trees
MAC=$W/work/build-window-1007/spec-prompt-s10-mac-a1
RECIPE=$W/frozen/stack10/RECIPE.txt
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
# frozen/stack10 RECIPE line 3 targets
recipe_targets=(llama-server test-backend-ops test-alloc test-recurrent-state-rollback test-fattn-kv-max-cache test-meta-input-stage)
targets=("${recipe_targets[@]}" test-speculative)
cd "$ROOT"
configure_line() { printf 'configure: cmake -S . -B build-cuda '; printf '%q ' "${flags[@]}"; }
preflight() {
    [[ $(git branch --show-current) == topic/spec-prompt-s10 ]] || return 6
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 6
    git merge-base --is-ancestor "$BASE" HEAD || return 6
    git grep -c 'common_speculative_uses_prompt(spec.get())' HEAD -- tools/server/server-context.cpp > /dev/null || return 6
    [[ $(configure_line) == "$(sed -n 2p "$RECIPE")" ]] || return 6
    [[ $(sed -n 3p "$RECIPE") == "build: bin/buildgate cmake --build build-cuda -j4 --target ${recipe_targets[*]}" ]] || return 6
    [[ ! -e $W/frozen/$NAME && ! -e build-cuda ]] || return 6
    if git show-ref --verify --quiet "refs/tags/frozen/$NAME"; then return 6; fi
    [[ -d $CCCL && -x /usr/local/cuda-13.2/bin/nvcc && -d $TMPDIR && -w $TMPDIR ]] || return 6
    for tool in cmake git python3 nice gpuq ccache strings patchelf taskset timeout; do command -v "$tool" > /dev/null || return 6; done
    for path in "$W/bin/buildgate" "$W/bin/wt" "$W/work/aa-load/ccx01.sh" "$W/frozen/stack10/SOURCE"; do
        [[ -f $path ]] || return 6
    done
    # the Mac gate passed on the same common, tools and tests trees
    grep -aq 'EXIT_CODE=0' "$MAC/mac-run.txt" || return 6
    local mac
    mac=$(sed -n 's/^HEAD=//p' "$MAC/mac-run.txt")
    for d in common tools tests ggml src; do
        [[ $(git rev-parse "$mac:$d") == $(git rev-parse "HEAD:$d") ]] || return 6
    done
    bash -n codex/specprompt/bw-spec-prompt-s10.sh codex/specprompt/postrun_proofs.sh || return 6
    bash codex/specprompt/postrun_proofs.sh --selftest > /dev/null || return 6
    python3 -c "compile(open('codex/specprompt/render_specprompt.py').read(), 'render_specprompt.py', 'exec')" || return 6
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
    printf 'ngram-simple: uses_prompt, draft size 6: OK\nngram-cache: uses_prompt, draft size 8: OK\n' > "$d/good"
    gate_log "$d/good" 0 '^ngram-simple: uses_prompt' '^ngram-cache: uses_prompt' || return 1
    if gate_log "$d/good" 1 '^ngram-simple: uses_prompt'; then return 1; fi
    if gate_log "$d/good" 0 '^ngram-mod: uses_prompt'; then return 1; fi
}
if [[ ${1:-} == --selftest ]]; then
    selftest || { echo 'bw-spec-prompt-s10: FAIL selftest'; exit 8; }
    echo 'bw-spec-prompt-s10: PASS selftest (good log passes; rc 1 and a missing line fail)'; exit 0
fi
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-spec-prompt-s10: FAIL preflight'; exit 6; }
    echo 'bw-spec-prompt-s10: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-spec-prompt-s10 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/specprompt/window/a[0-9]*) ;; *) exit 8 ;; esac
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
    echo "bw-spec-prompt-s10: $name rc=$rc wall_s_M=$((SECONDS-began))"
    [[ $rc == 0 ]] || { echo "bw-spec-prompt-s10: FAIL $name"; case $rc in 3|4|5|6|7|8) exit "$rc" ;; *) exit 8 ;; esac; }
}
build() {
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake -S . -B build-cuda "${flags[@]}" || return 8
    nice -n 19 taskset -c 6-11,18-23 "$W/bin/buildgate" cmake --build build-cuda -j4 --target "${targets[@]}" || return 8
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 8
}
stamps() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" > /dev/null || return 6
    strings build-cuda/bin/libllama-server-impl.so | grep -F 'spec_prompt: engaged' > /dev/null || return 6
    grep -ax 'GGML_CUDA:BOOL=ON' build-cuda/CMakeCache.txt || return 6
    grep -ax 'GGML_CUDA_NCCL:BOOL=OFF' build-cuda/CMakeCache.txt || return 6
    grep -aEx 'CMAKE_CUDA_ARCHITECTURES:(STRING|UNINITIALIZED)=120a' build-cuda/CMakeCache.txt || return 6
    cp build-cuda/CMakeCache.txt "$OUT/CMakeCache.txt" || return 6
    printf 'source_M=%s\nbase_D=%s\nCUDA_D=13.2\nCCCL_D=3.4.3\narch_D=120a\nNCCL_D=OFF\n' "$SHA" "$BASE" > "$OUT/STAMPS.txt"
}
# CPU only (the drafters under test never touch a device). The mutant (an n-gram drafter that does not declare the prompt
# it reads) is built and must fail in the Mac gate (stage 12).
cpu_tests() {
    local rc=0
    env CUDA_VISIBLE_DEVICES='' timeout 600 nice -n 19 build-cuda/bin/test-speculative > "$OUT/test-speculative.log" 2>&1 || rc=$?
    gate_log "$OUT/test-speculative.log" "$rc" '^ngram-simple: uses_prompt' '^ngram-map-k: uses_prompt' '^ngram-map-k4v: uses_prompt' \
        '^ngram-mod: uses_prompt' '^ngram-cache: uses_prompt' || return 8
}
freeze() {
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]] || return 6
    nice -n 19 "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda --desc "STEP fix 5 on stack10 $BASE: speculative prompt copy built only for drafters that read it (general patch + local boot log line); CUDA 13.2 120a CCCL343 NCCL OFF; frozen/stack10 recipe; test-speculative in the build window" || return 6
    [[ $(sed -n '1p' "$W/frozen/$NAME/SOURCE") == "$SHA" ]] || return 6
    {
        printf 'commit %s\n' "$SHA"
        configure_line
        printf '\nbuild: bin/buildgate cmake --build build-cuda -j4 --target %s\n' "${targets[*]}"
        printf 'base: %s (frozen/stack10)\nCPU gate (Mac): %s\nHost CPU gate: test-speculative, logs in %s\ntag: refs/tags/frozen/%s\n' "$BASE" "$MAC" "$OUT" "$NAME"
    } > "$W/frozen/$NAME/RECIPE.txt" || return 6
    cp "$OUT/STAMPS.txt" "$W/frozen/$NAME/STAMPS.txt" || return 6
    git tag "frozen/$NAME" "$SHA" || return 6
}
items() {
    nice -n 19 taskset -c 6-11,18-23 python3 codex/specprompt/render_specprompt.py
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 cpu-test cpu_tests
stage 48 freeze freeze
stage 49 render-and-check items
python3 - "$OUT" "$SHA" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1], 'summary.json').write_text(json.dumps({'source': sys.argv[2], 'frozen': 'spec-prompt-s10', 'gpu_executed': False, 'items_submitted': False}, indent=2)+'\n')
PY
echo 'bw-spec-prompt-s10: PASS'
