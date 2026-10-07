#!/usr/bin/env bash
# Usage: bw-spec-tf.sh [OUT_DIR | --preflight]. Run only as a gpuq build-window filler.
# Estimate G: 10800 s total. Manifest budget D: 16200 s (1.5 x estimate). RAM waits can exhaust this budget.
# Estimate G by stage: CPU build 600 s, carry 300 s, matrix 3600 s, controls 2400 s,
# slice 1200 s, deterministic rerun 300 s, CUDA build 1200 s, freeze/item/preflight 1200 s.
#
# Pass criteria and payload exit codes:
#  0: All checks pass. The GPU item is checked, never submitted.
#  6: Preflight inputs, tools, branch/ancestor, clean source, or absent spec-tf-v2 check fails.
#  7: Interrupted by SIGINT/SIGTERM. The wrapper maps this to deterministic failure.
#  8: Unexpected shell/setup failure or execution outside GPUQ_BUILD_WINDOW=1.
# 10: CPU configure/build of HEAD (including 8ff3e34e4), or renamed server copy fails.
# 11: Explicit estimator/parser suite fails, including real depth/SSE/journal artifacts and estimator negative control.
# 12: Carry replay fails: both cpu-exact streams have 256 tokens, p/q error <= 1e-3, no support changes, nonempty rows.
# 13: Matrix fails: cpu_pipeline.sh codex/cpu-validation 16, both seeds per prompt, same row gates for all 32 streams.
# 14: cpu_controls.sh fails: rerun bytes and zero paired delta/SE, or either negative control loses < 50% acceptance.
# 15: Real sliced-head control fails: top-128 coverage/ties proven, probabilities agree <= 1e-3, intended vocab loaded.
# 16: Wrong-vocab negative control fails: check_vocab.py must return exactly 1 with finite error > 1e-3.
# 17: Deterministic rerun fails: full draft file inventory/header/bytes equal and repeated server tokens/trace equal.
# 20: CUDA stack6 configure/build fails. No CUDA executable is run.
# 21: DeviceTopK count differs from 25, ggml stamp differs from HEAD, or source changed during validation.
# 22: Freeze fails or frozen SOURCE differs from HEAD.
# 23: GPU item/pin regeneration fails.
# 24: gpuq submit --check fails.
# 25: CPU-only GPU-item preflight fails (hashes, stamps, recorded input). No GPU executable is run.
# Codes 10-17 stop before Stage 2. The payload never returns 3 or 4.
# Each check writes OUT/<stage>.log and one verdict line to stdout.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
W=$HOME/repos/qwen38-eval-worktrees
cd "$(dirname "$(readlink -f "$0")")/.."
ROOT=$PWD
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
FROZEN=$W/frozen/spec-tf-v2
BUILDGATE=$W/bin/buildgate
CPUGATE=$W/bin/cpugate
CPU=$ROOT/codex/cpu-validation
trap 'echo "bw-spec-tf: FAIL setup rc=8"; exit 8' ERR
trap 'echo "bw-spec-tf: FAIL interrupted rc=7"; exit 7' INT TERM

preflight() {
    python3 codex/bw_spec_tf.py preflight || return 1
    for tool in gpuq cmake python3 git strings ccache nice readlink flock ldd; do
        command -v "$tool" >/dev/null || return 1
    done
    command -v "${PATCHELF:-patchelf}" >/dev/null || return 1
    [[ $(git branch --show-current) == topic/spec-tf ]] || return 1
    git merge-base --is-ancestor 8ff3e34e4 HEAD || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    for script in "$0" codex/gate.sh codex/cpu_pipeline.sh codex/cpu_controls.sh codex/run_tests.sh codex/gpu_ab.sh; do
        bash -n "$script" || return 1
    done
}

if [[ ${1:-} == --preflight ]]; then
    if preflight; then
        echo 'bw-spec-tf: PASS preflight rc=0'
        exit 0
    fi
    echo 'bw-spec-tf: FAIL preflight rc=6'
    exit 6
fi
[[ $# -le 1 && ${GPUQ_BUILD_WINDOW:-0} == 1 ]] || { echo 'bw-spec-tf: FAIL requires build window rc=8'; exit 8; }
OUT=$(readlink -m "${1:-$ROOT/codex/bw-spec-tf-results}")
mkdir -p "$OUT"

run_stage() {
    local name=$1 code=$2 rc
    shift 2
    # Do not call the subshell in an if/|| condition: bash would disable errexit inside it.
    trap - ERR
    set +e
    (trap - ERR; set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    trap 'echo "bw-spec-tf: FAIL setup rc=8"; exit 8' ERR
    if [[ $rc != 0 ]]; then
        echo "bw-spec-tf: FAIL $name rc=$code command_rc=$rc log=$OUT/$name.log"
        exit "$code"
    fi
    echo "bw-spec-tf: PASS $name rc=0 log=$OUT/$name.log"
}

run_stage preflight 6 preflight
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
printf '%s\n' "$SHA" > "$OUT/source-commit.txt"
[[ ! -e $CPU && ! -e $OUT/carry ]] || { echo 'bw-spec-tf: FAIL existing CPU output rc=6'; exit 6; }
# One payload owns the CPU model sequence. cpugate also waits for RAM before every model process.
mkdir -p build-cpu
exec 9> "$ROOT/build-cpu/.bw-model.lock"
flock -n 9 || { echo 'bw-spec-tf: FAIL another spec-tf CPU payload holds the lock rc=6'; exit 6; }

cpu_build() {
    ./codex/gate.sh "$BUILDGATE" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
        -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
    ./codex/gate.sh "$BUILDGATE" cmake --build build-cpu --target llama-spec-tf llama-server -j4
    cp build-cpu/bin/llama-server build-cpu/bin/spec-tf-cpu-server
    [[ -x build-cpu/bin/spec-tf-cpu-server ]]
}
run_stage cpu-build 10 cpu_build
run_stage parser-estimator-tests 11 bash codex/run_tests.sh

FLAGS=(-m "$MODEL" -c 4096 -np 1 -ngl 0 -fa on -ctk q8_0 -ctv q8_0 --kv-pad 4096 -b 2048 -ub 512 \
    --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --ignore-eos --backend-sampling --spec-draft-backend-sampling \
    --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-type-k f16 --spec-draft-type-v f16 -t 8 -tb 8 \
    --tf-chunk 6 --tf-carry-streams)
CPU_GATE=("$CPUGATE" ./codex/gate.sh)
carry_check() {
    mkdir -p "$OUT/carry"
    cp codex/cpu-exact/streams.json "$OUT/carry/streams.json"
    cp codex/cpu-exact/trace.jsonl "$OUT/carry/trace.jsonl"
    "${CPU_GATE[@]}" build-cpu/bin/llama-spec-tf "${FLAGS[@]}" --tf-stage AB \
        --tf-streams "$OUT/carry/streams.json" --tf-out "$OUT/carry/tool"
    "${CPU_GATE[@]}" python3 codex/check_rows.py --target "$OUT/carry/tool/target" --arm "$OUT/carry/tool/full" \
        --trace "$OUT/carry/trace.jsonl" --streams "$OUT/carry/streams.json" --output "$OUT/carry/rows.json" --tolerance 1e-3
    "${CPU_GATE[@]}" python3 codex/bw_spec_tf.py rows "$OUT/carry" --streams 2
}
run_stage carry-rows 12 carry_check
matrix_check() {
    "$CPUGATE" ./codex/gate.sh bash codex/cpu_pipeline.sh codex/cpu-validation 16
    "${CPU_GATE[@]}" python3 codex/bw_spec_tf.py rows "$CPU" --streams 32
}
run_stage cpu-matrix 13 matrix_check
run_stage cpu-controls 14 "$CPUGATE" ./codex/gate.sh bash codex/cpu_controls.sh

slice_check() {
    "${CPU_GATE[@]}" python3 codex/bw_spec_tf.py slice-vocab "$CPU"
    "${CPU_GATE[@]}" build-cpu/bin/llama-spec-tf "${FLAGS[@]}" --tf-stage B --tf-target "$CPU/tool/target" \
        --spec-draft-vocab "$CPU/slice-vocab.json" --tf-out "$CPU/tool/sliced"
    "${CPU_GATE[@]}" python3 codex/check_vocab.py --target "$CPU/tool/target" --full "$CPU/tool/full" \
        --sliced "$CPU/tool/sliced" --vocab "$CPU/slice-vocab.json" --output "$CPU/slice-control.json" --tolerance 1e-3
    "${CPU_GATE[@]}" python3 codex/bw_spec_tf.py slice-result "$CPU"
}
run_stage real-sliced-head 15 slice_check
slice_negative() {
    local rc=0
    "${CPU_GATE[@]}" python3 codex/check_vocab.py --target "$CPU/tool/target" --full "$CPU/tool/full" \
        --sliced "$CPU/tool/sliced" --vocab "$CPU/slice-wrong-vocab.json" --output "$CPU/slice-negative.json" \
        --tolerance 1e-3 || rc=$?
    [[ $rc == 1 ]]
    "${CPU_GATE[@]}" python3 codex/bw_spec_tf.py slice-negative "$CPU"
}
run_stage slice-negative-control 16 slice_negative
deterministic_check() {
    "${CPU_GATE[@]}" python3 codex/cpu_validate.py --prompts 1 --predict 256 --out "$CPU/deterministic-server"
    "${CPU_GATE[@]}" python3 codex/bw_spec_tf.py deterministic "$CPU"
}
run_stage deterministic-rerun 17 deterministic_check
echo 'bw-spec-tf: PASS stage-1 rc=0; CPU checks complete'

cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a \
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_build() {
    ./codex/gate.sh "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}"
    ./codex/gate.sh "$BUILDGATE" cmake --build build-cuda --target llama-spec-tf llama-server -j4
    cp build-cuda/bin/llama-spec-tf build-cuda/bin/round-trace
}
run_stage cuda-build 20 cuda_build
stamp_check() {
    local count
    count=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    printf 'DeviceTopK strings_M=%s expected=25\n' "$count"
    [[ $count == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
    echo "ggml stamp=$SHORT; source identity PASS"
}
run_stage cuda-stamps 21 stamp_check
freeze_check() {
    "$W/bin/wt" freeze "$ROOT" spec-tf-v2 --build build-cuda \
        --desc "stack6 + spec-tf served conditioning and carry; CUDA 13.2, 120a, CCCL v3.4.3, NCCL OFF; round-trace aliases llama-spec-tf"
    local source
    IFS= read -r source < "$FROZEN/SOURCE"
    [[ $source == "$SHA" ]]
    cmp build-cuda/bin/llama-spec-tf build-cuda/bin/round-trace
    printf 'commit %s\nconfigure: cmake -B build-cuda ' "$SHA" > "$OUT/RECIPE.txt"
    printf '%q ' "${cuda_flags[@]}" >> "$OUT/RECIPE.txt"
    printf '\nbuild: buildgate cmake --build build-cuda --target llama-spec-tf llama-server -j4\n' >> "$OUT/RECIPE.txt"
    echo 'CUDA_VISIBLE_DEVICES=""; DeviceTopK strings_M=25; no CUDA executable run' >> "$OUT/RECIPE.txt"
}
run_stage freeze 22 freeze_check
run_stage generate-gpu-item 23 "$CPUGATE" ./codex/gate.sh python3 codex/gen_gpu_item.py
run_stage check-gpu-item 24 gpuq submit --check codex/spec-tf-j5q.gpuq.toml
run_stage gpu-item-preflight 25 "$CPUGATE" ./codex/gate.sh bash codex/gpu_ab.sh --mode preflight
echo 'bw-spec-tf: PASS stage-2 rc=0; spec-tf-v2 frozen; GPU item checked, NOT submitted'
exit 0
