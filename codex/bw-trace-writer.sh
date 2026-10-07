#!/usr/bin/env bash
# Build-window payload only. 0 passes; 10 preflight; 20 CPU build; 21 correctness; 22 timing; 23 recorded parsers;
# 24 control sources; 25 TSan build; 26 TSan stress; 30 CUDA build; 31 freeze; 32 stamp; 33 CCCL; 34 moving HEAD; 40 item. Never returns 3 or 4.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
BUILDGATE=$W/bin/buildgate
CPUGATE=$W/bin/cpugate
STACK6=ac7465885dc9f13de396ccd91b4be5eaffde3730
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
EVAL=/home/xanewok/repos/qwen38-eval
log() { echo "bw-trace-writer: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    for p in "$BUILDGATE" "$CPUGATE" "$W/bin/wt" "$CCCL" /usr/local/cuda-13.2/bin/nvcc \
             "$ROOT/codex/trace-test/CMakeLists.txt" "$ROOT/codex/test-round-trace.cpp" \
             "$ROOT/codex/check_trace_writer.py" "$ROOT/codex/trace_gap.py" "$ROOT/codex/bw_make_trace_item.py" \
             "$ROOT/codex/gpu_ab.sh" "$ROOT/codex/trace-writer-depth-j5q.toml.in" \
             "$EVAL/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/round-trace.jsonl" \
             /home/xanewok/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf \
             /home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json; do
        [[ -e $p ]] || die 10 "missing $p"
    done
    for t in gpuq cmake ccache g++ strings git python3 timeout; do
        command -v "$t" >/dev/null || die 10 "missing command $t"
    done
    command -v "${PATCHELF:-patchelf}" >/dev/null || die 10 'wt freeze needs patchelf'
    for script in "$0" "$ROOT/codex/gpu_ab.sh"; do
        bash -n "$script" || die 10 "shell syntax: $script"
    done
    python3 - "$ROOT" <<'PY' || die 10 'Python/template syntax'
from pathlib import Path
import sys, tomllib
root = Path(sys.argv[1])
for p in root.joinpath('codex').glob('*.py'):
    compile(p.read_text(), str(p), 'exec')
tomllib.loads(root.joinpath('codex/trace-writer-depth-j5q.toml.in').read_text())
PY
    git merge-base --is-ancestor "$STACK6" HEAD || die 10 'stack6 not an ancestor'
    [[ ! -e $W/frozen/trace-writer ]] || die 10 'freeze trace-writer already exists'
    free_gib=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
    [[ $free_gib -ge 8 ]] || die 10 "need 8 GiB free; have $free_gib"
    exit 0
fi

[[ ${GPUQ_BUILD_WINDOW:-} == 1 ]] || die 10 'requires the build-window wrapper (no builds during measure items)'
OUT=$(readlink -m "${1:-$ROOT/codex/bw-results}")
case $OUT in "$ROOT"/codex/*) ;; *) die 10 'OUT must be inside this worktree/codex' ;; esac
mkdir -p "$OUT"
[[ -z $(git status --porcelain --untracked-files=no) ]] || die 10 'tracked changes; commit before freeze'
bash "$0" --preflight || exit $?
SHA=$(git rev-parse HEAD)
log "start HEAD=$SHA out=$OUT"

# Stage 1: independent old-writer oracle and a final-enqueue mutant, all CPU and no model.
mkdir -p build-cpu
git show "$STACK6:ggml/src/ggml-round-trace.cpp" > build-cpu/old-round-trace.cpp || die 24 'old writer source'
python3 - <<'PY' || die 24 'mutant source'
from pathlib import Path
s = Path('ggml/src/ggml-round-trace.cpp').read_text()
old = 'void ggml_round_trace_flush() {\n    if (ggml_round_trace_enabled()) {\n        trace_flush(0);'
new = 'void ggml_round_trace_flush() {\n    if (ggml_round_trace_enabled()) {'
if s.count(old) != 1:
    raise ValueError('public flush changed; update the negative control')
Path('build-cpu/mutant-round-trace.cpp').write_text(s.replace(old, new))
PY
{
    "$BUILDGATE" cmake -S codex/trace-test -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release &&
    "$BUILDGATE" cmake --build build-cpu -j4
} > "$OUT/build-cpu.log" 2>&1 || die 20 "CPU build: $OUT/build-cpu.log"
rc=0
"$CPUGATE" python3 codex/check_trace_writer.py build-cpu "$OUT/cpu" > "$OUT/cpu.log" 2>&1 || rc=$?
case $rc in 0) ;; 22) die 22 "timing >= 20 percent: $OUT/cpu.log" ;; *) die 21 "CPU correctness/negative control: $OUT/cpu.log" ;; esac
export PYTHONPATH="$EVAL"
"$CPUGATE" python3 codex/trace_gap.py --fixtures > "$OUT/parsers.log" 2>&1 || die 23 "recorded parsers: $OUT/parsers.log"
log 'CPU identity, negative control, drains, stress, timing and recorded parsers passed'

# Stage 1b: instrument the writer and all producers, without the old-writer oracle or timing gate.
{
    "$BUILDGATE" cmake -S codex/trace-test -B build-cpu-tsan -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_ROUND_TRACE_TSAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo &&
    "$BUILDGATE" cmake --build build-cpu-tsan --target trace-test-tsan -j4
} > "$OUT/build-cpu-tsan.log" 2>&1 || die 25 "TSan build: $OUT/build-cpu-tsan.log"
[[ ! -e $OUT/tsan-stress.jsonl ]] || die 26 'TSan output already exists'
rc=0
LLAMA_ROUND_TRACE="$OUT/tsan-stress.jsonl" GGML_ROUND_TRACE_SYNC=0 TSAN_OPTIONS=halt_on_error=1:exitcode=66 \
    "$CPUGATE" timeout 120 build-cpu-tsan/trace-test-tsan --stress > "$OUT/tsan-stress.log" 2>&1 || rc=$?
[[ $rc == 0 ]] || die 26 "TSan stress rc=$rc: $OUT/tsan-stress.log"
log 'TSan producer order, backpressure and concurrent flush stress passed'

# Stage 2: stack6 CUDA recipe; inspect artifacts but never execute a CUDA binary.
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
{
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}" &&
    "$BUILDGATE" cmake --build build-cuda --target llama-server -j4
} > "$OUT/cuda.log" 2>&1 || die 30 "CUDA build: $OUT/cuda.log"
[[ $(git rev-parse HEAD) == "$SHA" ]] || die 34 'HEAD moved during build'
cp build-cuda/bin/llama-server build-cuda/bin/round-trace || die 30 'copy renamed server'
n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK) || die 33 'DeviceTopK scan'
[[ $n == 25 ]] || die 33 "expected 25 DeviceTopK strings; got $n"
short=$(git rev-parse --short HEAD)
strings build-cuda/bin/libggml-base.so | grep -x "$short" >/dev/null || die 32 "missing ggml stamp $short"
"$W/bin/wt" freeze "$ROOT" trace-writer --build build-cuda \
    --desc "stack6 + async round trace writer; CUDA 13.2, CCCL343, NCCL OFF; CPU byte identity and mutant pass" \
    > "$OUT/freeze.log" 2>&1 || die 31 "freeze: $OUT/freeze.log"
# Keep provenance beside the frozen bytes.
{
    printf 'commit %s (base stack6 %s)\n' "$SHA" "$STACK6"
    printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
    printf '\nbuild: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
    printf 'DeviceTopK strings %s; ggml stamp %s; CUDA_VISIBLE_DEVICES=""; no CUDA binary executed\n' "$n" "$short"
    printf 'round-trace is a copy of llama-server; all frozen binaries have RUNPATH $ORIGIN and SHA256SUMS\n'
} > "$W/frozen/trace-writer/RECIPE.txt" || die 31 'frozen recipe write'
log "frozen $W/frozen/trace-writer"

# Stage 3: pin the real freeze and passing CPU evidence; validate, never submit.
python3 codex/bw_make_trace_item.py "$OUT/cpu/cpu-evidence.json" > "$OUT/items.log" 2>&1 || die 40 "item generation: $OUT/items.log"
gpuq submit --check codex/items/trace-writer-depth-j5q.gpuq.toml >> "$OUT/items.log" 2>&1 || die 40 "item contract: $OUT/items.log"
log 'GPU item checked, NOT submitted'
exit 0
