#!/usr/bin/env bash
# Build-window payload (gpuq filler item; never submits anything). Usage: bw-cand-neginf.sh [--preflight | OUT_DIR]
#
# Stage 1  CPU build (-DGGML_CUDA=OFF, tests on) of HEAD, then:
#          a. test-sampling and test-llama-archs --tp-sampling (Meta(CPU,CPU) banned-id candidates incl. an unextended-chain
#             control, synthetic rows with ties, rejection-sampling verdicts) on two seeds;
#          b. code mutant: sampling.cpp without the k + n_ban extension, linked into test-llama-archs ahead of
#             libllama-common; it must fail with a full-row read on a row whose top k is unique;
#          c. server_context driver on the 4B fixture, -c 4096, ignore_eos: Meta(CPU,CPU) MTP and plain x temp 0/1 x arms
#             cand/host/refused, and plain CPU MTP x temp 0/1 x cand/host; bw-cand-compare.py: tokens, draft counters and
#             post-sampling top-20 sets identical. The tied-head fixture cannot prove TP backend engagement; the CUDA
#             item's canary_log requires consumed banned-token candidates and a frozen-stack6 top-k refusal.
# Stage 2  CUDA build (stack6 RECIPE flags) + CCCL check + round-trace copy + wt freeze cand-neginf + RECIPE.txt.
# Stage 3  Render the depth item (stack6 vs cand-neginf, ABBA, cross-arm identity) and run gpuq submit --check only.
#
# Exit codes: 0 all pass | 2 unit test failure | 3 mutant escaped (passed, or failed for another reason) | 4 e2e identity
# failure | 8 CPU build/run error or preflight | 9 CUDA build/freeze/CCCL | 10 item render/check.
# 2 3 4 8 stop before stage 2 (no freeze, no item).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${BW_CPU_ONLY:-0} != 1 ]]; then
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
fi
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
cd "$(dirname "$(readlink -f "$0")")/.." || exit 8
ROOT=$PWD
MODEL=${BW_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
NAME=cand-neginf
ITEM_ID=cand-neginf-depth-j5q
log() { echo "bw-$NAME: $(date '+%H:%M:%S') $*"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }

if [[ ${1:-} == --preflight ]]; then
    paths=("${BW_BUILDGATE:-$W/bin/buildgate}" "${BW_GATE:-$W/bin/cpugate}" "$MODEL" "$ROOT/codex/bw-cand-driver.cpp" "$ROOT/codex/bw-cand-compare.py")
    if [[ ${BW_CPU_ONLY:-0} != 1 ]]; then
        paths+=("$W/bin/wt" "$CCCL" "$ROOT/codex/$ITEM_ID.toml.in" "$W/frozen/stack6/build/bin/round-trace"
                "$EVAL/harness/mkitem.py" "$EVAL/orchestration/gpuq/items/drafts")
    fi
    for p in "${paths[@]}"; do
        [[ -e $p ]] || die 8 "preflight missing $p"
    done
    [[ ${BW_CPU_ONLY:-0} != 1 ]] || exit 0
    command -v gpuq >/dev/null || die 8 "preflight: gpuq not on PATH"
    [[ ! -e $W/frozen/$NAME ]] || die 8 "preflight: $W/frozen/$NAME exists already"
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
cpu_extra=()
# BW_CPU_EXTRA is a whitespace-separated list of cmake arguments; no shell expansion.
read -r -a cpu_extra <<< "${BW_CPU_EXTRA:-}"
stage_start=$SECONDS
stage_time() {
    if [[ ${BW_CPU_ONLY:-0} == 1 ]]; then
        printf '%s\t%s\n' "$1" "$((SECONDS - stage_start))" >> "$OUT/stage-wall-seconds.tsv"
    fi
    stage_start=$SECONDS
}
RESULTS=$OUT/RESULTS.txt
: > "$RESULTS"
note() { echo "$*" | tee -a "$RESULTS"; }
note "cand-neginf stage 1, HEAD $SHA, $(date -Iseconds)"

# ---------------------------------------------------------------- stage 1: CPU build
{
    "$BUILDGATE" cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${cpu_extra[@]}" &&
    "$BUILDGATE" cmake --build build-cpu --target llama-server test-llama-archs test-sampling -j"${BW_JOBS:-4}" &&
    "$BUILDGATE" g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Ivendor \
        -Itools/mtmd -I. codex/bw-cand-driver.cpp build-cpu/tools/server/libserver-context.a build-cpu/common/libllama-common-base.a \
        -Lbuild-cpu/bin -Wl,-rpath,"$ROOT/build-cpu/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base \
        -o build-cpu/bin/bw-cand-driver
} > "$OUT/build-cpu.log" 2>&1 || die 8 "CPU build (see $OUT/build-cpu.log)"
[[ $(git rev-parse HEAD) == "$SHA" ]] || die 8 "HEAD moved during the CPU build"
log "CPU build ok"
stage_time cpu-build

# ---------------------------------------------------------------- stage 1a: unit tests
"$CPUGATE" build-cpu/bin/test-sampling > "$OUT/test-sampling.log" 2>&1 || die 2 "test-sampling (see $OUT/test-sampling.log)"
for seed in 42 1729; do
    "$CPUGATE" build-cpu/bin/test-llama-archs --tp-sampling -s "$seed" > "$OUT/tp-sampling-$seed.log" 2>&1 ||
        die 2 "test-llama-archs --tp-sampling -s $seed (see $OUT/tp-sampling-$seed.log)"
    grep -q "test_banned_candidates: .*: OK" "$OUT/tp-sampling-$seed.log" || die 2 "synthetic banned-candidate test did not report OK (seed $seed)"
    grep -q "unextended chain caught.*: OK" "$OUT/tp-sampling-$seed.log" || die 2 "unextended-chain control did not report OK (seed $seed)"
    note "unit seed $seed: $(grep -E 'banned ids|test_banned_candidates:' "$OUT/tp-sampling-$seed.log" | tr '\n' ' ')"
done
log "unit tests ok"
stage_time unit-tests

# ---------------------------------------------------------------- stage 1b: code mutant (no extension)
mutant_stage() {
    local m=$OUT/mutant
    if [[ ${BW_CPU_ONLY:-0} == 1 ]]; then
        mkdir -p "$m" || return 1
    else
        rm -rf "$m" && mkdir -p "$m" || return 1
    fi
    sed 's/llama_sampler_init_top_k(candidate_k + 1 + n_ban)/llama_sampler_init_top_k(candidate_k + 1)/' common/sampling.cpp > "$m/sampling.cpp"
    ! cmp -s common/sampling.cpp "$m/sampling.cpp" || { echo "mutant sed changed nothing"; return 1; }
    # the library's own compile command for sampling.cpp, pointed at the mutant source and object
    python3 - "$ROOT" "$m" > "$m/compile.sh" <<'EOF' || return 1
import json, shlex, sys
root, m = sys.argv[1:]
for e in json.load(open(f"{root}/build-cpu/compile_commands.json")):
    if e["file"].endswith("/common/sampling.cpp"):
        args = e.get("arguments") or shlex.split(e["command"])
        i = args.index("-o")
        args[i + 1] = f"{m}/sampling.o"
        args = [f"{m}/sampling.cpp" if a == e["file"] or a.endswith("/common/sampling.cpp") else a for a in args]
        print("cd " + shlex.quote(e["directory"]) + " && " + shlex.join(args))
        break
else:
    sys.exit("common/sampling.cpp not in compile_commands.json")
EOF
    "$BUILDGATE" bash "$m/compile.sh" || return 1
    # the test's own link line, plus the mutant object ahead of libllama-common (executable definitions win)
    local link
    if [[ -f build-cpu/build.ninja ]]; then
        ninja -C build-cpu -t commands test-llama-archs | tail -1 > "$m/link.txt" || return 1
        link=$(python3 - "$m" <<'EOF'
import shlex, sys
from pathlib import Path
m = Path(sys.argv[1])
args = shlex.split((m / 'link.txt').read_text())
if args[:2] == [':', '&&']:
    args = args[2:]
if args[-2:] == ['&&', ':']:
    args = args[:-2]
i = args.index('-o')
args[i + 1] = str(m / 'test-llama-archs-mutant')
args.insert(i + 2, str(m / 'sampling.o'))
print(shlex.join(args))
EOF
        ) || return 1
    else
        link=$(sed "s| -o [^ ]*test-llama-archs | -o $m/test-llama-archs-mutant $m/sampling.o |" build-cpu/tests/CMakeFiles/test-llama-archs.dir/link.txt)
    fi
    [[ $link == *"$m/sampling.o"* ]] || { echo "could not rewrite the link line"; return 1; }
    if [[ -f build-cpu/build.ninja ]]; then
        (cd build-cpu && "$BUILDGATE" bash -c "$link") || return 1
    else
        (cd build-cpu/tests && "$BUILDGATE" bash -c "$link") || return 1
    fi
    nm -C "$m/test-llama-archs-mutant" | grep " T common_sampler_init" >/dev/null || { echo "mutant common_sampler_init not in the executable"; return 1; }
}
mutant_stage > "$OUT/mutant-build.log" 2>&1 || die 8 "mutant build (see $OUT/mutant-build.log)"
"$CPUGATE" "$OUT/mutant/test-llama-archs-mutant" --tp-sampling -s 42 > "$OUT/mutant-run.log" 2>&1
mrc=$?
[[ $mrc != 0 ]] || die 3 "mutant escaped: test-llama-archs passed without the extension (see $OUT/mutant-run.log)"
grep -q "banned: read the full row although the top k is unique" "$OUT/mutant-run.log" ||
    die 3 "mutant failed (rc=$mrc) but not on a full-row read (see $OUT/mutant-run.log)"
note "mutant (no extension): rc=$mrc, caught by: $(grep -m1 'banned: read the full row' "$OUT/mutant-run.log")"
log "mutant caught"
stage_time mutant

# ---------------------------------------------------------------- stage 1c: server_context driver
run_driver() { # meta spec temp arm
    if [[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]]; then
        while ! python3 - <<'EOF'
import re, subprocess, sys
memory = subprocess.check_output(['memory_pressure', '-Q'], text=True)
fraction = int(re.search(r'free percentage: (\d+)%', memory)[1]) / 100
total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
models = {'bw-cand-driver', 'bw-np2-driver', 'chain-cpu', 'fold-cpu', 'outproj-cpu'}
busy = any(p.rsplit('/', 1)[-1] in models for p in subprocess.check_output(['ps', '-axo', 'comm'], text=True).splitlines())
if busy or total * fraction < 4 * 1024**3:
    print('WAIT: another model process or less than 4 GiB available:', memory.strip(), flush=True)
    sys.exit(1)
EOF
        do
            sleep 20
        done
    fi
    LD_LIBRARY_PATH="$ROOT/build-cpu/bin" "$CPUGATE" build-cpu/bin/bw-cand-driver "$MODEL" "$OUT/e2e/$1-$2-$3-$4.json" "$1" "$2" "$3" "$4" 1729 96 \
        > "$OUT/e2e/$1-$2-$3-$4.log" 2>&1 || die 8 "driver meta=$1 spec=$2 temp=$3 arm=$4 (see $OUT/e2e/$1-$2-$3-$4.log)"
    log "ran meta=$1 spec=$2 temp=$3 arm=$4"
}
if [[ ${BW_CPU_ONLY:-0} == 1 ]]; then
    mkdir -p "$OUT/e2e"
else
    rm -rf "$OUT/e2e" && mkdir -p "$OUT/e2e"
fi
for spec in 1 0; do
    for temp in 0 1; do
        for arm in cand host refused; do
            run_driver 1 "$spec" "$temp" "$arm"
        done
    done
done
for temp in 0 1; do
    for arm in cand host; do
        run_driver 0 1 "$temp" "$arm"
    done
done
python3 codex/bw-cand-compare.py "$OUT/e2e" > "$OUT/e2e-compare.log" 2>&1
crc=$?
tee -a "$RESULTS" < "$OUT/e2e-compare.log"
case $crc in
    0) ;;
    2) die 4 "e2e identity (see $OUT/e2e-compare.log)" ;;
    *) die 8 "e2e compare rc=$crc (see $OUT/e2e-compare.log)" ;;
esac
python3 codex/check-cand-proof.py "$OUT/e2e" > "$OUT/proof-controls.log" 2>&1 || die 8 "proof controls (see $OUT/proof-controls.log)"
tee -a "$RESULTS" < "$OUT/proof-controls.log"
note "stage 1 PASS"
log "stage 1 pass"
stage_time driver-and-compare
[[ ${BW_CPU_ONLY:-0} != 1 ]] || exit 0

# ---------------------------------------------------------------- stage 2: CUDA build, freeze
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_stage() {
    export CUDA_HOME=/usr/local/cuda-13.2
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}" || return 1
    "$BUILDGATE" cmake --build build-cuda --target llama-server -j4 || return 1
    [[ $(git rev-parse HEAD) == "$SHA" ]] || { echo "HEAD moved during the build"; return 1; }
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace || return 1
    local n; n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "CCCL DeviceTopK strings: $n"
    [[ $n == 25 ]] || { echo "expected 25 DeviceTopK strings (CCCL v3.4.3)"; return 1; }
    local short; short=$(git rev-parse --short HEAD)
    strings build-cuda/bin/libggml-base.so | grep -x "$short" >/dev/null || { echo "ggml stamp $short not in libggml-base.so"; return 1; }
    "$W/bin/wt" freeze "$ROOT" "$NAME" --build build-cuda \
        --desc "stack6 ac7465885 + backend candidates kept under -inf logit_bias (ignore_eos); CUDA 13.2, CCCL343, NCCL OFF; round-trace = cp of llama-server" || return 1
    {
        printf 'name %s\n' "$NAME"
        printf 'commit %s (base stack6 ac7465885dc9f13de396ccd91b4be5eaffde3730, branch topic/cand-neginf-bias)\n' "$SHA"
        printf 'worktree %s, build dir build-cuda\n' "$ROOT"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${cuda_flags[@]}"
        printf ' (CUDA_HOME=/usr/local/cuda-13.2, CUDA_VISIBLE_DEVICES="")\n'
        printf 'build: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'round-trace = cp of llama-server (harness arms must not execute a file named llama-server); listed in SHA256SUMS\n'
        printf 'DeviceTopK strings %s; ggml stamp %s; no CUDA binary executed; CPU evidence %s\n' "$n" "$short" "$RESULTS"
    } > "$W/frozen/$NAME/RECIPE.txt"
    [[ $(head -1 "$W/frozen/$NAME/SOURCE" | cut -d' ' -f1) == "$SHA" ]]
}
cuda_stage > "$OUT/cuda.log" 2>&1 || die 9 "CUDA build/freeze/CCCL (see $OUT/cuda.log)"
log "CUDA build frozen at $W/frozen/$NAME"

# ---------------------------------------------------------------- stage 3: item (check only, never submit)
item_stage() {
    local exp=$EVAL/harness/experiments/$ITEM_ID.toml item=$EVAL/orchestration/gpuq/items/drafts/$ITEM_ID.toml
    [[ ! -e $exp ]] || { echo "$exp exists; not overwriting"; return 1; }
    sed -e "s/@COMMIT@/$SHA/g" -e "s/@DATE@/$(date +%F)/g" -e "s|@RESULTS@|$RESULTS|g" "codex/$ITEM_ID.toml.in" > "$exp" || return 1
    ! grep -q '@[A-Z]*@' "$exp" || { echo "unsubstituted placeholder in $exp"; return 1; }
    (cd "$EVAL" && python3 -m harness.mkitem "$exp" --out "$item") || return 1
    gpuq submit --check "$item" || return 1
    echo "item checked, NOT submitted: $item"
}
item_stage > "$OUT/items.log" 2>&1 || die 10 "item render/check (see $OUT/items.log)"
log "item checked, NOT submitted: $EVAL/orchestration/gpuq/items/drafts/$ITEM_ID.toml"

log "done"
exit 0
