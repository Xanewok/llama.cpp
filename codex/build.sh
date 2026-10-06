#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
ulimit -c 0
cd "$(dirname "$0")/.."
bin=/home/xanewok/repos/qwen38-eval-worktrees/bin
mkdir -p codex/logs build-tmp
export TMPDIR="$PWD/build-tmp"

quiet() {
    local status
    # Inside the gpuq build-window item the queue slot itself is the quiet period (the item shows as RUN).
    [[ ${GPUQ_BUILD_WINDOW:-} == 1 ]] && return 0
    while true; do
        status=$(gpuq status --brief)
        printf '%s\n' "$status" >&2
        if [[ $status != *'gpuq: RUN '* && $status == *'ready 0 '* ]]; then break; fi
        sleep 15
    done
}

stop_job() {
    python3 - "$1" <<'PY'
import os, signal, subprocess, sys
root = int(sys.argv[1])
try:
    assert os.readlink(f'/proc/{root}/cwd').startswith(os.getcwd())
except FileNotFoundError:
    sys.exit(0)
children = {}
for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True).splitlines():
    pid, ppid = map(int, line.split())
    children.setdefault(ppid, []).append(pid)
pending = [root]
for pid in pending:
    pending.extend(children.get(pid, []))
for pid in pending:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
PY
}

gated() {
    local gate=$1 pid status aborted rc
    shift
    while true; do
        quiet
        "$gate" "$@" &
        pid=$! aborted=0
        while kill -0 "$pid" 2>/dev/null; do
            status=$(gpuq status --brief)
            if [[ ${GPUQ_BUILD_WINDOW:-} != 1 && $status == *'gpuq: RUN '* ]]; then
                printf 'Build/test interrupted by queue start: %s\n' "$status" >&2
                stop_job "$pid"
                aborted=1
                break
            fi
            sleep 5
        done
        if wait "$pid"; then rc=0; else rc=$?; fi
        if [[ $aborted == 0 ]]; then return "$rc"; fi
    done
}

compile() { gated "$bin/buildgate" "$@"; }
cpu() { gated "$bin/cpugate" "$@"; }

case ${1:-} in
    cpu)
        compile cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=ON \
            -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON
        compile cmake --build build-cpu --target test-backend-ops -j4
        cpu python3 codex/config_cpu.py
        compile c++ -O2 -UNDEBUG -std=c++17 -Iggml/include -Iggml/src/ggml-cuda codex/config-cpu.cpp -o codex/config-cpu
        cpu codex/config-cpu
        compile c++ -O2 -UNDEBUG -std=c++17 -DGGML_CUDA_MMQ_SMALLJ_DISABLE -Iggml/include -Iggml/src/ggml-cuda \
            codex/config-cpu.cpp -o codex/config-cpu-base
        cpu codex/config-cpu-base
        compile c++ -O2 -UNDEBUG -std=c++17 -Icodex/negative-config -Iggml/include -Iggml/src/ggml-cuda \
            codex/config-cpu.cpp -o codex/config-cpu-negative
        if cpu codex/config-cpu-negative > codex/logs/config-negative.log 2>&1; then
            echo 'Wrong I tile escaped the CPU config contract' >&2
            exit 1
        else
            rc=$?
            [[ $rc == 134 ]] || exit "$rc"
            echo 'CPU wrong-tile negative control PASS'
        fi
        cpu python3 codex/kernel_item.py memcheck --stage selftest
        ;;
    cuda|negative)
        build=build-cuda
        flags=-Xptxas=-v
        if [[ $1 == negative ]]; then
            build=build-cuda-negative
            flags='-Xptxas=-v -DGGML_CUDA_MMQ_SMALLJ_NEGATIVE_CONTROL'
        fi
        compile cmake -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
            -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF \
            -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
            -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 \
            -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON \
            -DCMAKE_CUDA_FLAGS="$flags"
        compile cmake --build "$build" --target llama-server test-backend-ops -j4
        # Without CCCL >= 3.4.3 the CUB DeviceTopK path is compiled out (stack6 has 25 strings): a speed confound, not a result.
        topk=$(strings "$build/bin/libggml-cuda.so" | grep -c DeviceTopK || true)
        [[ $topk == 25 ]] || { echo "CCCL check failed: $topk DeviceTopK strings in $build/bin/libggml-cuda.so, expected 25" >&2; exit 1; }
        cp "$build/bin/llama-server" "$build/bin/round-trace"
        ;;
    resources)
        mkdir -p codex/resources
        cpu python3 codex/gen_resource_commands.py > codex/resources/commands.tsv
        while IFS=$'\t' read -r mode quant command; do
            compile bash -c "$command" > "codex/resources/$mode-$quant.log" 2>&1
        done < codex/resources/commands.tsv
        cpu python3 codex/gen_resource_summary.py
        ;;
    *) echo 'usage: build.sh {cpu|cuda|negative|resources}' >&2; exit 2 ;;
esac
