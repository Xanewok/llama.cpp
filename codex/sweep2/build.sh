#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then exec nice -n 19 bash "$0" "$@"; fi
cd "$(dirname "$0")/../.."
export TMPDIR="$PWD/build-tmp" CUDA_CACHE_PATH="$PWD/build-tmp/cuda-cache"
mkdir -p "$TMPDIR" codex/sweep2/checks
[[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -ge 10485760 ]]
cccl=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$cccl" -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON >codex/sweep2/checks/configure.log 2>&1
python3 - <<'MONITOR'
import os
import signal
import subprocess
import time
from pathlib import Path
floor = 10 * 1024 * 1024
minimum = float("inf")
def available():
    return int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))
if available() < floor:
    raise SystemExit("RAM floor: retry later")
with open("codex/sweep2/checks/build.log", "a") as log:
    proc = subprocess.Popen(["cmake", "--build", "build-cuda", "--target", "bench-mmvq-shapes", "-j", "6"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while proc.poll() is None:
        ram = available()
        minimum = min(minimum, ram)
        if ram < floor:
            os.killpg(proc.pid, signal.SIGSTOP)
            print("RAM floor: build paused", flush=True)
            deadline = time.monotonic() + 120
            while available() < floor and time.monotonic() < deadline:
                time.sleep(0.25)
            if available() < floor:
                os.killpg(proc.pid, signal.SIGTERM)
                os.killpg(proc.pid, signal.SIGCONT)
                proc.wait()
                raise SystemExit("RAM floor: build stopped; retry later")
            os.killpg(proc.pid, signal.SIGCONT)
        time.sleep(0.1)
    Path("codex/sweep2/checks/build-memory.txt").write_text(f"MemAvailable minimum observed: {minimum} KiB (M); polling 0.1 s\n")
    raise SystemExit(proc.returncode)
MONITOR
