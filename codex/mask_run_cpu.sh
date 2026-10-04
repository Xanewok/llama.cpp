#!/usr/bin/env bash
# Stack3's CPU identity/contract tests against this tree; the "up" arm is unpatched stack3 (~/repos/stack3/build-cpu, read-only).
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
unset GGML_BACKEND_PATH LD_PRELOAD
export TMPDIR="$PWD/build-tmp"
mkdir -p "$TMPDIR" codex/mask-reuse-results
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF)
cmake -B build-cpu "${flags[@]}" > build-tmp/cmake-cpu.log 2>&1
cmake --build build-cpu --target llama-server test-alloc test-llama-archs test-sampling -j"${CPU_JOBS:-8}" > build-tmp/build-cpu.log 2>&1
build=build-cpu
g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Icommon -Iinclude -Iggml/include -Isrc -Ivendor -I. codex/reuse-cpu.cpp "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/reuse-cpu"
g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Ivendor -Itools/mtmd -I. codex/reuse-server-cpu.cpp "$build/tools/server/libserver-context.a" "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/reuse-server-cpu"
cp "$build/bin/llama-server" "$build/bin/cpu-serving"
for fixture in reuse-contract input-scheduler-contract; do
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Iggml/include -Isrc -I. "codex/$fixture.cpp" -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml -lggml-cpu -lggml-base -o "build-cpu/bin/$fixture"
done
g++ -std=c++17 -O2 -DGGML_SHARED -Iggml/include codex/prepare_cpu_target.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml-base -o build-cpu/bin/prepare-cpu-target
export MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
[[ -f build-tmp/reuse-untied-target.gguf ]] || build-cpu/bin/prepare-cpu-target "$MODEL" build-tmp/reuse-untied-target.gguf > codex/mask-prepare-target.log 2>&1
r() { local log=$1; shift; "$@" > "codex/mask-$log.log" 2>&1 && echo "PASS $log" || { echo "FAIL $log rc=$?"; return 0; }; }
r test-alloc build-cpu/bin/test-alloc
r reuse-contract build-cpu/bin/reuse-contract
GGML_SCHED_INPUT_STAGING=1 r input-scheduler-contract build-cpu/bin/input-scheduler-contract
r tp-sampling build-cpu/bin/test-llama-archs --tp-sampling
r sampling build-cpu/bin/test-sampling
r synthetic python3 codex/mask_run_reuse_cpu.py --seeds "${SEEDS:-2}" --rounds 12
r real python3 codex/mask_run_reuse_cpu.py --real --seeds 1 --rounds 12
r cpu-identity python3 codex/mask_cpu_identity.py
