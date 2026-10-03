#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
unset GGML_BACKEND_PATH LD_PRELOAD
export TMPDIR="$PWD/build-tmp"
mkdir -p "$TMPDIR" codex/reuse-results
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF)
cmake -B build-cpu "${flags[@]}" > codex/cmake-cpu.log 2>&1
cmake --build build-cpu --target llama-server test-alloc test-llama-archs test-sampling -j8 > codex/build-cpu.log 2>&1
if [[ ! -d codex/base-src ]]; then
    mkdir -p codex/base-src
    git archive "$(cat codex/BASE)" | tar -x -C codex/base-src
fi
cmake -S codex/base-src -B build-base-cpu "${flags[@]}" > codex/cmake-base-cpu.log 2>&1
cmake --build build-base-cpu --target llama-server -j8 > codex/build-base-cpu.log 2>&1
# Only the fixture needs trace symbols; the stack2 libraries remain unmodified.
g++ -std=c++17 -O2 -pthread -DGGML_SHARED -Iggml/include -c ggml/src/ggml-round-trace.cpp -o build-tmp/base-trace.o
for build in build-base-cpu build-cpu; do
    trace=()
    [[ $build != build-base-cpu ]] || trace=(build-tmp/base-trace.o)
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Icommon -Iinclude -Iggml/include -Isrc -Ivendor -I. codex/reuse-cpu.cpp "${trace[@]}" "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/reuse-cpu"
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Itools/server -Icommon -Iinclude -Iggml/include -Ivendor -Itools/mtmd -I. codex/reuse-server-cpu.cpp "${trace[@]}" "$build/tools/server/libserver-context.a" "$build/common/libllama-common-base.a" -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/reuse-server-cpu"
    cp "$build/bin/llama-server" "$build/bin/cpu-serving"
done
for fixture in reuse-contract input-scheduler-contract; do
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Iggml/include -Isrc -I. "codex/$fixture.cpp" -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml -lggml-cpu -lggml-base -o "build-cpu/bin/$fixture"
done
g++ -std=c++17 -O2 -DGGML_SHARED -Iggml/include codex/prepare_cpu_target.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml-base -o build-cpu/bin/prepare-cpu-target
export MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
if [[ ! -f build-tmp/reuse-untied-target.gguf ]]; then
    build-cpu/bin/prepare-cpu-target "$MODEL" build-tmp/reuse-untied-target.gguf > codex/prepare-target.log 2>&1
fi
build-cpu/bin/test-alloc > codex/test-alloc.log 2>&1
build-cpu/bin/reuse-contract > codex/contracts.log 2>&1
GGML_SCHED_INPUT_STAGING=1 build-cpu/bin/input-scheduler-contract > codex/input-scheduler-contract.log 2>&1
build-cpu/bin/test-llama-archs --tp-sampling > codex/tp-sampling.log 2>&1
build-cpu/bin/test-sampling > codex/sampling.log 2>&1
python3 codex/test_input_staging.py > codex/staging-growth.log 2>&1
python3 codex/negative_reuse.py > codex/negative-reuse.log 2>&1
python3 codex/negative_meta.py > codex/negative-meta.log 2>&1
python3 codex/run_reuse_cpu.py --seeds 8 --rounds 12 > codex/synthetic.log 2>&1
python3 codex/run_reuse_cpu.py --real --seeds 4 --rounds 12 > codex/real.log 2>&1
python3 codex/reuse_flag_control.py > codex/reuse-flag.log 2>&1
python3 codex/cpu_identity.py > codex/cpu-identity.log 2>&1

CPU_CACHE_PROMPT=0 python3 codex/prefill_cpu.py > codex/prefill.log 2>&1
g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED -Icommon -Iinclude -Iggml/include -Isrc -Ivendor -I. codex/input-cpu.cpp build-cpu/common/libllama-common-base.a -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama-common -lllama -lggml -lggml-cpu -lggml-base -o build-cpu/bin/input-cpu
python3 codex/run_owned_cpu.py > codex/owned-input.log 2>&1
