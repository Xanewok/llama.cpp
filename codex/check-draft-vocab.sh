#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF > codex/draft-vocab-cmake-cpu.log 2>&1
cmake --build build-cpu --target test-sampling-prefilter test-sampling -j8 > codex/draft-vocab-build-cpu.log 2>&1
for source in gguf-draft-vocab test-draft-vocab; do
    g++ -std=c++17 -O2 -DGGML_SHARED -DLLAMA_SHARED -Iinclude -Iggml/include -Isrc -Ivendor/nlohmann "codex/$source.cpp" -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml -lggml-base -o "build-cpu/bin/$source"
done
printf '[17,40037]\n' > codex/metadata-vocab-ids.json
build-cpu/bin/gguf-draft-vocab "$MODEL" "build-cpu/metadata-$BASHPID.gguf" codex/metadata-vocab-ids.json test > codex/draft-vocab-metadata.log 2>&1
build-cpu/bin/test-sampling > codex/draft-vocab-sampling.log 2>&1
build-cpu/bin/test-draft-vocab "$MODEL" cpu 3 identity > codex/draft-vocab-identity.log 2>&1
build-cpu/bin/test-sampling-prefilter "$MODEL" --draft-vocab-only > codex/draft-vocab-law.log 2>&1
build-cpu/bin/test-draft-vocab "$MODEL" cpu 9 deferred > codex/draft-vocab-subset.log 2>&1
build-cpu/bin/test-draft-vocab "$MODEL" tp 5 deferred > codex/draft-vocab-tp.log 2>&1
