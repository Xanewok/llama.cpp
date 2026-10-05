#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
ulimit -c 0
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build-tmp"
mkdir -p "$TMPDIR" codex/logs
cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON > codex/logs/cpu-configure.log 2>&1
cmake --build build-cpu --target ggml -j8 > codex/logs/cpu-build.log 2>&1
python3 codex/test_q8_dispatch.py > codex/q8-dispatch-results.txt 2>&1
c++ -O3 -DNDEBUG -std=c++17 -Iggml/include -Iggml/src codex/draft-attn-cpu.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml -lggml-base -lggml-cpu -pthread -o build-cpu/bin/q8-attn-cpu
build-cpu/bin/q8-attn-cpu > codex/cpu-q8-attn-results.txt
python3 codex/draft-attn-probe/probe.py gen --out build-tmp/probe-cases
c++ -O1 -DNDEBUG -DN_THREADS=8 -std=c++17 -Iggml/include codex/fa-case-schema.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml -lggml-base -lggml-cpu -pthread -o build-cpu/bin/q8-fa-schema
build-cpu/bin/q8-fa-schema build-tmp/probe-cases/cases.txt > codex/fa-case-schema-results.txt
build-cpu/bin/q8-fa-schema build-tmp/probe-cases/controls.txt > codex/fa-controls-schema-results.txt
python3 -m unittest discover -s codex/draft-attn-probe -p test_probe.py > codex/probe-test-results.txt 2>&1
