#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build-tmp"
mkdir -p "$TMPDIR" codex/logs
cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON > codex/logs/general-cpu-configure.log 2>&1
cmake --build build-cpu --target ggml -j8 > codex/logs/general-cpu-build.log 2>&1
python3 codex/test_q8_general_dispatch.py > codex/logs/general-stack-dispatch.log 2>&1
python3 codex/test_q8_general_loader.py > codex/logs/general-stack-loader.log 2>&1
c++ -O3 -DNDEBUG -std=c++17 -Iggml/include -Iggml/src codex/draft-attn-cpu.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml -lggml-base -lggml-cpu -pthread -o build-cpu/bin/q8-general-attn-cpu
build-cpu/bin/q8-general-attn-cpu > codex/q8-general-attn-cpu.txt
bash codex/q8-probe4/build_driver.sh > codex/logs/general-probe-build.log 2>&1
python3 codex/q8-probe4/probe.py gen --out build-tmp/q8-probe4/cases
c++ -O1 -DNDEBUG -DN_THREADS=8 -std=c++17 -Iggml/include codex/fa-case-schema.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml -lggml-base -lggml-cpu -pthread -o build-cpu/bin/q8-general-schema
build-cpu/bin/q8-general-schema build-tmp/q8-probe4/cases/cases.txt > codex/q8-general-case-schema.txt
build-cpu/bin/q8-general-schema build-tmp/q8-probe4/cases/controls.txt > codex/q8-general-controls-schema.txt
python3 -m unittest discover -s codex/q8-probe4 -p test_probe.py > codex/logs/general-probe-test.log 2>&1
