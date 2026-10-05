#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
mkdir -p build-tmp
export TMPDIR="$PWD/build-tmp"
cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build build-cpu --target test-mmvq-params test-backend-ops -j8
ctest --test-dir build-cpu -R '^test-mmvq-params$' --output-on-failure -j1
mkdir -p build-cpu/mmvq-negative
python3 - <<'CONTROL'
from pathlib import Path
p = Path('build-cpu/mmvq-negative')
header = Path('ggml/src/ggml-cuda/mmvq-params.h').read_text()
header = header.replace('return nrows / rows_per_block;', 'return nrows / rows_per_block + (nrows % rows_per_block != 0);')
assert header != Path('ggml/src/ggml-cuda/mmvq-params.h').read_text()
(p / 'mmvq-params.h').write_text(header)
test = Path('tests/test-mmvq-params.cpp').read_text().replace('../ggml/src/ggml-cuda/mmvq-params.h', 'mmvq-params.h')
(p / 'test.cpp').write_text(test)
CONTROL
c++ -std=c++17 -O2 -Iggml/include build-cpu/mmvq-negative/test.cpp -Lbuild-cpu/bin -lggml-base -Wl,-rpath,"$PWD/build-cpu/bin" -o build-cpu/mmvq-negative/test
ulimit -c 0
python3 - <<'CONTROL'
import subprocess
result = subprocess.run(['build-cpu/mmvq-negative/test'], capture_output=True)
assert result.returncode == -6, result.returncode
print('M: negative control rejected the old rounded-up grid')
CONTROL
PYTHONDONTWRITEBYTECODE=1 python3 codex/w2r4-general-probe/probe.py selftest
