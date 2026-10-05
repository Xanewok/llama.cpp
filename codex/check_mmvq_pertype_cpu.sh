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
cmake --build build-cpu --target test-mmvq-params -j8
ctest --test-dir build-cpu -R '^test-mmvq-params$' --output-on-failure -j1
mkdir -p build-cpu/mmvq-pertype-negative
python3 - <<'CONTROL'
from pathlib import Path
import subprocess
p = Path('build-cpu/mmvq-pertype-negative')
header = Path('ggml/src/ggml-cuda/mmvq-params.h').read_text()
for name, old, new in (
    ('warps', 'return ncols_dst == 5 ? 2 : 1;', 'return 2;'),
    ('rows', 'return type == GGML_TYPE_Q6_K && ncols_dst == 8 ? 6 : 4;', 'return 4;'),
    ('tail', 'return nrows / rows_per_block;', 'return nrows / rows_per_block + (nrows % rows_per_block != 0);'),
):
    assert old in header
    (p / f'{name}.h').write_text(header.replace(old, new))
    test = Path('tests/test-mmvq-params.cpp').read_text().replace('../ggml/src/ggml-cuda/mmvq-params.h', f'{name}.h')
    (p / f'{name}.cpp').write_text(test)
CONTROL
ulimit -c 0
for name in warps rows tail; do
    c++ -std=c++17 -O2 -Iggml/include "build-cpu/mmvq-pertype-negative/$name.cpp" -Lbuild-cpu/bin -lggml-base -Wl,-rpath,"$PWD/build-cpu/bin" -o "build-cpu/mmvq-pertype-negative/$name"
done
python3 - <<'CONTROL'
import subprocess
for name in ('warps', 'rows', 'tail'):
    result = subprocess.run([f'build-cpu/mmvq-pertype-negative/{name}'], capture_output=True)
    assert result.returncode == -6, (name, result.returncode)
    print(f'M: {name} negative control rejected')
CONTROL
PYTHONDONTWRITEBYTECODE=1 python3 codex/check_mmvq_pertype_tables.py
PYTHONDONTWRITEBYTECODE=1 python3 codex/pertype-probe/probe.py selftest
