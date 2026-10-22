#!/usr/bin/env bash
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 TMPDIR=/mnt/scratch/tmp-build
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/xanewok/repos/qwen38-eval
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
exec nice -n 19 python3 - "$@" <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

EVAL = Path('/home/xanewok/repos/qwen38-eval')
SOURCE = 'f26f264db1e52206e909fbd83d7d19a6c55cccf3'
BASE = '6ef25f572c544b705d3fe7a37c02978bc2d1aa56'
BF16_PASS = 'PASS: AllReduce residual RMSNorm cases=15 wire=BF16'
F32_PASS = 'PASS: AllReduce residual RMSNorm cases=15 wire=F32'
spec = importlib.util.spec_from_file_location('pkg101', EVAL / 'orchestration/gpuq/items/bw-pkg101-fork.py')
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)
base = pkg.base
base.NAME = 'ar-norm-fuse'
base.PURPOSE = 'Build package 101 + decode AllReduce residual RMSNorm fusion; gate before freezing ar-norm-fuse'
pkg.LIB_MARKERS['libggml-cuda.so'].append('fused AllReduce + ADD + RMS_NORM')
pkg.GRAPH_ENV['GGML_CUDA_GRAPH_DEBUG'] = '1'

original_graph_gates = pkg.graph_gates


def norm_gates():
    original_graph_gates()
    output = Path(os.environ['OUT'])
    text = (output / 'ar-graph.log').read_text(errors='replace')
    base.require(text.count(BF16_PASS + '\n') == 1, 'BF16 norm cases must execute once in the package 101 graph gate')
    binary = Path('/mnt/scratch/work') / output.name / 'build-cuda/bin'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'NCCL_'))}
    env.update(GIT_NO_LAZY_FETCH='1', TMPDIR='/mnt/scratch/tmp-build', CUDA_HOME='/usr/local/cuda-13.2',
               LD_LIBRARY_PATH=str(binary), GGML_CUDA_ALLREDUCE='internal', GGML_CUDA_AR_P2P='1',
               GGML_CUDA_P2P='1', GGML_CUDA_AR_COPY_THRESHOLD='2097152', GGML_CUDA_AR_BF16_THRESHOLD='0',
               GGML_CUDA_GRAPH_DEBUG='1')
    with (output / 'ar-norm-f32.log').open('w') as stream:
        rc = subprocess.run([base.ROOT / 'bin/cpugate', binary / 'test-ar-p2p', '--norm-only'],
                            cwd=base.ROOT / 'wt' / output.name, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, timeout=900).returncode
    text = (output / 'ar-norm-f32.log').read_text(errors='replace')
    (output / 'ar-norm-gates.json').write_text(json.dumps({'source': SOURCE, 'bf16': 0, 'f32': rc}, indent=2) + '\n')
    base.require(rc == 0 and text.count(F32_PASS + '\n') == 1 and 'SKIP' not in text,
                 'native F32 norm gate must pass all 15 cases without SKIP')
    base.require(pkg.P2P_LINE in text and not any(x in text for x in pkg.DECLINED), 'native F32 norm gate must use P2P staging')
    print('PASS: AllReduce residual RMSNorm BF16 and F32 gates', flush=True)


def marker_controls():
    pkg.selftest_markers()
    base.marker_selftest()
    print('PASS: inherited marker controls; fusion engagement and numeric gates run in the build window')


pkg.graph_gates = norm_gates
base.selftest = marker_controls
if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] not in ('--preflight', '--canary', '--selftest')):
    sys.exit('usage: ar-norm-fuse.sh [--preflight | --canary | --selftest]')
subprocess.run(['git', '-C', base.REPO, 'merge-base', '--is-ancestor', BASE, SOURCE], check=True)
with tempfile.TemporaryDirectory(prefix='ar-norm-recipe-') as tmp:
    base.RECIPE = Path(tmp) / 'recipe.json'
    base.RECIPE.write_text(json.dumps({'base': BASE, 'tip': SOURCE, 'steps': []}))
    # Inherited configure/build use CUDA 13.2, buildgate, -j4 and NCCL=OFF; runtime gates precede wt freeze.
    base.main()
PY
