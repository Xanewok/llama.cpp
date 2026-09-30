#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
nice -n 19 python3 codex/law_contract.py
nice -n 19 python3 codex/check_harness.py
nice -n 19 python3 codex/evidence.py
nice -n 19 python3 -c 'import ast,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path("codex").glob("*.py")]'
nice -n 19 bash -n codex/gpu_ab.sh codex/run_cpu.sh
nice -n 19 bash codex/gpu_ab.sh
