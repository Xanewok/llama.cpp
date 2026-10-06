#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CCACHE_READONLY=1 CCACHE_NOSTATS=1 CCACHE_TEMPDIR="$PWD/build-tmp/ccache"
gate() { bash codex/host-gate.sh "$@"; }
gate python3 codex/host-cpu-identity.py --resume > codex/host-results/cpu-identity-resume.log 2>&1
gate python3 codex/host-draft-identity.py > codex/host-results/draft-identity.log 2>&1
gate python3 codex/host-input-count.py > codex/host-results/input-count.log 2>&1
gate python3 codex/host-attrib.py > codex/host-results/phase-attrib.log 2>&1
gate python3 codex/host-parser-check.py > codex/host-results/parser-check.log 2>&1
gate python3 codex/host-negative.py > codex/host-results/negative-controls.log 2>&1
