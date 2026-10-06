#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export LD_PRELOAD="$PWD/build-cpu/bin/row0-mutant.so:$PWD/build-cpu/bin/backend-thread-cap.so"
export LD_LIBRARY_PATH="$PWD/build-cpu/bin"
set +e
/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate python3 codex/timed_cpu.py codex/logs/negative-row0.json \
    build-cpu/bin/gdn-backend-cpu test -b CPU -o GATED_DELTA_NET_INDEXED -j 1 > codex/logs/negative-row0.log 2>&1
rc=$?
set -e
if [[ $rc == 0 ]]; then
    echo 'FAIL: row-0 mutant escaped the CPU equivalence check' >&2
    exit 1
fi
if ! rg -q 'ROW0_MUTANT active' codex/logs/negative-row0.log || ! rg -q ': .*FAIL' codex/logs/negative-row0.log; then
    echo 'FAIL: mutant did not reach a failing test case' >&2
    exit 1
fi
printf 'row-0 negative control: PASS (mutant rc=%s)\n' "$rc"
