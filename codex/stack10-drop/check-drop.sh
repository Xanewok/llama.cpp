#!/usr/bin/env bash
# usage: check-drop.sh — host syntax check of the host-compilable files a stack10 component touches, in the current tree.
# CUDA sources cannot be compiled on this host; leftover conflict markers and the C++ files are the compile check.
set -euo pipefail
export GIT_NO_LAZY_FETCH=1
n=$({ git grep -l -E '^(<<<<<<<|>>>>>>>) ' -- ggml src tests common || true; } | wc -l)
[[ $n == 0 ]] || { echo "conflict markers in $n files" >&2; exit 1; }
inc=(-I"$(dirname "$0")/gen" -Iinclude -Iggml/include -Iggml/src -Icommon -Isrc -Ivendor -Itests -Iggml/src/ggml-cuda)
rc=0
for f in tests/test-backend-ops.cpp tests/test-fattn-kv-max-cache.cpp tests/test-meta-input-stage.cpp tests/test-mmvq-params.cpp \
         tests/test-topk-batched.cpp tests/test-recurrent-state-rollback.cpp src/llama-context.cpp src/llama-graph.cpp \
         src/models/delta-net-base.cpp src/models/qwen35.cpp ggml/src/ggml-backend-meta.cpp ggml/src/ggml-cpu/ops.cpp ggml/src/ggml.c; do
    [[ -f $f ]] || continue
    case $f in *.c) cc=(gcc -std=c11 -D_GNU_SOURCE);; *) cc=(g++ -std=c++17);; esac
    if nice -n 19 taskset -c 6-11,18-23 "${cc[@]}" -fsyntax-only -w "${inc[@]}" "$f" > "${TMPDIR:-/tmp}/check-drop.err" 2>&1; then
        echo "ok   $f"
    else
        echo "FAIL $f: $(grep -a -m1 'error' "${TMPDIR:-/tmp}/check-drop.err" || true)"; rc=1
    fi
done
exit $rc
