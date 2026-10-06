#!/usr/bin/env bash
# Parent-only gpuq entry. Usage: gpu_ab.sh np2-iq4|np2-j5q|np1-j5q --canary|--execute --out DIR [--port N]
# The generated .gpuq.toml items call the same harness directly and own the wall deadline.
set -euo pipefail
case "${1:-}" in
    np2-iq4|np2-j5q|np1-j5q) experiment="$(dirname "$0")/host-trims-$1.toml" ;;
    *) echo 'expected np2-iq4, np2-j5q or np1-j5q' >&2; exit 8 ;;
esac
shift
if [[ "${GPUQ_ACTIVE:-}" != 1 ]]; then
    echo 'run only through gpuq; this worker must not execute GPU workloads' >&2
    exit 8
fi
# harness.lib checks frozen hashes and library resolution, saves requests before gates,
# scopes the kernel journal to this boot, and stops the exact server PID.
exec bash /home/xanewok/repos/qwen38-eval/harness/run.sh depth "$experiment" "$@"
