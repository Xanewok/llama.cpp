#!/usr/bin/env bash
# Parent-only. Default dry run performs no network, GPU queries, or server launches.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec nice -n 19 python3 -B "$HERE/gpu_ab.py" "$@"
