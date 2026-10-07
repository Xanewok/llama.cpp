#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
while :; do
    gpuq status --brief
    if gpuq status --json | python3 -c '
import json,sys
d = json.load(sys.stdin)
blocked = [i for i in d["items"] if i.get("state") in ("running", "ready") and not i["id"].startswith(("bw-", "build-window-")) and (i.get("kind") == "measure" or any(s in i["id"] for s in ("depth", "perf", "probe")))]
if blocked:
    print("Waiting for measure items:", ", ".join(i["id"] for i in blocked))
with open("/proc/meminfo") as f:
    available = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:"))
low_ram = available < 10 * 1024 ** 2
if low_ram:
    print("Waiting for MemAvailable >= 10 GiB")
sys.exit(bool(blocked) or low_ram)'; then
        break
    fi
    sleep 30
done
exec "$@"
