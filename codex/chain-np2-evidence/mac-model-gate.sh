#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
while ! python3 - <<'PY'
import re, subprocess, sys
available = subprocess.check_output(['memory_pressure', '-Q'], text=True)
fraction = int(re.search(r'free percentage: (\d+)%', available)[1]) / 100
total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
processes = subprocess.check_output(['ps', '-axo', 'comm'], text=True).splitlines()
busy = any(p.rsplit('/', 1)[-1] in ('bw-np2-driver', 'chain-cpu', 'fold-cpu', 'outproj-cpu') for p in processes)
if busy or total * fraction < 10 * 1024**3:
    print('WAIT: another model process or less than 10 GiB available:', available.strip(), flush=True)
    sys.exit(1)
PY
do
    sleep 20
done
exec nice -n 19 "$@"
