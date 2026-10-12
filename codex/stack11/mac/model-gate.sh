#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
t0=$SECONDS
while ! python3 - <<'PY'
import re, subprocess, sys
memory = subprocess.check_output(['memory_pressure', '-Q'], text=True)
fraction = int(re.search(r'free percentage: (\d+)%', memory)[1]) / 100
total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
models = {'bw-cand-driver', 'bw-np2-driver', 'chain-cpu', 'fold-cpu', 'outproj-cpu', 'test-llama-archs', 'test-llama-archs-mutant', 'test-recurrent-state-rollback', 'llama-completion', 'llama-cli', 'llama-server', 'llama-perplexity'}
busy = any(p.rsplit('/', 1)[-1] in models for p in subprocess.check_output(['ps', '-axo', 'comm'], text=True).splitlines())
if busy or total * fraction < 4 * 1024**3:
    print('WAIT: another model process or less than 4 GiB available:', memory.strip(), flush=True)
    sys.exit(1)
PY
do
    (( SECONDS - t0 < 600 )) || { echo "FAIL Mac RAM/model wait exceeded 10 min"; exit 75; }
    sleep 1
done
exec nice -n 19 python3 - "$@" <<'PY'
import os, signal, subprocess, sys
job = subprocess.Popen(sys.argv[1:], start_new_session=True)
try:
    sys.exit(job.wait(timeout=600))
except subprocess.TimeoutExpired:
    os.killpg(job.pid, signal.SIGTERM)
    try:
        job.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGKILL)
        job.wait()
    print('FAIL model gate exceeded 10 min', flush=True)
    sys.exit(124)
PY
