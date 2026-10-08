import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1')
args = sys.argv[1:]
name = Path(args[0]).name
model = name not in ('cmake', 'c++', 'test-sampling', 'test-alloc', 'chain-capacity') and '--seq-rm' not in args
lock = Path(__file__).resolve().parents[1] / 'model.lock'
owned = False
child = None


def stop(signum, frame):
    if child is not None and child.poll() is None:
        child.terminate()
        child.wait()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    while True:
        if model and not owned:
            try:
                lock.mkdir()
                (lock / 'pid').write_text(str(os.getpid()))
                owned = True
            except FileExistsError:
                print('WAIT: task model lock', flush=True)
                time.sleep(20)
                continue
        pressure = subprocess.check_output(['memory_pressure', '-Q'], text=True)
        fraction = int(re.search(r'free percentage: (\d+)%', pressure)[1]) / 100
        total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
        names = subprocess.check_output(['ps', '-axo', 'comm'], text=True).splitlines()
        busy = model and any(Path(p).name in (
            'bw-np2-driver', 'chain-cpu', 'fold-cpu', 'outproj-cpu', 'test-llama-archs',
            'test-sampling-prefilter', 'test-backend-sampler', 'upstream-model-probe',
            'test-recurrent-state-rollback', 'test-grammar-backend-sampling', 'bw-cand-driver',
            'llama-perplexity') for p in names)
        if total * fraction >= 4 * 1024**3 and not busy:
            break
        print('WAIT: RAM or another CPU model:', pressure.strip(), flush=True)
        time.sleep(20)
    child = subprocess.Popen(['nice', '-n', '19', *args])
    sys.exit(child.wait())
finally:
    if owned:
        (lock / 'pid').unlink()
        lock.rmdir()
