import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
pkg = '21-mtp-draft-chain'
run = root / pkg
out = run / 'p21-np2-shape'
out.mkdir(exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PACK_LOCAL_APPLIED='1',
                  PACK_GATE_MIN_FREE_GIB='10', PACK_GATE_WAIT_SECONDS='600', OMP_NUM_THREADS='4')
started = time.monotonic()
with (out / 'gate.log').open('w') as log:
    child = subprocess.Popen(['nice', '-n', '19', 'bash', str(root / 'validation/model_tests.sh'), str(root), pkg],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = child.wait(timeout=3590)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        rc = 124
(out / 'gate.rc').write_text(f'{rc}\n')
(out / 'elapsed-seconds').write_text(f'{time.monotonic() - started:.3f}\n')
print(f'M: p21 gate rc={rc}; 75=resource wait skip, 124=wall budget')
sys.exit(rc)
