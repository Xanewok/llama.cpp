#!/usr/bin/env python3
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
path = Path(sys.argv[1])
command = sys.argv[2:]
is_model = Path(command[0]).name in {'test-recurrent-state-rollback', 'gdn-archs-cpu'} or any('reserve_cpu.py' in arg for arg in command)
if is_model and os.environ.get('GDN_MODEL_LOCK_HELD') != '1':
    with (Path(__file__).parent / 'cpu-model.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = dict(os.environ, GDN_MODEL_LOCK_HELD='1')
        # Recheck the queue and RAM after waiting for another model to finish.
        rc = subprocess.run(['/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate', sys.executable,
                             __file__, str(path), *command], env=env).returncode
        sys.exit(rc)
started = time.monotonic()
rc = subprocess.run(command).returncode
path.write_text(json.dumps({'command': command, 'rc': rc, 'wall_s_M': time.monotonic() - started}, indent=2))
sys.exit(rc)
