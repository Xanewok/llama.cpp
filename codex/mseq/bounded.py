"""Bound a detached Mac payload, including configure, build and all model runs."""
import os
import signal
import subprocess
import sys

p = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    rc = p.wait(timeout=int(sys.argv[1]))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    rc = 124
sys.exit(rc)
