"""Pause only this CPU command's process group if gpuq starts an item after the launch gate."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
snapshot = Path(os.environ.get("GPUQ_HOME", "~/.local/state/gpuq")).expanduser() / "status.json"


def running():
    try:
        return any(item["state"] == "running" for item in json.loads(snapshot.read_text())["items"])
    except (OSError, ValueError, KeyError):
        return True


def memory_ready():
    minimum = int(os.environ.get("CPUGATE_MIN_GIB", "10"))*1024**3
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])*1024 >= minimum
    return False


while running() or not memory_ready():
    time.sleep(0.5)
child = subprocess.Popen(sys.argv[1:], start_new_session=True)
paused = False


def interrupted(signum, _):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    while child.poll() is None:
        active = running()
        if active != paused:
            try:
                os.killpg(child.pid, signal.SIGSTOP if active else signal.SIGCONT)
            except ProcessLookupError:
                child.wait()
                break
            paused = active
            print(f"CPU gate: {'paused' if paused else 'resumed'} own process group {child.pid}", flush=True)
        time.sleep(0.5)
    sys.exit(child.returncode)
finally:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        if paused:
            os.killpg(child.pid, signal.SIGCONT)
        child.wait()
