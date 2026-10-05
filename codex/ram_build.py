#!/usr/bin/env python3
"""Pause the build process group while MemAvailable is below 10 GiB."""
import os
import signal
import subprocess
import sys
import time


def available():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')


def main():
    floor = 10 * 1024**3
    while available() < floor:
        time.sleep(1)
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    paused = False
    try:
        while child.poll() is None:
            low = available() < floor
            if low != paused:
                os.killpg(child.pid, signal.SIGSTOP if low else signal.SIGCONT)
                paused = low
                print('RAM wait' if low else 'RAM resumed', flush=True)
            time.sleep(0.25)
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()


if __name__ == '__main__':
    sys.exit(main())
