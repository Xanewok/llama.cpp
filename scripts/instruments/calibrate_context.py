"""Calibrate qualified Swift context windows with cold/warm per-card telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from scripts.instruments import hillclimb_probe as probe


SERVE = Path("/home/xanewok/.codex/skills/local-qwen-server/bin/serve.sh")
QUERY = "uuid,memory.used,memory.free,temperature.gpu,power.draw"


def gpu_sample():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + QUERY, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    rows = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            raise ValueError("invalid GPU telemetry row")
        rows[row[0]] = {
            "memory.used": float(row[1]), "memory.free": float(row[2]),
            "temperature.gpu": float(row[3]), "power.draw": float(row[4]),
        }
    if not rows:
        raise ValueError("nvidia-smi returned no GPUs")
    return {"time": time.time(), "gpus": rows}


class GpuSampler:
    def __init__(self, path, interval=0.5):
        self.path, self.interval = path, interval
        self.rows, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                self.rows.append(gpu_sample())
            except Exception as error:
                self.rows.append({"time": time.time(), "error": f"{type(error).__name__}: {error}"})
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)
        self.path.write_text(json.dumps(self.rows, indent=2) + "\n")


def service(serve, action, context=None):
    env = dict(os.environ, PROFILE="swift-pair")
    if context is not None:
        env["CTX"] = str(context)
    return subprocess.run([str(serve), action], env=env, text=True, check=action == "start")


def phase(url, task, target, seed, output, cache_prompt):
    output.mkdir(parents=True, exist_ok=False)
    before = gpu_sample()
    with GpuSampler(output / "gpu-telemetry.json") as sampler:
        result = probe.run_probe(url, task, target, 2048, 1, seed, output,
                                 cache_prompt=cache_prompt)
    after = gpu_sample()
    good = [row for row in sampler.rows if "gpus" in row]
    devices = sorted(before["gpus"])
    result = {
        "cache_prompt": cache_prompt, "requested_prompt_tokens": target,
        "max_output_tokens": 2048, "probe": result,
        "before": before, "after": after,
        "minimum_free_mib": {
            device: min((row["gpus"][device]["memory.free"] for row in good
                         if device in row["gpus"]), default=before["gpus"][device]["memory.free"])
            for device in devices
        },
        "peak_used_mib": {
            device: max((row["gpus"][device]["memory.used"] for row in good
                         if device in row["gpus"]), default=before["gpus"][device]["memory.used"])
            for device in devices
        },
        "peak_temperature_c": {
            device: max((row["gpus"][device]["temperature.gpu"] for row in good
                         if device in row["gpus"]), default=before["gpus"][device]["temperature.gpu"])
            for device in devices
        },
    }
    (output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--serve", type=Path, default=SERVE)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--windows", nargs="+", type=int, default=[71680, 90112, 102400])
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if any(window < 1 or window % 2048 for window in args.windows):
        parser.error("windows must be positive multiples of 2048")
    args.out.mkdir(parents=True, exist_ok=False)
    task = json.loads(args.tasks.read_text())["javascript/affine-cipher"]
    summary = {"windows": args.windows, "target_rule": "window - 2048", "phases": []}
    try:
        for window in args.windows:
            service(args.serve, "stop")
            service(args.serve, "start", window)
            loaded = gpu_sample()
            window_out = args.out / f"context-{window}"
            window_out.mkdir()
            (window_out / "loaded.json").write_text(json.dumps(loaded, indent=2) + "\n")
            phases = []
            target = window - 2048
            for name, cache_prompt in (("cold", True), ("warm", True)):
                phase_out = window_out / name
                try:
                    phases.append({"name": name, **phase(args.url, task, target, args.seed, phase_out, cache_prompt)})
                except Exception as error:
                    phase_out.mkdir(exist_ok=True)
                    failure = {"name": name, "status": "error", "error": f"{type(error).__name__}: {error}"}
                    (phase_out / "calibration.json").write_text(json.dumps(failure, indent=2) + "\n")
                    phases.append(failure)
            summary["phases"].append({"context": window, "loaded": loaded, "phases": phases})
            (window_out / "summary.json").write_text(json.dumps(summary["phases"][-1], indent=2) + "\n")
    finally:
        service(args.serve, "stop")
        service(args.serve, "start", 102400)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
