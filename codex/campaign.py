#!/usr/bin/env python3
"""Adapt the read-only eight-task runner to local models and an alternate exec name."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

from weekend_common import ASSETS, COMMIT, FROZEN, ROOT, configure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--sha", required=True)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--deadline-epoch", required=True, type=float)
    args = p.parse_args()
    h, s, _ = configure(args.arm, args.model, args.sha)
    original_spawn, original_acquire, original_model = h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model

    def spawn(self, command, log, env=None):
        if Path(command[0]).name in ("curl", "wget"):
            raise h.StopCampaign("weekend_download_forbidden")
        if command[0] == str(FROZEN / "build/bin/llama-server"):
            command = [str(FROZEN / "build/bin/weekend-srv"), *command[1:]]
            launch = Path(log).with_name("launch.json")
            data = h.read_sealed(launch)
            data["executed_command"] = command
            h.sealed_json(launch, data)
        return original_spawn(self, command, log, env)

    def ensure_model(self, model_id):
        model = next(m for m in self.manifest["models"] if m["id"] == model_id)
        if not Path(model["path"]).is_file():
            raise h.StopCampaign("weekend_local_model_missing")
        return original_model(self, model_id)

    def acquire(self):
        # Keep cross-runner locks without creating or modifying files outside ROOT.
        runtime = Path(f"/run/user/{os.getuid()}")
        for name in ("qwen-hillclimb.lock", "qwen-local-server.lock"):
            handle = (runtime / name).open("r")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            self.lockfiles.append(handle)
        return original_acquire(self)

    h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model = spawn, acquire, ensure_model
    os.environ["XDG_RUNTIME_DIR"] = str(ROOT / "runtime")
    (ROOT / "runtime").mkdir(exist_ok=True)
    def interrupted(*_):
        raise h.StopCampaign("weekend_signal")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    sys.argv = [str(s.__file__), "--campaign", str(args.campaign), "--assets", str(ASSETS),
                "--deadline-epoch", str(args.deadline_epoch)]
    s.main()
    result = json.loads((args.campaign / "SUMMARY.json").read_text())
    return 0 if result["state"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
