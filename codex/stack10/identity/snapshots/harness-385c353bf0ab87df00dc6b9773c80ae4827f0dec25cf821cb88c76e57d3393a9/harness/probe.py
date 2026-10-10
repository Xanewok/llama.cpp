"""Probe: one fresh boot of each planned arm, then the experiment's own driver command against it.

The driver owns the measurement; this owns the boot contract (frozen bytes, argv, env, nice) and the exit-code map.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from . import depth, lib
from .lib import Fail


def driver_argv(x, out, port):
    ctx = {"port": port, "out": out, "base_url": f"http://127.0.0.1:{port}", "model": x["model"]["path"]}
    return [lib._expand(token, ctx) for token in x["probe"]["cmd"]]


def capture(x, out, port):
    return depth.capture(x, out, port)


def execute(x, out, port, stub=False):
    out = Path(out)
    if out.exists() and any(p.name != "console.log" for p in out.iterdir()):
        raise Fail(f"output dir {out} is not fresh", "deterministic")
    out.mkdir(parents=True, exist_ok=True)
    for arm in x["arms"].values():
        lib.check_frozen(arm)
    lib.check_model(x["model"])
    if not stub:
        lib.verify_nice(os.getpriority(os.PRIO_PROCESS, 0))
    lib.event("measure_start")
    for i, name in enumerate(x["plan"]):
        arm = x["arms"][name]
        if i and x.get("devices") and not stub:
            lib.wait_gpu_idle(x["devices"])
        dest = out / f"{i}-{name}"
        dest.mkdir()
        command, env = depth.boot(x, arm, port, dest, canary=False, stub=stub)
        (dest / "command.json").write_text(json.dumps(command))
        (dest / "env.json").write_text(json.dumps(lib.env_record(env, depth.knobs(x))))
        server, began = lib.Server(command, env, dest / "server.log"), time.time()
        try:
            server.start()
            base = f"http://127.0.0.1:{port}"
            server.wait_ready(base, x["server"].get("boot_cap_s", 180))
            lib.verify_props(lib.http_json(base + "/props"), int(lib.flag_values(command, "-c")[0]))
            driver = driver_argv(x, dest, port)
            with (dest / "driver.log").open("ab") as stream:
                rc = subprocess.run(driver, stdout=stream, stderr=subprocess.STDOUT).returncode
            if rc:
                raise Fail(f"driver {driver[0]} rc {rc} on arm {name}", lib.EXIT.get(rc, "deterministic"))
            lib.verify_log_patterns(server.log.read_text(errors="replace"), arm["canary_log"], f"arm {name}")
        finally:
            server.stop(grace=30)
            lib.record_journal(began, dest / "journal.txt")
        lib.log(out, f"boot {i} {name}: driver ok in {time.time() - began:.0f}s")
    return 0
