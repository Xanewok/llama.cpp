#!/usr/bin/env python3
"""One eight-task Codex campaign for one arm, via the weekend runner, or a CPU stub with the same on-disk layout.

Real mode wraps the shared runner in place (monkeypatches, no edits): each arm brings its own frozen engine root, commit,
executable (never named llama-server) and server environment; cross-runner locks kept, probe timeout and retry count
optional (canary only). Stub mode never touches a GPU or llama-server.
"""
import argparse
import fcntl
import json
import os
import random
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from ab_common import (ARMS, ASSETS, CANARY_TASK, CTX, EVAL, EXTRA, HERE, ITEM_KEY, KNOBS, MODEL, MODEL_BYTES, MODEL_ID, MODEL_SHA, RUNNER,
                       STACK3_MASK, TASKS, WEEKEND)


def configure(arm, tasks=None):
    spec = ARMS[arm]
    os.environ.update(SWIFT_ENGINE="frozen", SWIFT_LLAMA=str(spec["runner_root"]), SWIFT_COMMIT=spec["commit"], SWIFT_DRAFT_MAX=str(spec["k"]),
                      SWIFT_EXTRA=shlex.join(EXTRA), SWIFT_MODEL="swift15-iq4-xs")
    os.environ.pop("SWIFT_TASKS", None)
    sys.path.insert(0, str(EVAL))
    g = runpy.run_path(str(RUNNER), run_name="knob_real")
    h, s = g["h"], g["s"]
    if ITEM_KEY != "iq4":  # same injection the weekend runner used for its non-IQ4 model
        g["SWIFT15"][MODEL_ID] = (MODEL.name, MODEL_BYTES, MODEL_SHA)
        g["LOCAL_BUILDS"][MODEL_ID] = {"repo": "local:gguf-j " + MODEL.stem, "revision": MODEL_SHA}
        g["MODEL_DIRS"][MODEL_ID] = MODEL.parent
        s.MODEL_PATH = MODEL
        s.MODEL = {"id": MODEL_ID, "file": MODEL.name, "bytes": MODEL_BYTES, "sha256": MODEL_SHA, "local": str(MODEL), **g["LOCAL_BUILDS"][MODEL_ID]}
        h.TEMPLATE_OVERRIDE_MODELS.add(MODEL_ID)
        os.environ["SWIFT_MODEL"] = MODEL_ID
    base = s.queue
    s.queue = lambda: [{**job, "configs": [{**job["configs"][0], "p_min": spec["p_min"]}]} for job in base()]
    if tasks:
        s.LONG_TAIL_TASKS[:] = tasks
    jobs = s.queue()
    s.configure_globals()
    want = tasks or TASKS
    assert [j["task"] for j in jobs] == want or sorted(j["task"] for j in jobs) == sorted(want), [j["task"] for j in jobs]
    for job in jobs:
        cfg = job["configs"][0]
        assert (cfg["maximum"], cfg["p_min"], cfg["split"], cfg["context"], cfg["slots"], cfg["mode"]) == \
            (spec["k"], spec["p_min"], "50,50", CTX, 1, "tensor"), cfg
    assert "--spec-draft-n-max" not in h.SERVER_EXTRA and h.SERVER_EXTRA == EXTRA
    return h, s, jobs


def server_env(base, spec):
    """The arm's serving knobs on top of a scrubbed copy of the runner env: no inherited knob or library path reaches the server."""
    env = {k: v for k, v in dict(base).items() if k not in KNOBS and k != "LD_LIBRARY_PATH"}
    return env | spec["env"]


def install(h, args, spawn_hook=None):
    """Patch the shared runner in place: frozen server under the renamed binary, probe cap, retry count, lock policy."""
    original_spawn, original_acquire, original_model, original_init = h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model, h.initialize
    spec = ARMS[args.arm]

    def spawn(self, command, log, env=None):
        if Path(command[0]).name in ("curl", "wget"):
            raise h.StopCampaign("download_forbidden")
        if command[0] == str(spec["runner_binary"]):
            command = [str(spec["exe"]), *command[1:]]
            env = server_env(self.env if env is None else env, spec)
            launch = Path(log).with_name("launch.json")
            data = h.read_sealed(launch)
            data["executed_command"] = command
            data["executed_env"] = {k: env.get(k) for k in (*KNOBS, "LD_LIBRARY_PATH")}
            data["server_nice"] = os.getpriority(os.PRIO_PROCESS, 0)  # the child inherits it
            h.sealed_json(launch, data)
        if args.probe_timeout and "scripts.instruments.agentic_probe" in command and "--timeout" in command:
            command = list(command)
            command[command.index("--timeout") + 1] = str(args.probe_timeout)
        if spawn_hook:
            return spawn_hook(command, log)
        return original_spawn(self, command, log, env)

    def ensure_model(self, model_id):
        model = next(m for m in self.manifest["models"] if m["id"] == model_id)
        if not Path(model["path"]).is_file():
            raise h.StopCampaign("local_model_missing")
        return original_model(self, model_id)

    def acquire(self):
        runtime = Path(f"/run/user/{os.getuid()}")
        for name in ("qwen-hillclimb.lock", "qwen-local-server.lock"):
            if not (runtime / name).exists():
                continue
            handle = (runtime / name).open("r")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            self.lockfiles.append(handle)
        return original_acquire(self)

    def initialize(*a, **k):
        if args.max_attempts:
            k["max_attempts"] = args.max_attempts
        return original_init(*a, **k)

    h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model, h.initialize = spawn, acquire, ensure_model, initialize
    root = spec["runner_root"]

    class Subprocess:
        """The runner checks `git -C <engine> rev-parse HEAD`; a runroot without its own .git answers from its SOURCE commit instead of the enclosing repo's HEAD."""

        def __getattr__(self, name):
            return getattr(subprocess, name)

        def check_output(self, cmd, *a, **k):
            if list(cmd) == ["git", "-C", str(root), "rev-parse", "HEAD"] and not (root / ".git").exists():
                source = (root / "SOURCE").read_text().split()[0]
                cmd = ["git", "-C", str(STACK3_MASK), "rev-parse", f"{source}^{{commit}}"]
            return subprocess.check_output(cmd, *a, **k)

    h.subprocess = Subprocess()


def real(args):
    tasks = args.tasks.split(",") if args.tasks else None
    h, s, _ = configure(args.arm, tasks)
    install(h, args)
    runtime = args.campaign.parent / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_RUNTIME_DIR"] = str(runtime)

    def interrupted(*_):
        raise h.StopCampaign("signal")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    sys.argv = [str(s.__file__), "--campaign", str(args.campaign), "--assets", str(ASSETS), "--deadline-epoch", str(args.deadline_epoch)]
    s.main()
    summary = json.loads((args.campaign / "SUMMARY.json").read_text())
    return 0 if summary["state"] == "complete" else 1


class Captured(Exception):
    pass


def capture(args):
    """CPU proof: drive the real Runner.attempt up to the server spawn and keep the launch.json it seals. No server, no GPU."""
    fixtures = sorted((WEEKEND / "results").glob(f"*/campaign-*/artifacts/{MODEL_ID}.json"))
    if not fixtures:
        raise SystemExit("no model metadata fixture (weekend S5 artifacts) for the template override")
    h, s, jobs = configure(args.arm, [CANARY_TASK])
    install(h, args, spawn_hook=lambda command, log: (_ for _ in ()).throw(Captured()))
    campaign = args.campaign
    h.initialize(campaign, EVAL / ".cache/hillclimb-models", profiles=jobs, max_attempts=1, agentic_assets=ASSETS)
    shutil.copy(fixtures[0], campaign / "artifacts" / f"{MODEL_ID}.json")
    gpu = {"temperature.gpu": 40.0, "memory.free": 15000.0, "power.limit": 150.0, "power.draw": 10.0, "utilization.gpu": 0.0,
           "pcie.link.gen.current": 3.0, "pcie.link.width.current": 4.0}
    h.telemetry = lambda: {"time": time.time(), "gpus": {d: dict(gpu) for d in h.DEVICES}, "ram_available_mib": 20000.0}
    h.endpoint_free = lambda port: None
    runner = s.agentic.AgenticRunner(campaign, 1, time.time() + 600)
    runner.ensure_model = lambda model_id: next(m for m in runner.manifest["models"] if m["id"] == model_id)
    job = runner.manifest["queue"][0]
    config = {"devices": h.DEVICES, **job["configs"][0], "workload": "agentic"}
    try:
        runner.attempt(job["id"], runner.ensure_model(job["model"]), config, 0, task=job["task"])
    except h.StopCampaign:
        pass
    launches = sorted(campaign.glob("attempts/*/launch.json"))
    if not launches:
        raise SystemExit("runner did not reach the server launch")
    print(json.dumps(h.read_sealed(launches[0])))
    return 0


# ---------------------------------------------------------------- stub

# (mean output tokens, tool-call responses) per task, shaped like the S5 control campaigns
PROFILE = {"rust/doubly-linked-list": (26000, 11), "rust/book-store": (11000, 9), "rust/react": (14000, 9), "rust/alphametics": (9000, 8),
           "rust/forth": (12000, 10), "rust/xorcism": (19000, 10), "rust/nucleotide-codons": (5500, 6), "javascript/complex-numbers": (6500, 8)}


def stub_command(arm, port, model=MODEL):
    spec = ARMS[arm]
    return [str(spec["runner_binary"]), "-m", str(model), "--alias", "local-benchmark", "-c", str(CTX), "-np", "1",
            "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-fa", "on", "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(spec["k"]), "--spec-draft-n-min", "0", "--spec-draft-p-min", str(spec["p_min"]),
            "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "-sm", "tensor", "--tensor-split", "50,50",
            "-b", "2048", "-ub", "128", "--host", "127.0.0.1", "--port", str(port), "--metrics",
            "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05", *EXTRA]


def stub_env():
    """Mirror the real spawn: the stub runs under the same arm environment, with the STUB_* defect switches passed through."""
    arm = os.environ["AB_STUB_ARM"]
    env = server_env(os.environ, ARMS[arm]) | {k: v for k, v in os.environ.items() if k.startswith("STUB_")} | {"STUB_ARM": arm}
    if arm == "B" and env.get("STUB_B_NO_STAGING") == "1":
        env.pop("GGML_SCHED_INPUT_STAGING", None)
    return env


def http(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def stub(args):
    sys.path.insert(0, str(EVAL))
    from scripts.instruments import emit
    spec = ARMS[args.arm]
    campaign = args.campaign
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))  # unwind through the server's finally
    (campaign / "attempts").mkdir(parents=True)
    tasks = args.tasks.split(",") if args.tasks else TASKS
    queue = [{"id": "solve-" + t.replace("/", "-"), "kind": "profile", "task": t, "workload": "agentic", "targets": [0],
              "configs": [{"context": CTX, "slots": 1, "mode": "tensor", "split": "50,50", "ubatch": 128, "maximum": spec["k"],
                           "p_min": spec["p_min"]}]} for t in tasks]
    defect_arms = os.environ.get("STUB_ARMS", "B")
    other = ARMS["A" if args.arm == "B" else "B"]
    wrong = other if os.environ.get("STUB_WRONG_ROOT") == "1" and args.arm in defect_arms else spec
    (campaign / "manifest.json").write_text(json.dumps({"engine_commit": wrong["commit"], "binary": str(wrong["runner_binary"]), "queue": queue,
                                                        "binary_pins": {str(Path(wrong["runner_binary"]).resolve()): "stub"},
                                                        "server_extra": EXTRA, "port": args.port, "stub": True}))
    rng = random.Random(f"{campaign.name}-{os.environ.get('STUB_SEED', '1')}")
    fail_task, runaway_task = os.environ.get("STUB_FAIL_TASK"), os.environ.get("STUB_RUNAWAY_TASK")
    stopped = False
    total = dict(wall=0.0, tokens=0.0, decode_s=0.0, solved=0, done=0, accepted=0.0, draft_tokens=0.0)
    for job in queue:
        task = job["task"]
        if time.time() >= args.deadline_epoch:
            stopped = True
            break
        key = sha_key(job["id"], args.arm)
        out = campaign / "attempts" / f"{key}-01"
        out.mkdir()
        command = stub_command(args.arm, args.port)
        executed = [str(HERE / "stub_srv.py"), *command[1:]]
        (out / "launch.json").write_text(json.dumps({"command": command, "executed_command": executed, "stub_exe": str((other if os.environ.get("STUB_WRONG_EXE") == "1" and args.arm in defect_arms else spec)["exe"]),
                                                     "target_kv": "q8_0/q8_0",
                                                     "model_sha256": "stub", "server_nice": os.getpriority(os.PRIO_PROCESS, 0),
                                                     "executed_env": {k: stub_env().get(k) for k in (*KNOBS, "LD_LIBRARY_PATH")}}))
        log = (out / "server.log").open("w")
        server = subprocess.Popen(executed, stdout=log, stderr=subprocess.STDOUT, env=stub_env())
        try:
            base = f"http://127.0.0.1:{args.port}"
            for _ in range(100):
                try:
                    if json.loads(http(base + "/health", timeout=1)).get("status") == "ok":
                        break
                except OSError:
                    time.sleep(0.1)
            (out / "props.json").write_text(http(base + "/props"))
            before = emit.scrape_metrics(base)
            (out / "metrics-before.json").write_text(json.dumps(before))
            mean, calls = PROFILE.get(task, (8000, 8))
            n = int(mean * rng.uniform(0.6, 1.5))
            sizes = [n // calls] * calls
            if runaway_task == task:
                sizes = [200, 41000, 150]
            for i, size in enumerate(sizes):
                http(base + "/stub/generate", {"n": size, "seed": rng.randrange(1 << 30), "top1_mean": 0.6}, timeout=120)
            time.sleep(float(os.environ.get("STUB_TASK_SLEEP", "0")))
            after = emit.scrape_metrics(base)
            (out / "metrics-after.json").write_text(json.dumps(after))
            delta = emit.metrics_delta(before, after)
        finally:
            server.terminate()
            server.wait(timeout=10)
            log.close()
        wall = delta["decode_seconds"] + 30 + delta["prefill_seconds"]
        rel = f"attempts/{key}-01"
        if fail_task == task:
            result = {"job": job["id"], "task": task, "status": "failed", "reason": "probe_failed", "attempt": rel}
        else:
            probe = {"task_id": task, "complete": True, "task_success": True, "timed_out": False, "verified_wall_seconds": wall,
                     "decode_seconds": delta["decode_seconds"], "prefill_seconds": delta["prefill_seconds"], "metrics_delta": delta,
                     "mtp_live": True}
            result = {"job": job["id"], "task": task, "status": "passed", "probe": probe, "attempt": rel,
                      "minimum_free_mib": {"stub0": 4900, "stub1": 5500}}
            total["wall"] += wall
            total["tokens"] += delta["decode_tokens"]
            total["decode_s"] += delta["decode_seconds"]
            total["solved"] += 1
            total["accepted"] += delta["draft_accepted_tokens"]
            total["draft_tokens"] += delta["draft_tokens"]
        total["done"] += 1
        (out / "result.json").write_text(json.dumps(result))
        (campaign / f"job-{job['id']}.json").write_text(json.dumps(result))
    state = "stopped" if stopped else "complete"
    (campaign / "state.json").write_text(json.dumps({"status": state, "reason": "budget_expired" if stopped else None}))
    (campaign / "SUMMARY.json").write_text(json.dumps({
        "state": state, "tasks": total["done"], "solved": total["solved"], "wall_seconds": total["wall"], "output_tokens": total["tokens"],
        "decode_seconds": total["decode_s"], "draft_acceptance": total["accepted"] / total["draft_tokens"] if total["draft_tokens"] else None}))
    return 0 if state == "complete" else 1


def sha_key(*parts):
    import hashlib
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--deadline-epoch", required=True, type=float)
    p.add_argument("--tasks", help="comma list; default all eight")
    p.add_argument("--probe-timeout", type=int, default=0, help="per-task Codex cap in seconds (canary)")
    p.add_argument("--max-attempts", type=int, default=0, help="runner retry count override (canary: 1)")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--capture", action="store_true", help="print the launch.json the real runner seals for this arm, without starting a server")
    p.add_argument("--check", action="store_true", help="build and validate the arm's queue with the real runner modules, then exit")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()
    args.campaign = args.campaign.resolve()
    os.environ["AB_STUB_ARM"] = args.arm
    if args.capture:
        return capture(args)
    if args.check:
        h, _, jobs = configure(args.arm, args.tasks.split(",") if args.tasks else None)
        install(h, args)
        root = str(ARMS[args.arm]["runner_root"])
        seen = h.subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        if seen != ARMS[args.arm]["commit"] or h.COMMIT != seen:
            raise SystemExit(f"engine commit check: runner would see {seen}, arm wants {ARMS[args.arm]['commit']}")
        print(json.dumps({"arm": args.arm, "jobs": len(jobs), "config": jobs[0]["configs"][0]}))
        return 0
    return stub(args) if args.stub else real(args)


if __name__ == "__main__":
    sys.exit(main())
