"""Campaign A/B: per arm, one Codex task campaign through the shared hillclimb runner (monkeypatched in place, never edited),
verified against the arm's launch contract; a canary of fresh boots proves identity and knob engagement first.

Process layout: the orchestrator (--execute/--canary/--preflight) runs each campaign as a child (--child) so a cap can kill
it and the runner's module-level patches stay per arm. --capture drives the real Runner.attempt up to the spawn: CPU only.
"""
import fcntl
import hashlib
import json
import math
import os
import random
import re
import runpy
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import lib
from .lib import Fail

CODE = Path(__file__).resolve().parent.parent
EVAL = Path(os.environ.get("HARNESS_DATA_ROOT", str(CODE)))
STUB = Path(__file__).resolve().parent / "stub_server.py"
# Stub traffic shape per task: (mean output tokens, tool-call responses), from the S5 control campaigns.
PROFILE = {"rust/doubly-linked-list": (26000, 11), "rust/book-store": (11000, 9), "rust/react": (14000, 9), "rust/alphametics": (9000, 8),
           "rust/forth": (12000, 10), "rust/xorcism": (19000, 10), "rust/nucleotide-codons": (5500, 6), "javascript/complex-numbers": (6500, 8)}


def harness_cmd(*args):
    return [sys.executable, "-P", "-m", "harness", *map(str, args)]


def harness_env(**extra):
    return dict(os.environ, PYTHONPATH=f"{CODE}:{EVAL}", PYTHONDONTWRITEBYTECODE="1", **extra)


# ---------------------------------------------------------------- the arm's launch contract

def knobs(x):
    return tuple(x["server"]["knobs"])


def agents_of(x):
    return x["campaign"].get("agents", 1)


def shape(x):
    """agents > 1: the server's -np and -c follow agents x per-agent context ({np} {ctx} in server.args), never a literal."""
    n = agents_of(x)
    return {"np": n, "ctx": n * x["campaign"]["agent_ctx"]} if n > 1 else None


def contract(x, arm, port="{port}"):
    s, more = x["server"], shape(x)
    argv = lib.expand_args(s["args"] + s.get("agentic_args", []) + s["extra"], arm, x, port, more)
    if more:
        lib.check_shape(argv, more["np"], more["ctx"])
    return argv


def extra(x, arm):
    return lib.expand_args(x["server"]["extra"], arm, x, "")


def runner_binary(arm):
    """What the hillclimb runner launches; an arm without a runroot (agents > 1) is launched as its frozen executable."""
    return arm["runroot"] / "build/bin" / lib.SERVER_NAME if arm["runroot"] else arm["exe"]


def template_config(x, arm):
    """The hillclimb job config the launch contract implies; the runner's queue must build exactly this."""
    args = lib.expand_args(x["server"]["args"], arm, x, "", shape(x))
    one = lambda flag: lib.flag_values(args, flag)[0]
    return (arm["k"], arm["p_min"], one("--tensor-split"), int(one("-c")), int(one("-np")), one("-sm"))


def queue_config(cfg):
    return (cfg.get("maximum"), cfg.get("p_min"), cfg.get("split"), cfg.get("context"), cfg.get("slots"), cfg.get("mode"))


def verify_launch(x, arm, launch, job="", stub=False, extra_ok=()):
    executed = launch.get("executed_command") or [""]
    optional = [f for f, _ in lib.pairs(x["server"].get("agentic_args", []))] if job.startswith(lib.FIT_JOB) else []
    lib.verify_launch(launch["command"], contract(x, arm), runner_binary(arm), executed,
                      STUB if stub else arm["exe"], optional=optional, allow=x["server"].get("allow", []), extra_ok=extra_ok)
    if stub and launch.get("stub_exe") != str(arm["exe"]):
        raise Fail(f"effective params mismatch: executed binary {launch.get('stub_exe')} != {arm['exe']}", "integrity")
    if arm["k"] > x["server"].get("max_k", arm["k"]):
        raise Fail(f"effective params mismatch: k {arm['k']} > {x['server']['max_k']}", "integrity")


# ---------------------------------------------------------------- hillclimb glue (child process)

def configure(x, arm, tasks):
    c, m = x["campaign"], x["model"]
    os.environ.update(SWIFT_ENGINE="frozen", SWIFT_LLAMA=str(arm["runroot"]), SWIFT_COMMIT=arm["commit"], SWIFT_DRAFT_MAX=str(arm["k"]),
                      SWIFT_EXTRA=shlex.join(extra(x, arm)), SWIFT_MODEL=c.get("runner_model", "swift15-iq4-xs"))
    os.environ.pop("SWIFT_TASKS", None)
    sys.path.insert(0, str(EVAL))
    g = runpy.run_path(c["runner"], run_name="harness_campaign")
    h, s = g["h"], g["s"]
    if m["id"] not in g["SWIFT15"]:  # the injection the weekend runner used for a model outside its table
        g["SWIFT15"][m["id"]] = (m["path"].name, m["bytes"], m["sha"])
        g["LOCAL_BUILDS"][m["id"]] = {"repo": "local:gguf-j " + m["path"].stem, "revision": m["sha"]}
        g["MODEL_DIRS"][m["id"]] = m["path"].parent
        s.MODEL_PATH = m["path"]
        s.MODEL = {"id": m["id"], "file": m["path"].name, "bytes": m["bytes"], "sha256": m["sha"], "local": str(m["path"]), **g["LOCAL_BUILDS"][m["id"]]}
        h.TEMPLATE_OVERRIDE_MODELS.add(m["id"])
        os.environ["SWIFT_MODEL"] = m["id"]
    base = s.queue
    s.queue = lambda: [{**job, "configs": [{**job["configs"][0], "p_min": arm["p_min"]}]} for job in base()]
    s.LONG_TAIL_TASKS[:] = tasks
    jobs = s.queue()
    s.configure_globals()
    if sorted(j["task"] for j in jobs) != sorted(tasks):
        raise Fail(f"inputs changed: runner queue {[j['task'] for j in jobs]} != {tasks}", "integrity")
    for job in jobs:
        if queue_config(job["configs"][0]) != template_config(x, arm):
            raise Fail(f"effective params mismatch (queue): {job['configs'][0]}", "integrity")
    if "--spec-draft-n-max" in h.SERVER_EXTRA or h.SERVER_EXTRA != extra(x, arm):
        raise Fail(f"effective params mismatch: runner server_extra {h.SERVER_EXTRA}", "integrity")
    return h, s, jobs


def install(h, x, arm, probe_timeout=0, max_attempts=0, spawn_hook=None):
    """Patch the shared runner in place: frozen server under its own name, scrubbed arm env recorded into launch.json."""
    original_spawn, original_acquire, original_model, original_init = h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model, h.initialize

    def spawn(self, command, log, env=None):
        if Path(command[0]).name in ("curl", "wget"):
            raise h.StopCampaign("download_forbidden")
        if command[0] == str(runner_binary(arm)):
            command = [str(arm["exe"]), *command[1:]]
            env = lib.server_env(self.env if env is None else env, arm, knobs(x))
            launch = Path(log).with_name("launch.json")
            data = h.read_sealed(launch)
            data["executed_command"] = command
            data["executed_env"] = lib.env_record(env, knobs(x))
            data["server_nice"] = os.getpriority(os.PRIO_PROCESS, 0)  # the child inherits it
            h.sealed_json(launch, data)
        if probe_timeout and "scripts.instruments.agentic_probe" in command and "--timeout" in command:
            command = list(command)
            command[command.index("--timeout") + 1] = str(probe_timeout)
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
        if max_attempts:
            k["max_attempts"] = max_attempts
        return original_init(*a, **k)

    h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model, h.initialize = spawn, acquire, ensure_model, initialize
    root = arm["runroot"]

    class Subprocess:
        """The runner checks `git -C <engine> rev-parse HEAD`; a runroot has no .git, so answer from its SOURCE commit."""

        def __getattr__(self, name):
            return getattr(subprocess, name)

        def check_output(self, cmd, *a, **k):
            if list(cmd) == ["git", "-C", str(root), "rev-parse", "HEAD"] and not (root / ".git").exists():
                source = (root / "SOURCE").read_text().split()[0]
                cmd = ["git", "-C", str(arm["repo"]), "rev-parse", f"{source}^{{commit}}"]
            return subprocess.check_output(cmd, *a, **k)

    h.subprocess = Subprocess()


def child_real(x, arm, campaign, deadline, tasks, probe_timeout, max_attempts):
    h, s, _ = configure(x, arm, tasks)
    install(h, x, arm, probe_timeout, max_attempts)
    runtime = campaign.parent / "runtime"  # gpuq units have no XDG_RUNTIME_DIR; the runner's sandbox needs one it owns
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_RUNTIME_DIR"] = str(runtime)

    def interrupted(*_):
        raise h.StopCampaign("signal")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    sys.argv = [str(s.__file__), "--campaign", str(campaign), "--assets", str(x["campaign"]["assets"]), "--deadline-epoch", str(deadline)]
    s.main()
    return 0 if json.loads((campaign / "SUMMARY.json").read_text())["state"] == "complete" else 1


class Captured(Exception):
    pass


def capture(x, arm, task, campaign, fit=False):
    """Seal the launch.json the real runner builds for this arm and task (or its capacity fit), stopping at the spawn.
    No server, no GPU."""
    h, s, jobs = configure(x, arm, [task])
    install(h, x, arm, spawn_hook=lambda command, log: (_ for _ in ()).throw(Captured()))
    h.initialize(campaign, EVAL / ".cache/hillclimb-models", profiles=jobs, max_attempts=1, agentic_assets=Path(x["campaign"]["assets"]))
    shutil.copy(x["model"]["metadata"], campaign / "artifacts" / f"{x['model']['id']}.json")
    gpu = {"temperature.gpu": 40.0, "memory.free": 15000.0, "power.limit": 150.0, "power.draw": 10.0, "utilization.gpu": 0.0,
           "pcie.link.gen.current": 3.0, "pcie.link.width.current": 4.0}
    h.telemetry = lambda: {"time": time.time(), "gpus": {d: dict(gpu) for d in h.DEVICES}, "ram_available_mib": 20000.0}
    h.endpoint_free = lambda port: None
    runner = s.agentic.AgenticRunner(campaign, 1, time.time() + 600)
    runner.ensure_model = lambda model_id: next(m for m in runner.manifest["models"] if m["id"] == model_id)
    job = next(j for j in runner.manifest["queue"] if j["task"] == task and not j["id"].startswith(lib.FIT_JOB))
    config = {"devices": h.DEVICES, **job["configs"][0]}
    try:
        if fit:  # as AgenticRunner.execute_job: same config, no workload, no task
            runner.attempt(lib.FIT_JOB + "-" + job["model"], runner.ensure_model(job["model"]), config, 0)
        else:
            runner.attempt(job["id"], runner.ensure_model(job["model"]), config | {"workload": "agentic"}, 0, task=job["task"])
    except h.StopCampaign:
        pass
    launches = sorted(campaign.glob("attempts/*/launch.json"))
    if not launches:
        raise Fail("launch capture: the runner did not reach the server launch", "deterministic")
    h.read_sealed(launches[0])
    print(launches[0])
    return 0


def check_queue(x, arm):
    h, _, jobs = configure(x, arm, x["campaign"]["tasks"])
    install(h, x, arm)
    seen = h.subprocess.check_output(["git", "-C", str(arm["runroot"]), "rev-parse", "HEAD"], text=True).strip()
    if seen != arm["commit"] or h.COMMIT != seen:
        raise Fail(f"frozen bytes changed: runner would see engine {seen}, arm wants {arm['commit']}", "integrity")
    print(json.dumps({"arm": arm["name"], "jobs": len(jobs), "config": jobs[0]["configs"][0]}))
    return 0


# ---------------------------------------------------------------- stub campaign (CPU, same on-disk layout)

def stub_arm(x, arm, name):
    """The arm a defect switch substitutes: STUB_<name>=1 on an arm in STUB_ARMS swaps in another arm's value."""
    if os.environ.get(name) == "1" and arm["name"] in os.environ.get("STUB_ARMS", "B"):
        return next(a for a in x["arms"].values() if a["name"] != arm["name"])
    return arm


def stub_env(x, arm):
    env = lib.server_env(os.environ, arm, knobs(x)) | {"STUB_ARM": arm["name"]}
    if os.environ.get("STUB_NO_STAGING") == "1" and arm["name"] in os.environ.get("STUB_ARMS", "B"):
        env.pop("GGML_SCHED_INPUT_STAGING", None)
    return env


def child_stub(x, arm, campaign, deadline, tasks, port):
    sys.path.insert(0, str(EVAL))
    from scripts.instruments import emit
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    (campaign / "attempts").mkdir(parents=True)
    k, p_min, split, ctx, slots, mode = template_config(x, arm)
    queue = [{"id": "solve-" + t.replace("/", "-"), "kind": "profile", "task": t, "workload": "agentic", "targets": [0],
              "configs": [{"context": ctx, "slots": slots, "mode": mode, "split": split, "ubatch": 128, "maximum": k, "p_min": p_min}]} for t in tasks]
    wrong = stub_arm(x, arm, "STUB_WRONG_ROOT")
    (campaign / "manifest.json").write_text(json.dumps({"engine_commit": wrong["commit"], "binary": str(runner_binary(wrong)), "queue": queue,
                                                        "binary_pins": {str(runner_binary(wrong).resolve()): "stub"},
                                                        "server_extra": extra(x, arm), "port": port, "stub": True}))
    rng = random.Random(f"{campaign.name}-{os.environ.get('STUB_SEED', '1')}")
    fail_task, runaway_task = os.environ.get("STUB_FAIL_TASK"), os.environ.get("STUB_RUNAWAY_TASK")
    stopped, total = False, dict(wall=0.0, tokens=0.0, decode_s=0.0, solved=0, done=0, accepted=0.0, draft_tokens=0.0)
    for job in queue:
        task = job["task"]
        if time.time() >= deadline:
            stopped = True
            break
        key = hashlib.sha256(f"{job['id']}|{arm['name']}".encode()).hexdigest()[:16]
        out = campaign / "attempts" / f"{key}-01"
        out.mkdir()
        command = [str(runner_binary(arm)), *contract(x, arm, port)]
        executed = [str(STUB), *command[1:]]
        env = stub_env(x, arm)
        (out / "launch.json").write_text(json.dumps({"command": command, "executed_command": executed, "stub_exe": str(stub_arm(x, arm, "STUB_WRONG_EXE")["exe"]),
                                                     "model_sha256": "stub", "server_nice": os.getpriority(os.PRIO_PROCESS, 0),
                                                     "executed_env": lib.env_record(env, knobs(x))}))
        server = lib.Server(executed, env, out / "server.log")
        server.start()
        try:
            base = f"http://127.0.0.1:{port}"
            server.wait_ready(base, 10)
            (out / "props.json").write_text(json.dumps(lib.http_json(base + "/props")))
            before = emit.scrape_metrics(base)
            mean, calls = PROFILE.get(task, (8000, 8))
            n = int(mean * rng.uniform(0.6, 1.5))
            sizes = [200, 41000, 150] if runaway_task == task else [n // calls] * calls
            for size in sizes:
                lib.http_json(base + "/stub/generate", {"n": size, "seed": rng.randrange(1 << 30), "top1_mean": 0.6}, timeout=120)
            time.sleep(float(os.environ.get("STUB_TASK_SLEEP", "0")))
            delta = emit.metrics_delta(before, emit.scrape_metrics(base))
        finally:
            server.stop(grace=10)
        wall = delta["decode_seconds"] + 30 + delta["prefill_seconds"]
        rel = f"attempts/{key}-01"
        if fail_task == task:
            result = {"job": job["id"], "task": task, "status": "failed", "reason": "probe_failed", "attempt": rel}
        else:
            probe = {"task_id": task, "complete": True, "task_success": True, "timed_out": False, "verified_wall_seconds": wall,
                     "decode_seconds": delta["decode_seconds"], "prefill_seconds": delta["prefill_seconds"], "metrics_delta": delta, "mtp_live": True}
            result = {"job": job["id"], "task": task, "status": "passed", "probe": probe, "attempt": rel}
            total.update(wall=total["wall"] + wall, tokens=total["tokens"] + delta["decode_tokens"], decode_s=total["decode_s"] + delta["decode_seconds"],
                         solved=total["solved"] + 1, accepted=total["accepted"] + delta["draft_accepted_tokens"],
                         draft_tokens=total["draft_tokens"] + delta["draft_tokens"])
        total["done"] += 1
        (out / "result.json").write_text(json.dumps(result))
        (campaign / f"job-{job['id']}.json").write_text(json.dumps(result))
    state = "stopped" if stopped else "complete"
    (campaign / "state.json").write_text(json.dumps({"status": state, "reason": "budget_expired" if stopped else None}))
    (campaign / "SUMMARY.json").write_text(json.dumps({"state": state, "tasks": total["done"], "solved": total["solved"],
                                                       "output_tokens": total["tokens"], "decode_seconds": total["decode_s"]}))
    return 0 if state == "complete" else 1


# ---------------------------------------------------------------- verification

def verify_campaign(x, path, arm, stub=False, partial=False):
    """Raise unless the campaign ran the arm: manifest, per-attempt argv/env/nice, /props of every attempt a passed job
    used, server log, and pooled draft-length counters."""
    c, path = x["campaign"], Path(path)
    manifest = lib.read_json(path / "manifest.json")
    if not manifest:
        raise Fail("campaign manifest missing", "deterministic")
    pin = str(runner_binary(arm).resolve())
    if manifest.get("engine_commit") != arm["commit"] or manifest.get("binary") != str(runner_binary(arm)) or pin not in (manifest.get("binary_pins") or {}):
        raise Fail("frozen bytes changed: manifest engine differs from the arm", "integrity")
    if manifest.get("server_extra") != extra(x, arm):
        raise Fail("effective params mismatch: server_extra differs", "integrity")
    for job in manifest["queue"]:
        if queue_config(job["configs"][0]) != template_config(x, arm):
            raise Fail(f"effective params mismatch (queue): {job['configs'][0]}", "integrity")
    attempts = sorted(path.glob("attempts/*/launch.json"))
    if not attempts:
        raise Fail("no server attempts recorded", "deterministic")
    jobs = [lib.read_json(p, {}) for p in sorted(path.glob("job-*.json"))]
    used = {j.get("attempt") for j in jobs if j.get("status") == "passed"}
    ctx, slots = template_config(x, arm)[3], template_config(x, arm)[4]
    echoes = 0
    for launch_file in attempts:
        out = launch_file.parent
        launch = json.loads(launch_file.read_text())
        job = (lib.read_json(out / "spec.json", {}) or {}).get("job", "")
        verify_launch(x, arm, launch, job, stub)
        lib.verify_env(launch.get("executed_env"), arm["env"], knobs(x))
        if not stub or os.environ.get("AB_STUB_CHECK_NICE") == "1":
            lib.verify_nice(launch.get("server_nice"))
        if (out / "props.json").exists():
            lib.verify_props(json.loads((out / "props.json").read_text()), ctx // slots, slots)
        elif f"attempts/{out.name}" in used:
            raise Fail(f"effective params mismatch: no /props record in {out.name}, which a passed job used", "integrity")
        elif not partial:
            # A boot that failed before readiness has no props; it must say so in its own sealed result.
            result = lib.read_json(out / "result.json", {}) or {}
            if result.get("status") != "failed" or result.get("reason") not in lib.PRE_READY:
                raise Fail(f"effective params mismatch: no /props record in {out.name} and no pre-readiness failure result", "integrity")
        log = (out / "server.log").read_text(errors="replace") if (out / "server.log").exists() else ""
        echoes += lib.verify_server_log(log, arm["k"], arm["p_min"], None if stub else x["server"].get("kv_type"))
    bad = [j for j in jobs if j.get("status") != "passed" and j.get("reason") not in lib.MODEL_OUTCOMES]
    if bad:
        raise Fail("campaign job failed for a non-model reason: " + ", ".join(f"{j.get('task')}:{j.get('reason')}" for j in bad), "deterministic")
    if not partial and len(jobs) != len(c["tasks"]):
        raise Fail(f"campaign has {len(jobs)} job records, expected {len(c['tasks'])}", "deterministic")
    if agents_of(x) > 1:  # the server's counters belong to the whole campaign, not to a job
        counted = [(lib.read_json(path / "SUMMARY.json", {}) or {}).get("metrics_delta") or {}]
    else:
        counted = [j["probe"]["metrics_delta"] for j in jobs if j.get("status") == "passed" and (j.get("probe") or {}).get("metrics_delta")]
    drafts = sum(m.get("drafts", 0) for m in counted)
    dtoks = sum(m.get("draft_tokens", 0) for m in counted)
    state, message, mean = lib.draft_length_check(arm["k"], arm["p_min"], drafts, dtoks, c["min_drafts"])
    if state == "fail":
        raise Fail("effective params mismatch (counters): " + message, "integrity")
    if state == "insufficient" and not partial:
        raise Fail("too few verify steps to confirm the knobs: " + message, "insufficient_data")
    return {"mean_draft_len": mean, "verify_steps": drafts, "log_echoes": echoes, "check": message}


# ---------------------------------------------------------------- summary

REQUEST_EVAL = re.compile(r"\|\s+eval time =\s*[\d.]+ ms /\s*(\d+) tokens")
METRICS = [("decode_tps", "decode tok/s", "{:.1f}"), ("decode_tps_excl", "decode tok/s excl. runaway tasks", "{:.1f}"),
           ("round_ms", "ms/round (verify + plain rounds)", "{:.2f}"), ("tokens_per_round", "tokens/round", "{:.3f}"),
           ("acceptance", "acceptance (accepted/drafted)", "{:.3f}"), ("mean_draft_len", "mean draft length", "{:.3f}"),
           ("plain_frac", "plain-round fraction", "{:.3f}"), ("output_tokens", "output tokens", "{:.0f}"),
           ("wall_s", "task wall s", "{:.0f}"), ("solved", "solved tasks", "{:.0f}"), ("runaways", "runaway tasks", "{:.0f}")]


def response_tokens(log):
    """Per-response output tokens from the server log; None when unknown (no log, or no timing lines in it)."""
    try:
        found = [int(n) for n in REQUEST_EVAL.findall(Path(log).read_text(errors="replace"))]
    except OSError:
        return None
    return found or None


def ratio(a, b):
    return a / b if a is not None and b else None


def derive(m):
    """Rounds with an empty draft are plain decode steps the server does not count as verify steps; each emits one token,
    so they are recovered as tokens - accepted - verify steps (clipped; +-~10 tokens of boundary noise per task)."""
    tokens, secs = m.get("decode_tokens"), m.get("decode_seconds")
    drafts, dtoks, acc = m.get("drafts"), m.get("draft_tokens"), m.get("draft_accepted_tokens")
    out = {"tokens": tokens, "decode_s": secs, "drafts": drafts, "draft_tokens": dtoks, "accepted": acc}
    if None in (tokens, secs, drafts, dtoks, acc):
        return out | {"plain": None, "rounds": None}
    plain = max(0.0, tokens - acc - drafts)
    return out | {"plain": plain, "rounds": drafts + plain}


def job_sample(x, campaign, job):
    probe = job.get("probe") or {}
    attempt = job.get("attempt")
    responses = response_tokens(Path(campaign) / attempt / "server.log") if attempt else None
    peak = max(responses) if responses else None
    return {"task": job.get("task"), "status": job.get("status"), "reason": job.get("reason"), "solved": bool(probe.get("task_success")),
            "timed_out": bool(probe.get("timed_out")), "wall_s": probe.get("verified_wall_seconds"),
            "responses": len(responses) if responses else None, "max_response": peak,
            "runaway": None if peak is None else peak >= x["campaign"]["runaway_tokens"]} | derive(probe.get("metrics_delta") or {})


def metrics_of(samples):
    t = {k: sum(s[k] for s in samples) for k in ("tokens", "decode_s", "drafts", "draft_tokens", "accepted", "plain", "rounds")}
    return {"decode_tps": ratio(t["tokens"], t["decode_s"]), "round_ms": ratio(1000 * t["decode_s"], t["rounds"]) if t["rounds"] else None,
            "tokens_per_round": ratio(t["tokens"], t["rounds"]), "acceptance": ratio(t["accepted"], t["draft_tokens"]),
            "mean_draft_len": ratio(t["draft_tokens"], t["drafts"]), "plain_frac": ratio(t["plain"], t["rounds"]),
            "output_tokens": t["tokens"] if samples else None, "wall_s": sum(s["wall_s"] or 0 for s in samples) if samples else None,
            "solved": sum(1 for s in samples if s["solved"]) if samples else None}


def campaign_row(x, path, meta=None):
    """Never raises on a partial campaign, a failed job or a missing log. Unknown response lengths are not "no runaway":
    they leave runaways and the runaway-excluded rate unknown (None) and the campaign ineligible."""
    path, tasks = Path(path), x["campaign"]["tasks"]
    meta = meta or lib.read_json(path / "ab.json", {}) or {}
    summary, state = lib.read_json(path / "SUMMARY.json", {}) or {}, lib.read_json(path / "state.json", {}) or {}
    samples, errors = [], []
    for file in sorted(path.glob("job-*.json")):
        job = lib.read_json(file)
        if job is None:
            errors.append(f"unreadable {file.name}")
            continue
        try:
            samples.append(job_sample(x, path, job))
        except (TypeError, ValueError, KeyError) as error:
            errors.append(f"{file.name}: {type(error).__name__}")
    ok = [s for s in samples if s["status"] == "passed" and None not in (s["tokens"], s["decode_s"], s["rounds"]) and s["decode_s"]]
    unknown = [s for s in ok if s["runaway"] is None]
    row = {"path": str(path), **{k: meta.get(k) for k in ("idx", "arm", "k", "p_min", "elapsed_s", "capped", "rc", "note")},
           "state": summary.get("state") or state.get("status") or "missing", "jobs": len(samples), "jobs_ok": len(ok),
           "failed": [{"task": s["task"], "reason": s["reason"], "max_response": s["max_response"]} for s in samples if s["status"] != "passed"],
           "runaway_tasks": [{"task": s["task"], "max_response": s["max_response"]} for s in samples if s["runaway"]],
           "runaway_unknown": [s["task"] for s in unknown], "errors": errors, "tasks": samples}
    row |= metrics_of(ok)
    row["decode_tps_excl"] = None if unknown else metrics_of([s for s in ok if not s["runaway"]])["decode_tps"]
    row["runaways"] = None if unknown else len([s for s in samples if s["runaway"]])
    row["eligible"] = row["state"] == "complete" and len(samples) == len(tasks) and row["jobs_ok"] == len(tasks) and not errors and not unknown
    return row


def fmt(spec, value):
    return "n/a" if value is None else spec.format(value)


def compare(x, rows, metrics=METRICS):
    arms = sorted(x["arms"])
    by_arm = {arm: [r for r in rows if r.get("arm") == arm and r["eligible"]] for arm in arms}
    result = {}
    for key, _, _ in metrics:
        stats = {}
        for arm, group in by_arm.items():
            values = [(r.get("idx"), r.get(key)) for r in group if r.get(key) is not None]
            nums = [v for _, v in values]
            stats[arm] = {"values": values, "mean": statistics.mean(nums) if nums else None, "min": min(nums, default=None), "max": max(nums, default=None)}
        base = stats[arms[0]]
        for arm in arms[1:]:
            s = stats[arm]
            s["vs_base"] = None if base["mean"] is None or s["mean"] is None else {
                "delta": s["mean"] - base["mean"], "pct": 100 * (s["mean"] - base["mean"]) / base["mean"] if base["mean"] else None,
                "separated_from_base_range": s["min"] > base["max"] or s["max"] < base["min"]}
        result[key] = stats
    return result


def render(x, rows, plan=None, notes=()):
    c, arms = x["campaign"], sorted(x["arms"])
    lines = [f"# {x['id']}: " + ", ".join(f"{a} = {x['arms'][a]['label']} (k{x['arms'][a]['k']} p_min {x['arms'][a]['p_min']})" for a in arms), ""]
    lines += list(notes) + [""] if notes else []
    lines += [f"Plan: {plan}", ""] if plan else []
    lines += [f"Sample unit: one {len(c['tasks'])}-task Codex campaign. Values are per campaign, never pooled. decode tok/s = sum(decode tokens) / "
              "sum(decode seconds) over the campaign's passed tasks; a round is a verify step plus a plain decode step (plain steps are "
              "recovered as tokens - accepted - verify steps). Compare tokens/round and decode tok/s, not acceptance. Runaway = any single "
              f"response >= {c['runaway_tokens']} tokens; unknown (no response timing in the server log) leaves the runaway count and the "
              "runaway-excluded rate n/a.", "",
              "| # | arm | state | tasks ok | solved | runaway | decode tok/s | excl. runaway | ms/round | tok/round | accept | draft len | plain | out tok | wall s | elapsed s |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        state = r["state"] + (" (capped)" if r.get("capped") else "") + ("" if r["eligible"] else " [excluded]")
        cells = [str(r.get("idx")), str(r.get("arm")), state, f"{r['jobs_ok']}/{len(c['tasks'])}", fmt("{:.0f}", r.get("solved")), fmt("{:.0f}", r.get("runaways"))]
        cells += [fmt(f, r.get(k)) for k, f in (("decode_tps", "{:.1f}"), ("decode_tps_excl", "{:.1f}"), ("round_ms", "{:.2f}"), ("tokens_per_round", "{:.3f}"),
                                               ("acceptance", "{:.3f}"), ("mean_draft_len", "{:.2f}"), ("plain_frac", "{:.3f}"), ("output_tokens", "{:.0f}"),
                                               ("wall_s", "{:.0f}"), ("elapsed_s", "{:.0f}"))]
        lines.append("| " + " | ".join(cells) + " |")
    flagged = [r for r in rows if r["runaway_tasks"] or r["failed"] or r.get("runaway_unknown")]
    if flagged:
        lines += ["", "Runaway, unknown or failed tasks:"]
        for r in flagged:
            lines += [f"- #{r.get('idx')} {r.get('arm')}: {t['task']} response of {t['max_response']} tokens" for t in r["runaway_tasks"]]
            lines += [f"- #{r.get('idx')} {r.get('arm')}: {t} response lengths unknown" for t in r.get("runaway_unknown") or []]
            lines += [f"- #{r.get('idx')} {r.get('arm')}: {t['task']} not passed ({t['reason']}), longest response {t['max_response']}" for t in r["failed"]]
    comparison = compare(x, rows)
    lines += ["", f"## Arms side by side (campaign values in run order; delta = arm mean - {arms[0]} mean)", "",
              "| metric | " + " | ".join(arms) + " | " + " | ".join(f"{a} - {arms[0]}" for a in arms[1:]) + " |",
              "|---|" + "---|" * (2 * len(arms) - 1)]
    for key, label, spec in METRICS:
        cells = [", ".join(f"#{i}: {fmt(spec, v)}" for i, v in comparison[key][a]["values"]) or "n/a" for a in arms]
        for a in arms[1:]:
            v = comparison[key][a].get("vs_base")
            pct = "" if not v or v["pct"] is None else f" ({v['pct']:+.1f}%)"
            cells.append("n/a" if not v else f"{v['delta']:+.4g}{pct} " + ("separated" if v["separated_from_base_range"] else "within spread"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "separated = the arm's campaign values lie entirely outside the base arm's min-max range; with few campaigns per arm that is "
              "a weak screen, not a test. Campaigns are not paired by task output.", ""]
    return "\n".join(lines) + "\n"


def summarize(x, run_dir, plan=None, notes=()):
    from . import agents
    multi = agents_of(x) > 1
    row_of, render_of, metrics = (agents.campaign_row, agents.render, agents.METRICS) if multi else (campaign_row, render, METRICS)
    run_dir, rows = Path(run_dir), []
    for path in sorted(p for p in run_dir.glob("campaign-*") if p.is_dir()):
        match = re.fullmatch(r"campaign-(\d+)-(\w+)", path.name)
        if not match:
            continue
        arm = x["arms"].get(match[2], {})
        meta = {"idx": int(match[1]), "arm": match[2], "k": arm.get("k"), "p_min": arm.get("p_min")} | (lib.read_json(path / "ab.json", {}) or {})
        try:
            rows.append(row_of(x, path, meta))
        except Exception as error:  # a damaged campaign dir must not hide the others
            rows.append({"path": str(path), "state": "unreadable", "eligible": False, "jobs_ok": 0, "runaways": None, "runaway_tasks": [],
                         "runaway_unknown": [], "failed": [], "errors": [repr(error)], "tasks": [], **meta})
    rows.sort(key=lambda r: r["idx"])
    text = render_of(x, rows, plan, notes)
    (run_dir / "summary.md").write_text(text)
    (run_dir / "summary.json").write_text(json.dumps({"plan": plan, "notes": list(notes), "campaigns": rows, "comparison": compare(x, rows, metrics)}, indent=2, default=str) + "\n")
    return rows, text


# ---------------------------------------------------------------- canary: fresh boots, identity, knobs engaged

SYL = ["ka", "to", "mi", "ra", "zu", "pe", "lo", "ve", "ni", "sa", "du", "fo", "gi", "bu", "he", "wo", "xa", "ye", "qu", "ja"]


def build_prompt(tokenize, target, seed):
    """Pseudo-random invented words (never a copy or repeated-token prompt), deterministic given the tokenizer."""
    rng = random.Random(seed)
    words = lambda count: " ".join("".join(rng.choice(SYL) for _ in range(rng.choice((2, 3, 3, 4)))) for _ in range(count))
    text = words(4000)
    toks = tokenize(text)
    for _ in range(12):
        if len(toks) >= target:
            break
        per_word = len(toks) / max(text.count(" ") + 1, 1)
        text += " " + words(max(200, int((target - len(toks)) / max(per_word, 0.1))))
        toks = tokenize(text)
    if not target <= len(toks) <= target * 1.25:
        raise Fail(f"canary prompt has {len(toks)} tokens, wanted {target}..{int(target * 1.25)}", "insufficient_data")
    return toks


def complete(c, base, prompt, n_predict):
    body = {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "cache_prompt": True, "seed": c["seed"], **c["sampling"],
            "stream": False, "return_tokens": True}
    reply = lib.call(base + "/completion", body, c["request_cap_s"])
    tokens, timings = reply.get("tokens"), reply.get("timings") or {}
    keys = ("predicted_n", "draft_n", "draft_n_accepted", "prompt_n", "cache_n")
    if not tokens or any(not isinstance(timings.get(k), (int, float)) for k in keys):
        raise Fail("missing final token identity or draft counters in the completion reply", "insufficient_data")
    if len(tokens) != timings["predicted_n"] or len(tokens) != n_predict:
        raise Fail(f"missing final token identity: {len(tokens)} tokens (predicted_n {timings['predicted_n']}), wanted {n_predict}", "insufficient_data")
    return {"tokens": tokens, "timings": {k: timings.get(k) for k in keys}}


def check_identity(a, b):
    if a["tokens"] != b["tokens"]:
        first = next((i for i, (p, q) in enumerate(zip(a["tokens"], b["tokens"])) if p != q), min(len(a["tokens"]), len(b["tokens"])))
        raise Fail(f"cross-arm identity failed: tokens differ at {first} of {len(a['tokens'])}/{len(b['tokens'])}", "integrity")
    for key in ("predicted_n", "draft_n", "draft_n_accepted"):
        if a["timings"][key] != b["timings"][key] or a["timings"][key] is None:
            raise Fail(f"cross-arm identity failed: {key} {a['timings'][key]} != {b['timings'][key]}", "integrity")
    return {"tokens": len(a["tokens"]), "draft_n": a["timings"]["draft_n"], "draft_n_accepted": a["timings"]["draft_n_accepted"]}


def canary_arm(x, arm, out, port, stub, say):
    """Fresh debug boot -> deep seeded request (identity) -> same prompt from the cache (decode-path proofs)."""
    c = x["canary"]
    out.mkdir(parents=True, exist_ok=True)
    command = [str(runner_binary(arm)), *contract(x, arm, port), *c.get("flags", [])]
    executed = [str(STUB) if stub else str(arm["exe"]), *command[1:]]
    debug = dict(c.get("env", {})) | {"LLAMA_ROUND_TRACE": str(out / "rounds.jsonl")}
    env = lib.server_env(os.environ, arm, knobs(x), debug)
    env |= ({k: v for k, v in os.environ.items() if k.startswith("STUB_")} | {"STUB_ARM": arm["name"]}) if stub else {"CUDA_VISIBLE_DEVICES": ",".join(x["devices"])}
    if stub and os.environ.get("STUB_NO_STAGING") == "1" and arm["name"] in os.environ.get("STUB_ARMS", "B"):
        env.pop("GGML_SCHED_INPUT_STAGING", None)
    record, nice = lib.env_record(env, knobs(x)), os.getpriority(os.PRIO_PROCESS, 0)
    pair = lambda flags: list(zip(flags[::2], flags[1::2]))
    verify_launch(x, arm, {"command": command, "executed_command": executed, "stub_exe": str(arm["exe"])}, stub=stub, extra_ok=pair(c.get("flags", [])))
    lib.verify_env(record, arm["env"] | debug, knobs(x))
    if not stub:
        lib.verify_nice(nice)
    (out / "launch.json").write_text(json.dumps({"command": command, "executed_command": executed, "executed_env": record, "server_nice": nice}, indent=1))
    server = lib.Server(executed, env, out / "server.log")
    try:
        server.start()
        base = f"http://127.0.0.1:{port}"
        server.wait_ready(base, c["boot_cap_s"])
        say(f"canary arm {arm['name']}: fresh boot ready")
        toks = build_prompt(lambda text: lib.call(base + "/tokenize", {"content": text}, 300)["tokens"], c["prompt_tokens"], c["seed"])
        r1 = complete(c, base, toks, c["n_predict"])
        result = {"arm": arm["name"], "prompt_tokens": len(toks), "prompt_sha": hashlib.sha256(json.dumps(toks).encode()).hexdigest(), "r1": r1}
        depth = r1["timings"]["prompt_n"] + r1["timings"]["cache_n"]
        if depth < c["min_deep_kv"]:
            raise Fail(f"canary request reached only {depth} cached+prompt tokens < {c['min_deep_kv']}", "insufficient_data")
        mark = server.log.stat().st_size
        r2 = complete(c, base, toks, c["deep_n_predict"])
        result["r2"] = {"timings": r2["timings"], "tokens": len(r2["tokens"])}
        if r2["timings"]["cache_n"] < c["min_deep_kv"]:
            raise Fail(f"R2 reused only {r2['timings']['cache_n']} cached tokens < {c['min_deep_kv']}: no deep decode to inspect", "insufficient_data")
        if "compact_mask" in c.get("proofs", []):
            result["mask"] = lib.check_compact_masks(lib.mask_uploads(server.log, mark), c["max_suffix_kib"])
        if "staging" in c.get("proofs", []):
            result["staging"] = lib.check_staging(lib.trace_counts(out / "rounds.jsonl"))
        lib.verify_log_patterns(server.log.read_text(errors="replace"), arm["canary_log"], f"arm {arm['name']}")
        if server.proc.poll() is not None:
            raise Fail(f"canary server for arm {arm['name']} died: {server.tail()}", "transient")
        if not stub and lib.process_nice(server.proc.pid) != nice:
            raise Fail("effective params mismatch (nice): the live server's nice differs from the orchestrator's", "integrity")
        say(f"canary arm {arm['name']}: " + "; ".join(f"{k} {result[k]}" for k in ("mask", "staging") if k in result))
        return result
    finally:
        server.stop()


def run_canary(x, out, port, stub):
    """Proofs only, inside gpuq's canary cap: every arm boots fresh, tokens must match across arms, knobs must be engaged."""
    started, c = time.time(), x["canary"]
    say = lambda m: lib.log(out, m)
    cpu_checks(x, out, stub, capture_proof=False)
    records = {}
    for i, name in enumerate(c.get("plan", sorted(x["arms"]))):
        if i and not stub:
            lib.wait_gpu_idle(x["devices"])
        began = time.time()
        try:
            records[name] = canary_arm(x, x["arms"][name], out / f"boot-{name}", port, stub, say)
        finally:
            lib.record_journal(began, out / f"boot-{name}" / "journal.txt")
    first, *rest = records.values()
    for other in rest:
        if other["prompt_sha"] != first["prompt_sha"]:
            raise Fail("canary prompt differs between arms (tokenizer mismatch)", "integrity")
        identity = check_identity(first["r1"], other["r1"])
    (out / "canary-proofs.json").write_text(json.dumps({"identity": identity if rest else None, **{a: {k: v for k, v in r.items() if k != "r1"} for a, r in records.items()}}, indent=1))
    say(f"canary ok: {len(records)} arms, identity {identity if rest else 'n/a'}; wall {time.time() - started:.0f}s")
    return 0


# ---------------------------------------------------------------- orchestration

def cpu_checks(x, out, stub, capture_proof=True):
    if agents_of(x) > 1:
        from . import agents
        return agents.cpu_checks(x, out, stub, capture_proof)
    c = x["campaign"]
    if not stub:
        for arm in x["arms"].values():
            lib.check_frozen(arm)
        lib.check_model(x["model"])
        lib.check_input(x["draft_vocab"], "draft vocab")
        lib.log(out, "frozen roots of every arm verified (sums, pinned hashes, commits, runroot links)")
    tasks = {t["task_id"] for t in json.loads((Path(c["assets"]) / "tasks.json").read_text())}
    if not set(c["tasks"] + [c["fit_task"]]) <= tasks:
        raise Fail(f"inputs changed: assets lack {set(c['tasks'] + [c['fit_task']]) - tasks}", "integrity")
    for name in x["arms"]:
        p = subprocess.run(harness_cmd("campaign", x["file"], "--check-queue", "--arm", name), capture_output=True, text=True, env=harness_env())
        if p.returncode:
            raise Fail(f"inputs changed: arm {name} queue did not validate: {(p.stdout + p.stderr)[-400:]}", "integrity")
    lib.log(out, "queue and engine-commit check ok for arms " + ",".join(x["arms"]))
    if capture_proof:
        capture_checks(x, out, stub)


def capture_checks(x, out, stub):
    """The real runner's own launch builder must emit each arm's contract (argv, env, executable)."""
    for name, arm in x["arms"].items():
        with tempfile.TemporaryDirectory(prefix="capture-", dir=str(out)) as tmp:
            p = subprocess.run(harness_cmd("campaign", x["file"], "--capture", "--arm", name, "--out", Path(tmp) / "c"),
                               capture_output=True, text=True, env=harness_env(), timeout=300)
            if p.returncode:
                raise Fail(f"launch capture for arm {name} failed: {(p.stdout + p.stderr)[-400:]}", "deterministic")
            launch = json.loads(Path(p.stdout.strip().splitlines()[-1]).read_text())
            verify_launch(x, arm, launch)
            lib.verify_env(launch.get("executed_env"), arm["env"], knobs(x))
    lib.log(out, "real launch builder emits the contract argv, env and executable for arms " + ",".join(x["arms"]))


def caps(x):
    b, wall = x["budget"], x["campaign"]["wall_s"]
    return math.ceil(wall * b["margin"] + b["cleanup_s"]), wall, b["cleanup_s"]


def run_campaign(x, out, idx, arm, cap_s, port, stub, state):
    path = out / f"campaign-{idx}-{arm['name']}"
    started = time.time()
    cmd = harness_cmd("campaign", x["file"], "--child", "--arm", arm["name"], "--out", path,
                      "--deadline-epoch", started + cap_s - caps(x)[2], "--port", port, *(["--stub"] if stub else []))
    lib.log(out, f"campaign #{idx} arm {arm['name']} ({arm['label']}) cap {cap_s:.0f}s")
    with open(out / f"campaign-{idx}-{arm['name']}.log", "ab") as output:
        state["child"] = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=harness_env())
        try:
            rc, capped = state["child"].wait(timeout=cap_s), False
        except subprocess.TimeoutExpired:
            lib.log(out, f"campaign cap of {cap_s:.0f}s reached; stopping its runner")
            rc, capped = 124, True
        finally:
            lib.stop_child(state.pop("child"), caps(x)[2])
    lib.record_journal(started, out / f"campaign-{idx}-{arm['name']}.journal.txt")
    meta = {"idx": idx, "arm": arm["name"], "k": arm["k"], "p_min": arm["p_min"], "elapsed_s": time.time() - started, "capped": capped, "rc": rc}
    status = lib.read_json(path / "state.json", {}) or {}
    summary = lib.read_json(path / "SUMMARY.json", {}) or {}
    if not capped and status.get("reason") == "budget_expired":
        capped = meta["capped"] = True
    path.mkdir(parents=True, exist_ok=True)
    (path / "ab.json").write_text(json.dumps(meta))
    complete = summary.get("state") == "complete" and rc == 0
    multi = agents_of(x) > 1
    if multi and summary.get("state") == "invalid":
        raise Fail(f"campaign #{idx} {arm['name']} invalid: " + "; ".join(summary.get("gates") or []), summary.get("gate_class") or "deterministic")
    if not complete and not capped and not (multi and summary.get("state") == "partial"):
        logf = out / f"campaign-{idx}-{arm['name']}.log"
        raise Fail(f"campaign #{idx} {arm['name']} stopped (rc {rc}, reason {status.get('reason')}): {logf.read_text(errors='replace')[-600:]}",
                   lib.EXIT.get(rc, "deterministic") if multi else "deterministic")
    check = verify_campaign(x, path, arm, stub=stub, partial=not complete)
    lib.log(out, f"campaign #{idx} {arm['name']} {'complete' if complete else 'capped, partial' if capped else 'partial'} in {meta['elapsed_s']:.0f}s; {check['check']}")


def run_all(x, out, port, budget_s, stub, state):
    t0, plan = time.time(), x["plan"]
    cap_s, min_start, _ = caps(x)
    lib.log(out, f"plan {''.join(plan)}, budget {budget_s:.0f}s, stub={stub}")
    cpu_checks(x, out, stub)
    lib.event("measure_start")
    rest, ran, notes, failure = list(plan), [], [], None

    def safe_summary():
        try:
            return summarize(x, out, f"{''.join(plan)} (ran {''.join(ran)})", notes)[0]
        except Exception as error:  # the summary is best effort: it must never lose the rc
            lib.log(out, f"summary failed: {type(error).__name__}: {error}")
            return []

    while rest:
        left = t0 + budget_s - time.time()
        cap = min(cap_s, left - 60)
        if cap < min_start:
            notes.append(f"wall budget: {left:.0f}s left, below the {min_start}s one campaign needs; remaining {''.join(rest)} not run")
            lib.log(out, notes[-1])
            break
        name = rest.pop(0)
        if ran and not stub:
            lib.wait_gpu_idle(x["devices"])
        try:
            run_campaign(x, out, len(ran) + 1, x["arms"][name], cap, port, stub, state)
        except Fail as error:
            failure = error
            notes.append(f"campaign #{len(ran) + 1} {name} aborted the item: {error}")
            ran.append(name)
            break
        ran.append(name)
        safe_summary()
    rows = safe_summary()
    counts = {a: sum(1 for r in rows if r.get("arm") == a and r["eligible"]) for a in x["arms"]}
    want = {a: plan.count(a) for a in x["arms"]}
    lib.log(out, "eligible campaigns per arm: " + " ".join(f"{a}:{counts[a]}/{want[a]}" for a in x["arms"]))
    lib.event(f"summary eligible {' '.join(f'{a}={counts[a]}' for a in x['arms'])}; plan {''.join(plan)} ran {''.join(ran)}")
    if failure:
        raise failure
    if any(counts[a] < want[a] for a in x["arms"]):
        raise Fail("campaign set short of the schedule (cap, wall budget or excluded campaigns); see summary.md", "insufficient_data")
    return 0


def main(x, a):
    out = Path(a.out).resolve() if a.out else None
    if a.child:
        tasks = x["campaign"]["tasks"]
        arm = x["arms"][a.arm]
        if agents_of(x) > 1:
            from . import agents
            return agents.child(x, arm, out, a.deadline_epoch, tasks, a.port, a.stub)
        if a.stub:
            return child_stub(x, arm, out, a.deadline_epoch, tasks, a.port)
        return child_real(x, arm, out, a.deadline_epoch, tasks, a.probe_timeout, a.max_attempts)
    if a.capture and agents_of(x) > 1:
        from . import agents
        return agents.capture(x, x["arms"][a.arm], out)
    if a.capture:
        return capture(x, x["arms"][a.arm], a.task or x["campaign"]["canary_task"], out, a.fit)
    if a.check_queue and agents_of(x) > 1:
        return 0  # no hillclimb queue to check: the launch contract is checked by --preflight
    if a.check_queue:
        return check_queue(x, x["arms"][a.arm])
    if a.summarize:
        print(summarize(x, Path(a.summarize))[1], end="")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    state = {}

    def on_term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        if a.preflight:
            cpu_checks(x, out, a.stub)
            lib.log(out, f"preflight ok; plan {''.join(x['plan'])}")
            return 0
        if a.canary:
            return run_canary(x, out, a.port, a.stub)
        return run_all(x, out, a.port, a.budget_s, a.stub, state)
    finally:
        lib.stop_child(state.get("child"), 30)
