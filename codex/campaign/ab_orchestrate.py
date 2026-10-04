#!/usr/bin/env python3
"""Sequence the arm campaigns (A B B A), gate each one, verify the server really ran the arm, summarise.

A = stack2-metafix, B = stack3-mask + input staging, each from its own frozen root.
Exit codes (item exit_codes): 0 ok, 3 hazard (kernel journal / memory offset only), 4 transient (GPU or port gates, boot or request transport),
5 not enough data, 6 inputs or effective server params differ from the arm, 7 canary wall cap only, 8 campaign failure.
Success output must stay clear of gpuq's built-in log rules and of the hazard signature.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from ab_common import (ARMS, ASSETS, CTX, EXTRA, FIT_TASK, HERE, IDS, IDS_SHA, MEMOC_GUARD, MIN_DRAFTS, MODEL, MODEL_BYTES,
                       MODEL_OUTCOME_REASONS, PINNED_ROOT_SUMS, STACK3_MASK, TASKS, WEEKEND, Fail, draft_length_check, read_json, sha,
                       verify_env, verify_launch, verify_nice, verify_props, verify_server_log, xid_clean)
import ab_canary
import ab_summary


def safe_summary(ctx, plan, notes):
    """The summary is best effort: a bug or a damaged campaign dir must never abort measurement or lose the rc."""
    try:
        return ab_summary.summarize(ctx.out, plan=plan, notes=notes)[0]
    except Exception as error:
        ctx.log(f"summary failed: {type(error).__name__}: {error}")
        return []

PLAN = "ABBA"
CAMPAIGN_CAP_S = int(os.environ.get("AB_CAP_S", "2400"))
CLEANUP_S = int(os.environ.get("AB_CLEANUP_S", "150"))
MIN_START_S = int(os.environ.get("AB_MIN_START_S", "1800"))  # slowest measured campaign is 1605 s; a start with less left would be cut partial
MEMORY_IDLE_MIB = 1500
CANARY_CAP_S = int(os.environ.get("AB_CANARY_CAP_S", "850"))  # gpuq caps the canary at 900 s
CHILD = None


class Ctx:
    def __init__(self, out, stub=False, port=8081, budget_s=8200, check_nice=False):
        self.out, self.stub, self.port, self.budget_s, self.check_nice = Path(out), stub, port, budget_s, check_nice
        self.t0 = time.time()
        self.out.mkdir(parents=True, exist_ok=True)

    def left(self):
        return self.t0 + self.budget_s - time.time()

    def log(self, message):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:.0f}s] {message}"
        print(line, flush=True)
        with (self.out / "console.log").open("a") as stream:
            stream.write(line + "\n")

    def event(self, text):
        path = os.environ.get("GPUQ_EVENTS")
        if path:
            with open(path, "a") as stream:
                stream.write(text + "\n")


def stop_child(grace=CLEANUP_S):
    global CHILD
    child, CHILD = CHILD, None
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


# ---------------------------------------------------------------- checks

def sums_ok(path):
    p = subprocess.run(["sha256sum", "-c", "--quiet", path.name], cwd=path.parent, capture_output=True, text=True)
    if p.returncode:
        raise Fail(f"frozen bytes changed ({path}): " + (p.stdout + p.stderr)[-300:], 6)


def full_commit(ref):
    return subprocess.run(["git", "-C", str(STACK3_MASK), "rev-parse", f"{ref}^{{commit}}"], capture_output=True, text=True).stdout.strip()


def frozen_checks(arm):
    """Each arm's bytes are the frozen ones: sums, pinned top-level sums, commit, and that the executed file is not named llama-server."""
    spec = ARMS[arm]
    for sums in spec["sums"]:
        sums_ok(Path(sums))
    exe, runner_bin = Path(spec["exe"]), Path(spec["runner_binary"])
    if exe.name == "llama-server" or not exe.is_file():
        raise Fail(f"frozen bytes changed: arm {arm} executable {exe}", 6)
    if arm == "A":
        if sha(runner_bin) != sha(exe):
            raise Fail("frozen bytes changed: weekend-srv differs from llama-server", 6)
        head = subprocess.run(["git", "-C", str(spec["root"]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != spec["commit"] or (spec["root"] / "SOURCE").read_text().splitlines()[0] != spec["commit"]:
            raise Fail(f"frozen bytes changed: arm A source commit {head}", 6)
        return
    root, runroot = Path(spec["root"]), Path(spec["runner_root"])
    for name, want in PINNED_ROOT_SUMS.items():
        if Path(name).read_text().split() != want.split():
            raise Fail(f"frozen bytes changed: {name} differs from the reported hashes", 6)
    source = (root / "SOURCE").read_text().splitlines()[0].strip()
    if source != spec["commit"] or full_commit(source) != spec["commit"] or (runroot / "SOURCE").read_text().split()[0] != spec["commit"]:
        raise Fail(f"frozen bytes changed: arm B source {source}", 6)
    if runner_bin.resolve() != exe.resolve():
        raise Fail(f"frozen bytes changed: runroot llama-server resolves to {runner_bin.resolve()}", 6)
    for link in (runroot / "build/bin").iterdir():
        if not link.is_symlink() or Path(spec["bin"]).resolve() not in link.resolve().parents:
            raise Fail(f"frozen bytes changed: runroot entry {link.name} is not a link into the frozen bin", 6)


def cpu_checks(ctx, selftest=True):
    if not ctx.stub:
        for arm in ARMS:
            frozen_checks(arm)
        if MODEL.stat().st_size != MODEL_BYTES:
            raise Fail("model SHA changed: size differs", 6)
        if sha(IDS) != IDS_SHA:
            raise Fail("inputs changed: draft vocab", 6)
        if ctx.check_nice:
            verify_nice(os.getpriority(os.PRIO_PROCESS, 0))
        ctx.log("frozen roots of both arms verified (sums, pinned hashes, commits, runroot links)")
    tasks = {t["task_id"] for t in json.loads((ASSETS / "tasks.json").read_text())}
    if not set(TASKS + [FIT_TASK]) <= tasks:
        raise Fail(f"inputs changed: assets lack {set(TASKS + [FIT_TASK]) - tasks}", 6)
    for arm in ARMS:
        # Real runner modules build and validate the queue (p_min/k per job) and the engine-commit shim; no GPU is touched.
        p = subprocess.run([sys.executable, str(HERE / "ab_campaign.py"), "--arm", arm, "--campaign", "unused",
                            "--deadline-epoch", "0", "--check"], capture_output=True, text=True, cwd=str(HERE))
        if p.returncode:
            raise Fail(f"arm {arm} queue did not validate: {(p.stdout + p.stderr)[-400:]}", 6)
    ctx.log("queue validation and engine-commit check ok for arms " + ",".join(ARMS))
    capture_checks(ctx)
    if selftest:
        negative_control(ctx)


def capture_checks(ctx):
    """The real runner's own launch builder must emit each arm's contract (argv, env, executable, nice)."""
    import tempfile
    for arm in ARMS:
        with tempfile.TemporaryDirectory(prefix="ab-capture-", dir=str(ctx.out)) as tmp:
            p = subprocess.run([sys.executable, str(HERE / "ab_campaign.py"), "--arm", arm, "--campaign", str(Path(tmp) / "c"),
                                "--deadline-epoch", "0", "--capture"], capture_output=True, text=True, cwd=str(HERE), timeout=300)
            if p.returncode:
                if "no model metadata fixture" in p.stderr + p.stdout:
                    ctx.log("launch capture skipped: no weekend model-metadata fixture")
                    return
                raise Fail(f"launch capture for arm {arm} failed: {(p.stdout + p.stderr)[-400:]}", 6)
            launch = json.loads(p.stdout.strip().splitlines()[-1])
            verify_launch(launch["command"], arm, launch.get("executed_command"))
            verify_env(launch.get("executed_env"), arm)
            if not ctx.stub and ctx.check_nice:
                verify_nice(launch.get("server_nice"))
    ctx.log("real launch builder emits the contract argv, env and executable for arms " + ",".join(ARMS))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stub_run(arm, env, tasks="rust/forth,rust/react"):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ab-neg-", dir=os.environ.get("AB_SCRATCH") or None)
    path = Path(tmp) / "campaign"
    p = subprocess.run([sys.executable, str(HERE / "ab_campaign.py"), "--arm", arm, "--campaign", str(path), "--deadline-epoch",
                        str(time.time() + 120), "--tasks", tasks, "--stub", "--port", str(free_port())],
                       env=dict(os.environ, **env), capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise Fail(f"control stub failed to run: {(p.stdout + p.stderr)[-300:]}", 6)
    return path


def negative_control(ctx):
    """Stub campaigns with a defect: verification must reject each for the right reason; honest stubs must pass."""
    import shutil
    controls = (("B without input staging", "B", {"STUB_B_NO_STAGING": "1"}, "(env)"),
                ("B executed from the wrong binary", "B", {"STUB_WRONG_EXE": "1"}, "executed binary"),
                ("B on the wrong root", "B", {"STUB_WRONG_ROOT": "1"}, "manifest engine"),
                ("A on the wrong root", "A", {"STUB_WRONG_ROOT": "1", "STUB_ARMS": "A"}, "manifest engine"),
                ("A ignores k, lying echo", "A", {"STUB_FORCE_NMAX": "4", "STUB_LIE": "1"}, "(counters)"),
                ("B ignores k, lying echo", "B", {"STUB_FORCE_NMAX": "4", "STUB_LIE": "1"}, "(counters)"))
    for label, arm, env, via in controls:
        path = stub_run(arm, env)
        try:
            try:
                verify_campaign(path, arm, stub=True, expect_tasks=2)
            except Fail as error:
                if via not in str(error) or error.rc != 6:
                    raise Fail(f"negative control '{label}' rejected for the wrong reason (wanted {via}): {error}", 6)
                ctx.log(f"negative control '{label}': rejected via {via}")
                continue
            raise Fail(f"negative control '{label}' was NOT detected: verification passed a defective server", 6)
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)
    for arm in ARMS:
        path = stub_run(arm, {})
        try:
            verify_campaign(path, arm, stub=True, expect_tasks=2)
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)
    ctx.log("positive control: honest stubs pass verification for both arms")


def gates(ctx):
    if not xid_clean(journal_file=os.environ.get("AB_JOURNAL_FILE")):
        raise Fail("kernel journal shows a GPU fault since the cutoff", 3)
    if ctx.stub:
        return
    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    used = [int(v) for v in p.stdout.split()] if p.returncode == 0 else []
    if len(used) != 2:
        raise Fail("GPU query failed", 4)
    if sum(used) > MEMORY_IDLE_MIB:
        raise Fail(f"REFUSE: GPUs hold {sum(used)} MiB (> {MEMORY_IDLE_MIB})", 4)
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", ctx.port))
        except OSError:
            raise Fail(f"REFUSE: port {ctx.port} not free", 4)
    p = subprocess.run(["bash", str(MEMOC_GUARD)], capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise Fail("memory offset guard: " + (p.stdout + p.stderr)[-200:], 3)


# ---------------------------------------------------------------- verification

def verify_campaign(path, arm, stub=False, expect_tasks=8, partial=False):
    """Raise Fail unless the campaign ran the arm: manifest, per-attempt argv/env/nice/props/log, and pooled draft-length counters."""
    spec = ARMS[arm]
    manifest = read_json(Path(path) / "manifest.json")
    if not manifest:
        raise Fail("campaign manifest missing", 8)
    pin = str(Path(spec["runner_binary"]).resolve())
    if manifest.get("engine_commit") != spec["commit"] or manifest.get("binary") != str(spec["runner_binary"]) or pin not in (manifest.get("binary_pins") or {}):
        raise Fail("frozen bytes changed: manifest engine differs from the arm", 6)
    if manifest.get("server_extra") != EXTRA:
        raise Fail("effective params mismatch: server_extra differs", 6)
    for job in manifest["queue"]:
        cfg = job["configs"][0]
        if (cfg.get("maximum"), cfg.get("p_min"), cfg.get("context"), cfg.get("slots"), cfg.get("split"), cfg.get("mode")) != \
                (spec["k"], spec["p_min"], CTX, 1, "50,50", "tensor"):
            raise Fail(f"effective params mismatch (queue): {cfg}", 6)
    attempts = sorted(Path(path).glob("attempts/*/launch.json"))
    if not attempts:
        raise Fail("no server attempts recorded", 8)
    echoes = 0
    for launch_file in attempts:
        out = launch_file.parent
        launch = json.loads(launch_file.read_text())
        job = (read_json(out / "spec.json", {}) or {}).get("job", "")
        executed = launch.get("executed_command") or [""]
        if stub:
            if launch.get("stub_exe") != str(spec["exe"]) or Path(executed[0]).name != "stub_srv.py":
                raise Fail(f"effective params mismatch: executed binary {launch.get('stub_exe')} != {spec['exe']}", 6)
            verify_launch(launch["command"], arm, None, agentic=not job.startswith("agentic-fit"))
        else:
            verify_launch(launch["command"], arm, executed, agentic=not job.startswith("agentic-fit"))
        verify_env(launch.get("executed_env"), arm)
        if not stub or os.environ.get("AB_STUB_CHECK_NICE") == "1":
            verify_nice(launch.get("server_nice"))
        if (out / "props.json").exists():
            verify_props(json.loads((out / "props.json").read_text()))
        elif not partial:
            raise Fail(f"effective params mismatch: no /props record in {out.name}", 6)
        log = (out / "server.log").read_text(errors="replace") if (out / "server.log").exists() else ""
        echoes += verify_server_log(log, arm)
        if not stub:
            kinds = re.findall(r"type_[kv]\s*=\s*(\w+)", log) + re.findall(r"\b[KV] \((\w+)\): +[\d.]+ MiB", log)
            if any(kind != "q8_0" for kind in kinds):
                raise Fail(f"effective params mismatch: target KV type {set(kinds)}", 6)
    jobs = [read_json(p, {}) for p in sorted(Path(path).glob("job-*.json"))]
    bad = [j for j in jobs if j.get("status") != "passed" and j.get("reason") not in MODEL_OUTCOME_REASONS]
    if bad:
        raise Fail("campaign job failed for a non-model reason: " + ", ".join(f"{j.get('task')}:{j.get('reason')}" for j in bad), 8)
    if not partial and len(jobs) != expect_tasks:
        raise Fail(f"campaign has {len(jobs)} job records, expected {expect_tasks}", 8)
    passed = [j for j in jobs if j.get("status") == "passed" and (j.get("probe") or {}).get("metrics_delta")]
    drafts = sum(j["probe"]["metrics_delta"].get("drafts", 0) for j in passed)
    dtoks = sum(j["probe"]["metrics_delta"].get("draft_tokens", 0) for j in passed)
    state, message, mean = draft_length_check(arm, drafts, dtoks)
    if state == "fail":
        raise Fail("effective params mismatch (counters): " + message, 6)
    if state == "insufficient" and not partial:
        raise Fail("too few verify steps to confirm the knobs: " + message, 5)
    return {"mean_draft_len": mean, "verify_steps": drafts, "log_echoes": echoes, "check": message}


# ---------------------------------------------------------------- campaigns

def run_child(ctx, cmd, log_path, cap_s):
    global CHILD
    with open(log_path, "ab") as output:
        CHILD = subprocess.Popen(cmd, cwd=str(HERE), stdout=output, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        try:
            rc = CHILD.wait(timeout=cap_s)
            capped = False
        except subprocess.TimeoutExpired:
            ctx.log(f"campaign cap of {cap_s:.0f}s reached; stopping its runner")
            stop_child()
            rc, capped = 124, True
        finally:
            stop_child()
    return rc, capped


def post_checks(ctx, what):
    if not xid_clean(journal_file=os.environ.get("AB_JOURNAL_FILE")):
        raise Fail(f"kernel journal shows a GPU fault after the {what}", 3)
    if not ctx.stub:
        p = subprocess.run(["bash", str(MEMOC_GUARD)], capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise Fail(f"memory offset guard after the {what}: " + (p.stdout + p.stderr)[-200:], 3)


def run_campaign(ctx, idx, arm, cap_s):
    path = ctx.out / f"campaign-{idx}-{arm}"
    spec = ARMS[arm]
    started = time.time()
    cmd = [sys.executable, str(HERE / "ab_campaign.py"), "--arm", arm, "--campaign", str(path),
           "--deadline-epoch", str(started + cap_s - CLEANUP_S), "--port", str(ctx.port)] + (["--stub"] if ctx.stub else [])
    ctx.log(f"campaign #{idx} arm {arm} ({spec['label']}) cap {cap_s:.0f}s")
    rc, capped = run_child(ctx, cmd, ctx.out / f"campaign-{idx}-{arm}.log", cap_s)
    meta = {"idx": idx, "arm": arm, "k": spec["k"], "p_min": spec["p_min"], "elapsed_s": time.time() - started, "capped": capped, "rc": rc}
    state = read_json(path / "state.json", {}) or {}
    summary = read_json(path / "SUMMARY.json", {}) or {}
    reason = state.get("reason")
    if not capped and reason == "budget_expired":
        capped = meta["capped"] = True
    path.mkdir(parents=True, exist_ok=True)
    (path / "ab.json").write_text(json.dumps(meta))
    result = {"meta": meta, "state": summary.get("state") or state.get("status"), "reason": reason}
    post_checks(ctx, "campaign")
    complete = (summary.get("state") == "complete") and rc == 0
    if not complete and not capped:
        logf = ctx.out / f"campaign-{idx}-{arm}.log"
        tail = logf.read_text(errors="replace")[-600:] if logf.exists() else ""
        raise Fail(f"campaign #{idx} {arm} stopped (rc {rc}, reason {reason}): {tail}", 8)
    check = verify_campaign(path, arm, stub=ctx.stub, partial=not complete)
    result["check"] = check
    ctx.log(f"campaign #{idx} {arm} {'complete' if complete else 'capped, partial'} in {meta['elapsed_s']:.0f}s; {check['check']}")
    return result


def run_all(ctx, plan=PLAN, selftest=True):
    ctx.log(f"plan {plan}, budget {ctx.budget_s:.0f}s, stub={ctx.stub}")
    cpu_checks(ctx, selftest=selftest)
    gates(ctx)
    ctx.event("measure_start")
    rest, done, notes, idx, failure = list(plan), [], [], 0, None

    def ran():
        return "".join(r["meta"]["arm"] for r in done)

    while rest:
        cap = min(CAMPAIGN_CAP_S, ctx.left() - 60)
        if cap < MIN_START_S:
            note = f"wall budget: {ctx.left():.0f}s left, below the {MIN_START_S}s one campaign needs; remaining {''.join(rest)} not run"
            notes.append(note)
            ctx.log(note)
            break
        idx += 1
        arm = rest.pop(0)
        if idx > 1:
            try:
                gates(ctx)
            except Fail as error:
                failure = error
                notes.append(f"gates before campaign #{idx} {arm} failed: {error}")
                break
        try:
            result = run_campaign(ctx, idx, arm, cap)
        except Fail as error:
            failure = error
            notes.append(f"campaign #{idx} {arm} aborted the item: {error}")
            break
        done.append(result)
        safe_summary(ctx, f"{plan} (ran {ran()})", notes)
    rows = safe_summary(ctx, f"{plan} (ran {ran()})", notes)
    counts = {a: sum(1 for r in rows if r.get("arm") == a and r["eligible"]) for a in ARMS}
    want = {a: plan.count(a) for a in ARMS}
    ctx.log("eligible campaigns per arm: " + " ".join(f"{a}:{counts[a]}/{want[a]}" for a in ARMS))
    ctx.event(f"summary eligible {' '.join(f'{a}={counts[a]}' for a in ARMS)}; plan {plan} ran {ran()}")
    if failure:
        ctx.log(f"item failed: {failure}")
        return failure.rc
    if any(counts[a] < want[a] for a in ARMS):
        ctx.log("campaign set short of the schedule (cap, wall budget or excluded campaigns); see summary.md")
        return 5
    return 0


def run_canary(ctx):
    """Proofs only: the item canary is capped at 900 s by gpuq, which two fresh boots with a deep request each already use most of."""
    started = time.time()
    deadline = started + CANARY_CAP_S - 30
    ctx.log(f"canary: fresh boots of A and B at ctx {CTX}, cross-arm identity, B mask and staging proofs; cap {CANARY_CAP_S}s")
    cpu_checks(ctx, selftest=False)
    gates(ctx)
    proof = ab_canary.proofs(ctx, ctx.out, ctx.stub, deadline, ctx.log)
    post_checks(ctx, "canary boots")
    ident = proof["identity"]
    ctx.log(f"canary ok: identity {ident['tokens']} tokens draft_n {ident['draft_n']}/{ident['draft_n_accepted']}; "
            f"B mask uploads max {proof['B']['mask']['max_kib']} KiB over {proof['B']['mask']['uploads']}; "
            f"B staged {proof['B']['staging']['input_fence_staged']} sync {proof['B']['staging']['input_fence_sync']}; "
            f"wall {time.time() - started:.0f}s")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["preflight", "run", "canary", "summarize"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--port", type=int, default=int(os.environ.get("AB_PORT", "8081")))
    p.add_argument("--budget-s", type=float, default=float(os.environ.get("BUDGET_S", "8200")))
    p.add_argument("--plan", default=os.environ.get("PLAN", PLAN))
    p.add_argument("--no-selftest", action="store_true", help="skip the stub negative/positive controls (tests that inject their own defect)")
    args = p.parse_args()
    ctx = Ctx(args.out, args.stub or os.environ.get("AB_STUB") == "1", args.port, args.budget_s, check_nice=args.mode in ("run", "canary"))

    def on_term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        if args.mode == "summarize":
            print(ab_summary.summarize(ctx.out)[1], end="")
            return 0
        if args.mode == "preflight":
            cpu_checks(ctx)
            ctx.log(f"preflight ok; plan {args.plan}")
            return 0
        return run_canary(ctx) if args.mode == "canary" else run_all(ctx, args.plan, selftest=not args.no_selftest)
    except Fail as error:
        ctx.log(f"FAILED rc={error.rc}: {error}")
        return error.rc
    except KeyboardInterrupt:
        ctx.log("signal received; stopping own campaign process")
        stop_child(30)
        return 130
    finally:
        stop_child(30)


if __name__ == "__main__":
    sys.exit(main())
