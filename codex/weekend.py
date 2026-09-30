#!/usr/bin/env python3
"""Sequential campaign chain. GPU access exists only after the dry-run return."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import summarize
from weekend_common import ARMS, ASSETS, COMMIT, EVAL, FROZEN, IDS, MODEL_A, ORDER, ROOT, RUNNER, SHA_A, SP, check_campaign, check_frozen, configure, extra, model_b, sha

CAP = 3600
CLEANUP = 120
START = dt.datetime(2026, 9, 30, 21, tzinfo=ZoneInfo("Europe/Warsaw")).timestamp()
DEADLINE = dt.datetime(2026, 10, 3, 6, tzinfo=ZoneInfo("Europe/Warsaw")).timestamp()
CHILD = None


def log(message):
    line = f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (ROOT / "weekend.log").open("a") as stream:
        stream.write(line + "\n")


def xid():
    journal = subprocess.check_output(["journalctl", "-k", "-b", "--no-pager"], stderr=subprocess.PIPE, text=True, timeout=15)
    if not journal.strip():
        raise RuntimeError("empty/unreadable kernel journal")
    import re
    if re.search("Xid|fallen off", journal):
        raise RuntimeError("Xid/fallen off in this boot; stop the whole chain")


def guard():
    subprocess.run(["bash", str(SP / "memoc/memoc_guard.sh")], check=True, timeout=35)


def busy():
    import re
    jobs = re.compile(r"(?:j5_driver|run_queue\.sh|quant_arm|llama-server|vocab-ab|verify-ab|fa-check-srv|weekend-srv|llama-cli|llama-bench|test-backend-ops|compute-sanitizer|nsys|llama-quantize|vllm|tabbyAPI)")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if jobs.search(command):
                return f"GPU job pid={proc.name}: {command[:160]}"
        except FileNotFoundError:
            continue
        except PermissionError:
            return f"cannot inspect pid={proc.name}"
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=15)
    values = [int(line.strip()) for line in memory.splitlines()]
    if len(values) != 2:
        raise RuntimeError(f"expected two GPUs, got {len(values)}")
    if sum(values) >= 1500:
        return f"GPU memory {sum(values)} MiB >= 1500"
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True, timeout=15)
    for line in apps.splitlines():
        if line.strip() and "1password" not in line.lower():
            return "other GPU compute: " + line
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8081)) == 0:
            return "port 8081 busy"
    return None


def enough(deadline, now=None):
    return deadline - (time.time() if now is None else now) >= CAP


def stop_child():
    global CHILD
    if CHILD is not None:
        # Popen holds this exact PID; the runner owns and stops its exact child groups.
        if CHILD.poll() is None:
            CHILD.terminate()
            try:
                CHILD.wait(timeout=CLEANUP)
            except subprocess.TimeoutExpired:
                CHILD.kill()
                CHILD.wait()
        CHILD = None


def interrupted(*_):
    raise KeyboardInterrupt


def preflight():
    check_frozen()
    assert MODEL_A.stat().st_size == 15475951296
    assert sha(MODEL_A) == SHA_A, "IQ4_XS checksum mismatch"
    second, digest = model_b()
    assert sha(second) == digest, "MODEL_B checksum mismatch"
    assert ASSETS.is_dir() and RUNNER.is_file() and (SP / "memoc/memoc_guard.sh").is_file()
    ids = json.loads(IDS.read_text())
    assert len(ids) == len(set(ids)) == 25879 and all(type(i) is int and 0 <= i < 248320 for i in ids)
    pins = json.loads((ROOT / "codex/input-pins.json").read_text())
    for path, digest_pin in pins.items():
        assert sha(Path(path)) == digest_pin, f"input changed: {path}"
    for arm in ARMS:
        for model, model_sha in ((MODEL_A, SHA_A), (second, digest)):
            _, _, queue = configure(arm, model, model_sha)
            assert len(queue) == 8
    tools = json.loads((ASSETS / "tools.json").read_text())
    for path, expected in tools["binary_hashes"].items():
        assert sha(Path(path)) == expected, f"asset binary changed: {path}"
    asset_tasks = json.loads((ASSETS / "tasks.json").read_text())
    h, s, queue = configure("S0", MODEL_A, SHA_A)
    h.validate_profiles(queue, allowed_tasks={t["task_id"] for t in asset_tasks})
    for name in ("qwen-hillclimb.lock", "qwen-local-server.lock"):
        assert (Path(f"/run/user/{os.getuid()}") / name).is_file(), "missing shared lock; refuse creating outside work dir"
    log(f"checked SHA256: frozen {COMMIT}; IQ4_XS {SHA_A}; B {second} {digest}; subset, runner, asset and tool pins")
    log(f"engine: SWIFT_ENGINE=frozen SWIFT_LLAMA={FROZEN} SWIFT_COMMIT={COMMIT}; all eight tasks, tensor 50,50, target KV q8_0/q8_0")
    return second, digest, pins


def identity(phase, arm, pair, label, model, model_sha):
    return {"phase": phase, "arm": arm, "pair": pair, "model_label": label, "model_path": str(model),
            "model_sha": model_sha, "commit": COMMIT, "k": ARMS[arm], "extra": extra(arm)}


def main():
    global CHILD
    mode = os.environ.get("DRY_RUN", "1")
    if mode not in ("0", "1"):
        raise ValueError("DRY_RUN must be 0 or 1")
    dry = mode == "1"
    second, digest, pins = preflight()
    import uuid
    run_id = os.environ.get("RUN_ID", "weekend-20260930-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("RUN_ID must contain only letters, digits, underscore and hyphen")
    result_root = ROOT / "results" / run_id
    deadline = float(os.environ.get("DEADLINE_EPOCH", DEADLINE))
    phase1_only = os.environ.get("PHASE1_ONLY", "0") == "1"
    default = os.environ.get("WINNER_ARM", "S3")
    assert default in ARMS
    plan = [(1, arm, pair, "A-IQ4_XS", MODEL_A, SHA_A) for pair, order in enumerate(ORDER, 1) for arm in order]

    def path_for(phase, arm, pair, label, model_sha):
        return result_root / f"campaign-swift15-{run_id}-phase{phase}-{arm}-{label}-{model_sha[:12]}-pair{pair}"

    def reuse(path, metadata):
        if not path.exists():
            return False
        assert json.loads((path / "weekend.json").read_text()) == metadata, f"foreign dir {path}"
        check_campaign(path, metadata["arm"], Path(metadata["model_path"]), metadata["model_sha"])
        return True

    for phase, arm, pair, label, model, model_sha in plan:
        path = path_for(phase, arm, pair, label, model_sha)
        reused = reuse(path, identity(phase, arm, pair, label, model, model_sha))
        log(f"{'DRY_RUN ' if dry else ''}phase{phase} pass={pair} arm={arm} k={ARMS[arm]} model={model} "
            f"campaign={path} {'reuse complete' if reused else 'new'}; SWIFT_EXTRA={extra(arm)}")
    if dry:
        if not phase1_only:
            log(f"DRY_RUN phase2: winner = highest mean phase1 decode tok/s if all passes eligible; fallback WINNER_ARM={default}; "
                f"alternate A={MODEL_A} B={second}, pair=N in names; SHA prefix pins the model. New RUN_ID avoids collisions; partial/foreign dirs refuse.")
        log(f"DRY_RUN budget: start={dt.datetime.fromtimestamp(START, ZoneInfo('Europe/Warsaw')).isoformat()}; "
            f"deadline={dt.datetime.fromtimestamp(deadline, ZoneInfo('Europe/Warsaw')).isoformat()}; "
            f"{CAP}s cap includes {CAP-CLEANUP}s runner window + {CLEANUP}s cleanup. Check idle, zero Xid, memoc=0 before/after every campaign.")
        log("DRY_RUN: no GPU queries, guard execution, CUDA binaries, or campaign processes; preflight used SHA256, runner imports, git and ldd only.")
        return

    result_root.mkdir(parents=True, exist_ok=True)
    # One coordinator per work directory; common runner locks also guard GPU ownership.
    import fcntl
    with (ROOT / "weekend.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        while time.time() < START:
            time.sleep(min(30, START - time.time()))

        def run(phase, arm, pair, label, model, model_sha):
            global CHILD
            path = path_for(phase, arm, pair, label, model_sha)
            metadata = identity(phase, arm, pair, label, model, model_sha)
            xid()
            if reuse(path, metadata):
                log(f"reused {path}")
                xid()
                return True
            last = None
            while enough(deadline):
                xid()
                reason = busy()
                if not reason:
                    break
                if reason != last:
                    log("waiting: " + reason)
                    last = reason
                time.sleep(30)
            if not enough(deadline):
                log("deadline: less than one campaign cap remains; no campaign started")
                return False
            guard()
            check_frozen()
            for source, expected in pins.items():
                assert sha(Path(source)) == expected, f"input changed: {source}"
            if not enough(deadline):
                return False
            # h.initialize requires an absent dir. Metadata is added after it creates the manifest.
            metadata_file = path.with_suffix(".planned.json")
            metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
            command = [sys.executable, str(ROOT / "codex/campaign.py"), "--arm", arm, "--model", str(model),
                       "--sha", model_sha, "--campaign", str(path), "--deadline-epoch", str(time.time() + CAP - CLEANUP)]
            log("start " + str(path))
            env = dict(os.environ, TMPDIR=str(ROOT / "build-tmp"), PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(EVAL))
            env.pop("LD_LIBRARY_PATH", None)
            env.pop("LD_PRELOAD", None)
            env.pop("SWIFT_TASKS", None)
            with path.with_suffix(".log").open("ab") as output:
                CHILD = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
                try:
                    rc = CHILD.wait(timeout=CAP-CLEANUP)
                except subprocess.TimeoutExpired:
                    log("campaign cap reached; stop exact runner PID")
                    stop_child()
                    rc = 124
                finally:
                    stop_child()
            if path.is_dir():
                (path / "weekend.json").write_text(json.dumps(metadata, indent=2) + "\n")
            xid()
            guard()
            if rc:
                raise RuntimeError(f"campaign failed/capped rc={rc}: {path}; refuse partial reuse")
            check_campaign(path, arm, model, model_sha)
            data = summarize.rows(result_root)
            (result_root / "summary.md").write_text(summarize.render(data))
            (result_root / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
            log("complete " + str(path))
            return True

        try:
            for entry in plan:
                if not run(*entry):
                    return
            if phase1_only:
                return
            chosen, reason = summarize.winner(summarize.rows(result_root), default)
            log(f"phase2 winner {chosen}: {reason}")
            pair = 1
            while enough(deadline):
                for label, model, model_sha in (("A-IQ4_XS", MODEL_A, SHA_A), ("B-" + second.stem, second, digest)):
                    if not run(2, chosen, pair, label, model, model_sha):
                        return
                pair += 1
            log("done; deadline budget exhausted")
        finally:
            stop_child()
            data = summarize.rows(result_root)
            (result_root / "summary.md").write_text(summarize.render(data))
            (result_root / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
            xid()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        stop_child()
        log(f"STOP: {type(error).__name__}: {error}")
        raise
