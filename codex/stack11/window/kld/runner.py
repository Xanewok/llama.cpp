"""gpuq owns admission; this harness owns identity, dump validity and PID cleanup.

Paired full-vocabulary KLD of one candidate build against its base at the six-column verify width. Report-only:
analysis never returns 5 for missing resolution; the measured KL is the result.
stack11 copy of work/kld-s9-gdn4/runner.py: each build has its own kld_dump (built against that build's headers, ABI), the config
lives in gen/ (written by the bw-stack11-base window), preflight evidence goes to the stage's --out.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time

import numpy as np
from analysis import analyze, tokens, validate_arm

ROOT = Path(__file__).resolve().parent
FAULT = re.compile(r"\bXid\b|fallen off the bus")
PENDING = "PENDING"
# ABBA brackets drift; f16 KV is the known-different control (first depth that lists it).
ARMS = {"base": ("base", "q8_0"), "cand": ("cand", "q8_0"), "cand_repeat": ("cand", "q8_0"),
        "repeat": ("base", "q8_0"), "f16": ("base", "f16")}


class Stop(Exception):
    def __init__(self, code, reason):
        self.code, self.reason = code, reason


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity(config, out):
    for name, b in config["builds"].items():
        if PENDING in (b["binary_sha256"], b["commit"]):
            raise Stop(6, f"config not finalized for {name}")
        if digest(b["binary"]) != b["binary_sha256"]:
            raise Stop(6, f"kld_dump hash changed: {name}")
    for path, want in config["sha256"].items():
        if digest(path) != want:
            raise Stop(6, f"input hash changed: {path}")
    if digest(config["model"]) != config["model_sha"]:
        raise Stop(6, "model hash changed")
    commits = set()
    for name, b in config["builds"].items():
        if Path(b["root"], "SOURCE").read_text().splitlines()[0] != b["commit"]:
            raise Stop(6, f"frozen {name} source mismatch")
        commits.add(b["commit"])
        with open(out / "preflight-checks.log", "a") as log:
            p = subprocess.run(["sha256sum", "--check", "SHA256SUMS"], cwd=b["bin"],
                               stdout=log, stderr=log, timeout=300)
        if p.returncode:
            raise Stop(6, f"frozen {name} checksum manifest failed")
    if len(commits) != len(config["builds"]):
        raise Stop(6, "base and candidate are the same commit")
    sums = [{l.split()[1]: l.split()[0] for l in Path(b["bin"], "SHA256SUMS").read_text().splitlines() if l.strip()}
            for b in config["builds"].values()]
    if sums[0]["libggml-cuda.so.0"] == sums[1]["libggml-cuda.so.0"]:
        raise Stop(6, "base and candidate share one libggml-cuda: the pair would be vacuous")
    ts = tokens(config["tokens"])
    need = 0
    for depth in config["depths"]:
        ds = tokens(depth["tokens"])
        want = [ts[i][:depth["prefix"]] if depth.get("prefix") else ts[i] for i in depth["selection"]]
        if len(ds) != len(want) or any(not np.array_equal(a, b) for a, b in zip(ds, want)):
            raise Stop(6, f"{depth['name']}: token selection mismatch")
        if any(len(t) <= depth["tail"] + 1 for t in ds):
            raise Stop(6, f"{depth['name']}: sequence shorter than tail")
        need += sum(depth["tail"] - 1 for _ in ds) * 248320 * 4 * len(depth["arms"])
    cs = tokens(config["canary"])
    if len(cs) != 1 or not np.array_equal(cs[0], ts[config["canary_source"]][:len(cs[0])]):
        raise Stop(6, "canary prefix mismatch")
    scratch = Path(config["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < need + 5 * 1024**3:
        raise Stop(6, f"need {need / 1024**3:.1f} GiB of dumps plus 5 GiB free under {scratch}")


def health(since, stub):
    if stub:
        return
    p = subprocess.run(["journalctl", "-k", "--no-pager", "--since", since],
                       capture_output=True, text=True, timeout=15)
    if p.returncode or FAULT.search(p.stdout):
        raise Stop(3, "kernel health gate failed")


def start_time(pid):
    # comm can contain spaces; starttime is field 22 after its final parenthesis.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def stop_owned(p, birth):
    if p.poll() is not None:
        return
    try:
        if start_time(p.pid) != birth:
            raise Stop(6, "owned PID starttime changed")
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if start_time(p.pid) != birth:
                raise Stop(6, "owned PID starttime changed")
            p.kill()
            p.wait(timeout=10)
    except FileNotFoundError:
        p.wait(timeout=10)


def interrupted(signum, frame):
    raise Stop(7, f"stage interrupted by signal {signum}")


def dump(config, out, name, build, kv, token_file, tail, deadline, since, stub, arm_cap):
    health(since, stub)
    left = deadline - time.monotonic() - 120
    if left < 1:
        raise Stop(7, "budget exhausted before arm")
    binary = [sys.executable, str(ROOT / "tests/stub_dump.py")] if stub else [config["builds"][build]["binary"]]
    cmd = binary + ["-m", config["model"], "-c", "65536", "-b", "2048", "-ub", "512",
                    "-ngl", "99", "-fa", "on", "-sm", "tensor", "-ts", "1,1",
                    "-ctk", kv, "-ctv", kv, "-t", "8", "-tb", "8"]
    env = dict(os.environ, LD_LIBRARY_PATH=config["builds"][build]["bin"],
               KLD_TOKENS=str(token_file), KLD_OUT=str(out / f"{name}.kldo"), KLD_TOPK="64",
               KLD_THREADS="8", KLD_DECODE_TAIL=str(tail), KLD_DECODE_TAIL_BS="6",
               KLD_TAIL_WRITE=str(out / f"{name}.klf"), STUB_ARM=name)
    request = dict(name=name, build=build, kv=kv, argv=cmd, commit=config["builds"][build]["commit"],
                   env={k: env[k] for k in ("LD_LIBRARY_PATH", "KLD_TOKENS", "KLD_DECODE_TAIL", "KLD_DECODE_TAIL_BS")})
    with open(out / "requests.jsonl", "a") as f:
        f.write(json.dumps(request) + "\n")
    events = os.environ.get("GPUQ_EVENTS")
    if events:
        with open(events, "a") as f:
            f.write("measure_start\n")
    t0 = time.monotonic()
    p, birth = None, None
    with open(out / f"{name}.log", "w") as log:
        try:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
            birth = start_time(p.pid)
            (out / "pid.json").write_text(json.dumps(dict(pid=p.pid, starttime=birth, name=name)))
            rc = p.wait(timeout=min(arm_cap, left))
        except subprocess.TimeoutExpired:
            raise Stop(7, "arm exceeded budget")
        finally:
            if p is not None and birth is not None:
                stop_owned(p, birth)
    health(since, stub)
    with open(out / "timings.jsonl", "a") as f:
        f.write(json.dumps(dict(name=name, seconds_M=time.monotonic() - t0, rc=rc)) + "\n")
    if rc:
        raise Stop(8, f"dump failed: {name}, rc={rc}")
    validate_arm(out, name, tokens(token_file), tail)


def canary_check(config, out):
    """Candidate libs load against this kld_dump ABI and produce sane, self-consistent rows."""
    ts = tokens(config["canary"])
    v, rec, full = validate_arm(out, "canary", ts, 96)
    nll = -float(np.mean(rec[0]["lp_next"]))
    limit = config.get("canary_max_nll", 6.0)
    if nll > limit:
        raise Stop(6, f"canary mean NLL {nll:.2f} > {limit}: model output is garbage")
    (out / "SUMMARY.json").write_text(json.dumps(dict(canary_mean_nll=nll, vocab=v), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("preflight", "canary", "run", "analysis"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-s", type=int, default=4400)
    ap.add_argument("--arm-timeout-s", type=int, default=800)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "8"
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.config.resolve().parent != ROOT / "gen" and not args.stub:
            raise Stop(6, "a GPU config must be the window's gen/config-*.json")
        if args.stub and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise Stop(6, "CPU stubs require hidden GPUs")
        if args.stage == "preflight":
            if args.stub:
                for d in config["depths"]:
                    tokens(d["tokens"])
            else:
                identity(config, args.out)
            print("preflight PASS" + (" (CPU stub)" if args.stub else ""))
            return 0
        if args.stage == "analysis":
            r = analyze(config, args.out, stub=args.stub)
            print(json.dumps({d: v["verdict_all_tail"] for d, v in r["depths"].items()}))
            return 0
        if not args.stub and not os.environ.get("GPUQ_ITEM"):
            raise Stop(6, "GPU stages require gpuq")
        if not args.stub and len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 2:
            raise Stop(6, "TP2 requires two visible devices")
        deadline = time.monotonic() + args.budget_s
        since = os.environ.get("XID_SINCE") or datetime.datetime.now(datetime.timezone.utc).isoformat()
        if args.stage == "canary":
            dump(config, args.out, "canary", "cand", "q8_0", config["canary"], 96, deadline, since, args.stub, args.arm_timeout_s)
            canary_check(config, args.out)
        else:
            for depth in config["depths"]:
                d = args.out / depth["name"]
                d.mkdir(parents=True, exist_ok=True)
                for name in depth["arms"]:
                    build, kv = ARMS[name]
                    dump(config, d, name, build, kv, depth["tokens"], depth["tail"], deadline, since, args.stub, args.arm_timeout_s)
        if time.monotonic() > deadline:
            raise Stop(7, "stage budget exhausted")
        print(f"{args.stage} PASS")
        return 0
    except Stop as e:
        (args.out / "FAILURE.json").write_text(json.dumps(dict(code=e.code, reason=e.reason)))
        print(e.reason, file=sys.stderr)
        return e.code
    except (ValueError, OSError, IndexError, TypeError, struct.error) as e:
        (args.out / "FAILURE.json").write_text(json.dumps(dict(code=6, reason=str(e))))
        print(f"integrity: {e}", file=sys.stderr)
        return 6
    except subprocess.TimeoutExpired:
        return 7


if __name__ == "__main__":
    sys.exit(main())
