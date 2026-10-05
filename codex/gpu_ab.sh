#!/usr/bin/env bash
# Parent-only token-accounting A/B. Prepare frozen UP_BIN_DIR and PATCH_BIN_DIR
# with stream-tokens-srv, shared libraries, SHA256SUMS, and ../SOURCE.
# Queue with ab=true, retry="fresh", and pins.files covering this script and frozen inputs.
# Run --self-test locally; queue --canary before the full up/patch/patch/up run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec nice -n 19 python3 - "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
SP = Path("/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad")


def parse_sse(raw):
    rows = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data and data != "[DONE]":
            rows.append(json.loads(data))
    return rows


def accounting(rows):
    tokens, text, predicted, final = [], [], None, False
    for row in rows:
        if not isinstance(row, dict):
            continue
        ids = row.get("tokens")
        if isinstance(ids, list):
            if not all(type(tok) is int for tok in ids):
                raise ValueError("invalid token IDs")
            tokens.extend(ids)
        if isinstance(row.get("content"), str):
            text.append(row["content"])
        for choice in row.get("choices") or []:
            delta = choice.get("delta") or {}
            text.append(delta.get("content") or "")
            final |= choice.get("finish_reason") is not None
        timing = row.get("timings")
        if isinstance(timing, dict) and type(timing.get("predicted_n")) is int:
            predicted = timing["predicted_n"]
        final |= row.get("stop") is True
        final |= isinstance(row.get("usage"), dict)
    return {"tokens": tokens, "text": "".join(text), "predicted_n": predicted, "final": final}


def hazard(text):
    return bool(re.search(r"\bXid\b|\bfallen off\b", text))


def journal(cutoff):
    return subprocess.check_output([
        "journalctl", "-k", "-b", "--since", "@" + str(cutoff), "--no-pager",
    ], text=True, stderr=subprocess.STDOUT)


def source_matches(source, expected):
    return bool(re.search(r"\b" + expected + r"[0-9a-f]*\b", source))


def manifest_names(manifest):
    return {line[66:].lstrip("*") for line in manifest.splitlines()}


def frozen(directory, expected):
    source = (directory.parent / "SOURCE").read_text()
    if not source_matches(source, expected):
        raise ValueError("wrong source identity: " + str(directory))
    subprocess.run(["sha256sum", "--check", "--strict", "--quiet", "SHA256SUMS"],
                   cwd=directory, check=True)
    manifest = (directory / "SHA256SUMS").read_text()
    names = manifest_names(manifest)
    required = {"stream-tokens-srv"} | {p.name for p in directory.glob("*.so*")}
    if not required <= names:
        raise ValueError("unfrozen binary or shared library")
    return hashlib.sha256(manifest.encode()).hexdigest()


def self_test():
    real = json.loads((FIXTURES / "llama-sse/long-tool-s1-1.json").read_text())
    raw = "\n".join("data: " + json.dumps(row) for row in real["chunks"]) + "\ndata: [DONE]\n"
    parsed = accounting(parse_sse(raw))
    assert parsed["final"] and parsed["predicted_n"] == 122
    nightly = json.loads(Path("/home/xanewok/repos/qwen38-eval/results/2026-10-05-mask-np2-depth/full-a1/8-up/depth-32768.json").read_text())
    parsed = accounting(nightly["streams"][1]["chunks"])
    assert parsed["final"] and parsed["predicted_n"] == 512 and len(parsed["tokens"]) == 507
    assert hazard((FIXTURES / "xid-lines.txt").read_text())
    assert not hazard("stop_xid_hash_s_G")
    recorded = SP / "queue/build/stack-gpu/resync-fmq"
    assert source_matches((recorded / "SOURCE").read_text(), "68d99c67d")
    assert not source_matches((recorded / "SOURCE").read_text(), "98c695111")
    assert "llama-server" in manifest_names((recorded / "bin/SHA256SUMS").read_text())
    print("M: recorded SSE, token accounting, kernel-log, source, and manifest parser checks passed")


parser = argparse.ArgumentParser()
parser.add_argument("--self-test", action="store_true")
parser.add_argument("--canary", action="store_true")
parser.add_argument("--budget-s", type=int)
args = parser.parse_args()
if args.self_test:
    self_test()
    sys.exit(0)

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    sys.exit("Parent must select CUDA_VISIBLE_DEVICES explicitly")
up = Path(os.environ["UP_BIN_DIR"]).resolve()
patch = Path(os.environ["PATCH_BIN_DIR"]).resolve()
identities = {"up": frozen(up, "98c695111"), "patch": frozen(patch, "f80185077")}
model = Path(os.environ.get("MODEL", "/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf"))
if not model.is_file():
    sys.exit("missing model")
model_identity = [str(model.resolve()), model.stat().st_size, model.stat().st_mtime_ns]
out = Path(os.environ.get("OUT", "codex/gpu-ab-" + time.strftime("%Y%m%d-%H%M%S")))
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    sys.exit("use a fresh output directory")
arms = ["patch"] if args.canary else ["up", "patch", "patch", "up"]
budget = args.budget_s or (600 if args.canary else 1200)
if len(arms) * (90 + 2 * 30) > budget * 0.7:
    sys.exit("stage estimates exceed budget headroom")
deadline = time.monotonic() + budget - 15


def interrupted(signum, frame):
    raise TimeoutError("budget" if signum == signal.SIGALRM else "signal")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGALRM, interrupted)
signal.setitimer(signal.ITIMER_REAL, budget - 15)
cutoff = int(time.time()) - 1
port = int(os.environ.get("PORT", "18189"))
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", port))
url = "http://127.0.0.1:" + str(port)
text = ("\u2554A" * 8) + "B"
request = {"prompt": "Repeat exactly: " + text, "grammar": 'root ::= "' + text + '"',
           "seed": 42, "temperature": 0, "n_predict": 64, "return_tokens": True,
           "cache_prompt": False}
(out / "identity.json").write_text(json.dumps({"bins": identities, "model": model_identity,
                                             "request": request, "budget_s": budget}, indent=2))
reference = None
summaries = []


def remaining(cap):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("budget")
    return min(cap, left)


def fetch(path, payload=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=remaining(timeout)) as response:
        return response.read().decode()


try:
    for index, arm in enumerate(arms):
        if hazard(journal(cutoff)):
            raise RuntimeError("hazard")
        directory = up if arm == "up" else patch
        frozen(directory, "98c695111" if arm == "up" else "f80185077")
        command = [str(directory / "stream-tokens-srv"), "-m", str(model), "--offline",
                   "--host", "127.0.0.1", "--port", str(port), "-np", "1", "-t", "8",
                   "-c", "2048", "-b", "512", "-ub", "128", "-ngl", "99", "-fa", "on",
                   "-sm", "tensor", "--tensor-split", "50,50", "-ctk", "q8_0", "-ctv", "q8_0",
                   "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-n-min", "0",
                   "--spec-draft-p-min", "0", "--no-spec-draft-backend-sampling",
                   "--spec-draft-sampling", "probabilistic"]
        errors, replies = [], {}
        env = {**os.environ, "LD_LIBRARY_PATH": str(directory), "OMP_NUM_THREADS": "8"}
        with (out / f"{index}-{arm}.server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
            (out / "pid").write_text(str(process.pid))
            try:
                ready_by = time.monotonic() + remaining(90)
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("server died")
                    try:
                        fetch("/health", timeout=1)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= ready_by:
                            raise TimeoutError("budget" if time.monotonic() >= deadline else "boot")
                        time.sleep(0.2)
                for stream in [False, True]:
                    kind = "stream" if stream else "plain"
                    (out / f"{index}-{arm}-{kind}.request.json").write_text(json.dumps({**request, "stream": stream}))
                    try:
                        raw = fetch("/completion", {**request, "stream": stream})
                        (out / f"{index}-{arm}-{kind}.raw").write_text(raw)
                        replies[kind] = accounting(parse_sse(raw) if stream else [json.loads(raw)])
                    except Exception as exc:
                        if isinstance(exc, urllib.error.HTTPError):
                            (out / f"{index}-{arm}-{kind}.error").write_bytes(exc.read())
                        errors.append(kind + ": " + str(exc))
                        if isinstance(exc, TimeoutError) and str(exc) in ("budget", "signal"):
                            raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        if hazard(journal(cutoff)):
            raise RuntimeError("hazard")
        if errors:
            raise RuntimeError("INVALID: " + "; ".join(errors))
        plain, streamed = replies["plain"], replies["stream"]
        assert plain["final"] and streamed["final"], "missing final chunk"
        assert plain["text"] == streamed["text"] == text, "output mismatch or length gate"
        assert len(plain["tokens"]) == plain["predicted_n"] == streamed["predicted_n"], "invalid counts"
        assert reference is None or plain["tokens"] == reference, "seed identity mismatch"
        reference = plain["tokens"]
        loss = plain["predicted_n"] - len(streamed["tokens"])
        if arm == "patch":
            assert streamed["tokens"] == plain["tokens"], "stream token loss"
            raw_rows = parse_sse((out / f"{index}-{arm}-stream.raw").read_text())
            assert any(len(row.get("tokens", [])) > 1 for row in raw_rows if isinstance(row, dict)), "UTF-8 gate not exercised"
        else:
            assert loss > 0, "base did not reproduce regression"
        summaries.append({"arm": arm, "predicted_n": plain["predicted_n"], "lost_ids": loss, "label": "M"})
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    remaining(1)
    print(json.dumps({"verdict": "pass", "boots": summaries}))
except BaseException as exc:
    verdict = "hazard" if str(exc) == "hazard" else "budget" if time.monotonic() >= deadline else "INVALID"
    (out / "verdict.json").write_text(json.dumps({"verdict": verdict, "error": str(exc)}))
    sys.exit(verdict + ": " + str(exc))
PY
