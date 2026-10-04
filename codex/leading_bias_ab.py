"""Parent-only frozen A/B. Default mode prints the plan without GPU or HTTP access."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
BASE = "98c695111df5b5517245b184ef4e818333cf5776"
FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(root):
    for directory in (root, root / "bin"):
        entries = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split(None, 1)
            target = directory / name
            if name in entries or not target.resolve().is_relative_to(root.resolve()) or digest(target) != expected:
                raise ValueError("changed or unsafe frozen file: " + name)
            entries[name] = expected
        required = {"SOURCE", "bin/SHA256SUMS"} if directory == root else {"round-trace"}
        if not required <= entries.keys():
            raise ValueError("incomplete frozen manifest")
    if set(p.name for p in (root / "bin").iterdir()) != set(entries) | {"SHA256SUMS"}:
        raise ValueError("unhashed frozen binary file")
    source = (root / "SOURCE").read_text().splitlines()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("invalid frozen SOURCE")
    return source


def memory(text):
    rows = []
    for line in text.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if not line.strip() or fields[0] == "index":
            continue
        if len(fields) != 4 or not fields[1].startswith("GPU-"):
            raise ValueError("bad selected GPU memory row")
        used, total = (int(x.removesuffix(" MiB")) for x in fields[2:])
        if not 0 <= used <= total or total <= 0:
            raise ValueError("bad selected GPU memory bounds")
        rows.append((fields[1], used))
    if len(rows) != 2 or len({x[0] for x in rows}) != 2:
        raise ValueError("need exactly two selected GPUs")
    return sum(x[1] for x in rows)


def hazard(text):
    return bool(re.search(r"\bXid\b|fallen off the bus", text))


def cpu_fallback(text):
    return bool(re.search(r"backend sampling[^\n]*(?:using CPU|disabling)", text))


def result(record):
    response = record.get("response", record)
    timings = response.get("timings") or {}
    n = timings.get("predicted_n")
    ms = timings.get("predicted_ms")
    rounds = record.get("metrics", {}).get("drafts")
    if not all(isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in (n, ms, rounds)):
        return None
    choices = response.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    identity = response.get("tokens") or {
        "content": response.get("content", message.get("content")),
        "reasoning": message.get("reasoning_content"),
        "tools": [x["function"] for x in message.get("tool_calls", [])],
    }
    return {"tokens": n, "rounds": rounds, "ms_round": ms / rounds, "identity": identity}


def metric_counts(text):
    values = {}
    for name in ("drafts", "draft_tokens", "accepted_tokens"):
        match = re.search(r"^llamacpp:spec_decode_num_" + name + r"_total ([0-9.eE+-]+)$", text, re.M)
        if not match or not math.isfinite(float(match[1])):
            raise ValueError("missing authoritative metric " + name)
        values[name] = float(match[1])
    return values


def self_test():
    count = 0
    for path in (FIXTURES / "llama-sse").glob("*.json"):
        record = json.loads(path.read_text())
        if "response" not in record:
            continue
        parsed = result(record)
        assert parsed and parsed["ms_round"] > 0
        for key in ("timings",):
            broken = json.loads(json.dumps(record))
            broken["response"][key] = None
            assert result(broken) is None
        count += 1
    assert count == 3
    frozen(Path("/home/xanewok/repos/host-bundle/codex/frozen/host-bundle"))
    artifact = Path("/home/xanewok/repos/qwen38-eval/results/2026-08-serving-comparisons/drafter-pilot/metrics-mtp-1.txt")
    assert metric_counts(artifact.read_text()) == {"drafts": 1534.0, "draft_tokens": 4594.0, "accepted_tokens": 2798.0}
    assert memory((FIXTURES / "nvidia-smi-query.csv").read_text()) == 123
    assert hazard((FIXTURES / "xid-lines.txt").read_text())
    assert not hazard("stop_xid_hash_s_G")
    log = (FIXTURES / "llama-sse/server.log").read_text()
    assert not hazard(log)
    assert not cpu_fallback(log)
    assert cpu_fallback("backend sampling with SPLIT_MODE_TENSOR needs top-k; using CPU")
    print("M: real fixture parsers passed: 3 responses, selected GPU memory, journal hazard, server log")


class Run:
    def __init__(self, args):
        self.args = args
        self.end = time.monotonic() + args.budget
        self.child = None
        self.env = dict(os.environ, TMPDIR=str(args.out / "tmp"), CUDA_CACHE_PATH=str(args.out / "cuda-cache"))
        self.env["LD_LIBRARY_PATH"] = ""
        for name in ("GGML_BACKEND_PATH", "LD_PRELOAD", "LLAMA_ROUND_TRACE"):
            self.env.pop(name, None)
        (args.out / "tmp").mkdir(parents=True)

    def left(self, limit=120):
        left = self.end - time.monotonic() - 45
        if left <= 0:
            raise TimeoutError("budget")
        return min(limit, left)

    def journal(self):
        text = subprocess.run(["journalctl", "-k", "-b", "--since", self.args.since, "--no-pager"],
                              check=True, capture_output=True, text=True, timeout=self.left(15)).stdout
        if not text.strip():
            raise RuntimeError("kernel journal is empty or unreadable")
        if hazard(text):
            raise RuntimeError("hazard: device fault since cutoff")

    def idle(self):
        while True:
            self.journal()
            text = subprocess.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=self.left(15)).stdout
            ram = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1])
            if memory(text) <= 1500 and ram >= 10 * 1024**2:
                break
            self.left()
            time.sleep(2)
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.args.port))

    def http(self, path, body=None):
        request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}" + path,
            data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.left(45 if body else 2)) as response:
            raw = response.read().decode()
        return raw if path == "/metrics" else json.loads(raw)

    def metrics(self):
        text = self.http("/metrics")
        return metric_counts(text)

    def stop(self):
        if self.child and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=10)
        self.child = None

    def boot(self, arm, number, canary):
        self.idle()
        binary = self.args.patch if arm == "patch" else self.args.up
        cmd = [str(binary / "bin/round-trace"), "-m", str(self.args.model), "-np", "1", "-fa", "on",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
            "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
            "-sm", "tensor", "--tensor-split", "50,50", "-ngl", "99", "-c", "32768", "-b", "2048", "-ub", "128",
            "-ctk", "q8_0", "-ctv", "q8_0", "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
            "--host", "127.0.0.1", "--port", str(self.args.port), "--metrics", "-t", "8", "-tb", "8"]
        tag = f"{arm}-{number}"
        results, errors = {}, []
        with (self.args.out / (tag + ".server.log")).open("w") as log:
            self.child = subprocess.Popen(cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT)
            (self.args.out / (tag + ".pid")).write_text(str(self.child.pid))
            ready_until = time.monotonic() + self.left(120)
            while True:
                try:
                    if self.http("/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.HTTPError):
                    pass
                if self.child.poll() is not None or time.monotonic() > ready_until:
                    raise RuntimeError("server did not become healthy")
                time.sleep(1)
            cases = {"warmup": {"ignore_eos": True}, "depth": {"ignore_eos": True}, "finite": {"ignore_eos": True, "logit_bias": [[17, 2.5], [40037, -1.0]]}, "control": {}}
            if canary:
                cases["depth-repeat"] = cases["depth"]
            for case, extra in cases.items():
                record = {"case": case}
                try:
                    before = self.metrics()
                    body = dict(prompt="Continue an ordered list of every integer and its English name from 1 through 10000, one per line. Never summarize. Begin: 1: one\n2: two\n3:",
                                n_predict=64 if canary or case == "warmup" else 384, seed=17, temperature=1, top_k=20, top_p=0.95, min_p=0.05,
                                return_tokens=True, cache_prompt=False, **extra)
                    record["response"] = self.http("/completion", body)
                    after = self.metrics()
                    record["metrics"] = {name: after[name] - before[name] for name in before}
                except urllib.error.HTTPError as error:
                    record["error"] = error.read().decode()
                except Exception as error:
                    record["error"] = str(error)
                (self.args.out / (tag + "-" + case + ".json")).write_text(json.dumps(record, indent=2))
                self.journal()
                if record.get("error") == "budget":
                    raise TimeoutError("budget")
                parsed = result(record)
                token_ids = record.get("response", {}).get("tokens", [])
                if not token_ids or any(not isinstance(t, int) for t in token_ids):
                    errors.append(case + ": missing token ids")
                counts = record.get("metrics", {})
                if not (0 <= counts.get("accepted_tokens", -1) <= counts.get("draft_tokens", -1)):
                    errors.append(case + ": invalid speculative counters")
                if parsed:
                    parsed["counts"] = counts
                if not parsed or parsed["rounds"] < 5 or parsed["tokens"] < (32 if canary or case == "warmup" else 128):
                    errors.append(case + ": insufficient or invalid output")
                else:
                    results[case] = parsed
        self.stop()
        self.journal()
        if arm == "patch" and cpu_fallback((self.args.out / (tag + ".server.log")).read_text()):
            errors.append("patch sampler fell back to CPU")
        if canary and "depth" in results and "depth-repeat" in results:
            if (results["depth"]["identity"], results["depth"]["counts"]) != (results["depth-repeat"]["identity"], results["depth-repeat"]["counts"]):
                errors.append("repeat identity changed")
        if errors:
            raise RuntimeError("INVALID: " + "; ".join(errors))
        results.pop("warmup", None)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-parent", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--up", type=Path)
    parser.add_argument("--patch", type=Path, default=ROOT / "codex/frozen/leading-bias")
    parser.add_argument("--model", type=Path, default=Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--since", default=os.environ.get("XID_SINCE"))
    parser.add_argument("--budget", type=int, default=2500)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run_parent:
        print("Parent plan: one patch canary, then up/patch/patch/up frozen boots; warmup + 3 measured requests per boot; identity + draft-round metrics. Requires --up at base " + BASE)
        return
    if not args.up or not args.out or not args.since or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("parent run needs --up, --out, --since and explicit selected CUDA_VISIBLE_DEVICES")
    if args.canary:
        args.budget = min(args.budget, 600)
    elif args.budget < 2450:
        parser.error("one 375 s canary and four 330 s boots need at least 2450 s budget")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    pins = {"up": frozen(args.up), "patch": frozen(args.patch), "model_bytes": args.model.stat().st_size,
            "harness_sha256": digest(Path(__file__))}
    if pins["up"] != BASE or pins["model_bytes"] != 15475951296:
        raise ValueError("wrong base revision or model size")
    (args.out / "pins.json").write_text(json.dumps(pins, indent=2))
    run = Run(args)
    try:
        run.boot("patch", 0, True)
        if args.canary:
            (args.out / "PASS").write_text("canary passed\n")
            return
        boots = [run.boot(arm, number, False) for number, arm in enumerate(("up", "patch", "patch", "up"), 1)]
        summary = {}
        for case in boots[0]:
            if any((boot[case]["identity"], boot[case]["counts"]) != (boots[0][case]["identity"], boots[0][case]["counts"]) for boot in boots):
                raise RuntimeError("INVALID: output identity changed: " + case)
            up = [boots[i][case]["ms_round"] for i in (0, 3)]
            patch = [boots[i][case]["ms_round"] for i in (1, 2)]
            summary[case] = {"M_up_ms_round": up, "M_patch_ms_round": patch,
                             "D_patch_minus_up_ms_round": statistics.mean(patch) - statistics.mean(up)}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        run.stop()


if __name__ == "__main__":
    main()
