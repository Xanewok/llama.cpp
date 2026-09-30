#!/usr/bin/env python3
"""Parent-only GPU harness. Plan mode reads frozen files and never touches a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time

from attrib import analyze

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf"
MODEL_SHA = "e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df"
VOCAB = Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
INPUTS = ROOT / "codex/inputs"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen(root):
    root = root.resolve()
    source = (root / "SOURCE").read_text()
    if not re.fullmatch(r"[0-9a-f]{40}", source.splitlines()[0]):
        raise ValueError("invalid SOURCE commit")
    directory = root / "bin"
    expected = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        if Path(name).name != name or name in expected:
            raise ValueError("unsafe or duplicate manifest entry")
        path = directory / name
        if not path.resolve().is_relative_to(directory) or sha(path) != digest:
            raise ValueError("frozen bytes changed: " + name)
        expected[name] = digest
    actual = {p.name for p in directory.iterdir() if p.name != "SHA256SUMS"}
    if actual != set(expected) or "round-trace" not in actual:
        raise ValueError("frozen file set changed")
    return expected


def command(binary, k, port):
    return [str(binary), "-m", str(MODEL), "--alias", "local-benchmark", "-c", "102400", "-np", "1",
            "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-fa", "on", "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(k), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
            "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "-sm", "tensor",
            "--tensor-split", "50,50", "-b", "2048", "-ub", "128", "--host", "127.0.0.1",
            "--port", str(port), "--metrics", "--jinja", "--chat-template-file", str(INPUTS / "chat-template.jinja"),
            "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05",
            "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
            "-t", "8", "-tb", "8", "--spec-draft-vocab", str(VOCAB)]


def owned_pid(binary):
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / "cmdline").read_bytes().split(b"\0")
            if args and args[0] == os.fsencode(binary):
                found.append(int(p.name))
        except FileNotFoundError:
            pass
    if len(found) != 1:
        raise RuntimeError(f"expected one exact executable PID, got {found}")
    return found[0]


class Harness:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.deadline = time.monotonic() + args.budget
        self.child = None
        self.pid = None
        self.session = None
        self.capturing = False
        self.binary = None
        self.env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
                        PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("LD_PRELOAD", None)
        self.env.pop("LLAMA_ROUND_TRACE", None)

    def timeout(self, limit=300, reserve=75):
        left = self.deadline - time.monotonic() - reserve
        if left <= 0:
            raise TimeoutError("queue wall budget exhausted")
        return min(limit, left)

    def run(self, cmd, **kwargs):
        return subprocess.run(cmd, check=True, timeout=self.timeout(kwargs.pop("limit", 60)),
                              env=self.env, **kwargs)

    def xid(self):
        journal = self.run(["journalctl", "-k", "-b", "--no-pager"], capture_output=True, text=True).stdout
        if not journal.strip() or re.search(r"Xid|fallen off", journal):
            raise RuntimeError("empty/unreadable journal or boot Xid")

    def idle(self):
        self.xid()
        data = self.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                         "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        used = [int(line.strip()) for line in data.splitlines()]
        if len(used) != 2 or sum(used) > 1500:
            raise RuntimeError("selected GPUs are busy or not a pair")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.args.port)) == 0:
                raise RuntimeError("port already occupied")
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                name = (p / "comm").read_text().strip()
                if name in {"llama-server", "weekend-srv", "vocab-ab", "verify-ab", "round-trace"}:
                    raise RuntimeError("other GPU server PID " + p.name)
            except FileNotFoundError:
                pass

    def http(self, path, body=None, raw=False):
        import urllib.request
        request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}{path}",
            None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout()) as response:
            data = response.read().decode()
        return data if raw else json.loads(data)

    def metrics(self):
        raw = self.http("/metrics", raw=True)
        values = {}
        for name in ("drafts", "draft_tokens", "accepted_tokens"):
            token = "llamacpp:spec_decode_num_" + name + "_total"
            match = re.search(r"^" + re.escape(token) + r" ([0-9.eE+-]+)$", raw, re.M)
            if not match:
                raise RuntimeError("missing authoritative metric " + token)
            values[name] = float(match[1])
        return values

    def stop(self):
        if not self.pid and self.binary and self.child:
            try:
                pid = owned_pid(self.binary)
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                self.pid = pid, fields[19]
            except (FileNotFoundError, RuntimeError):
                pass
        # Exact PID with retained start time prevents stopping a reused PID.
        if self.pid:
            pid, start = self.pid
            def alive():
                try:
                    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    return fields[0] != "Z" and fields[19] == start
                except FileNotFoundError:
                    return False
            if alive():
                os.kill(pid, signal.SIGTERM)
                until = time.monotonic() + 20
                while alive() and time.monotonic() < until:
                    time.sleep(0.2)
                if alive():
                    os.kill(pid, signal.SIGKILL)
            self.pid = None
        if self.child:
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.terminate()
                try:
                    self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=5)
            self.child = None
        if self.session:
            # The session belongs to this launch; never use a process-name stop.
            subprocess.run(["nsys", "shutdown", "--session=" + self.session], env=self.env,
                           timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.session = None

    def boot(self, root, label, trace=False, nsys=False):
        self.idle()
        frozen(root)
        dest = self.out / label
        dest.mkdir()
        shutil.copytree(root / "bin", dest / "bin", symlinks=True)
        shutil.copy2(root / "SOURCE", dest / "SOURCE")
        frozen(dest)
        binary = dest / "bin/round-trace"
        self.binary = binary
        self.env["LD_LIBRARY_PATH"] = str(dest / "bin")
        self.env["LLAMA_ROUND_TRACE"] = str(dest / "rounds.jsonl") if trace else ""
        cmd = command(binary, self.args.k, self.args.port)
        (dest / "command.json").write_text(json.dumps({"command": cmd, "CUDA_VISIBLE_DEVICES": self.env["CUDA_VISIBLE_DEVICES"],
                                                       "LLAMA_ROUND_TRACE": self.env["LLAMA_ROUND_TRACE"]}, indent=2))
        if nsys:
            self.session = f"round-trace-{os.getpid()}-{label}"
            cmd = ["nsys", "launch", "--session=" + self.session, "--trace=cuda,nvtx,osrt", "--cuda-graph-trace=node", *cmd]
        with (dest / "server.log").open("w") as log:
            self.child = subprocess.Popen(cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        until = time.monotonic() + self.timeout()
        while time.monotonic() < until:
            try:
                pid = owned_pid(binary)
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                self.pid = pid, fields[19]
                if self.http("/health").get("status") == "ok":
                    break
            except (OSError, RuntimeError):
                time.sleep(0.5)
        else:
            raise RuntimeError("server load timeout")
        if not self.pid:
            raise RuntimeError("server PID ownership unresolved")
        self.request(dest, "warmup", 0)
        return dest

    def request(self, dest, case, number):
        body = json.loads((INPUTS / f"{case}.json").read_text())
        before = self.metrics()
        begin = time.monotonic_ns()
        response = self.http("/v1/chat/completions", body)
        end = time.monotonic_ns()
        after = self.metrics()
        record = {"case": case, "begin_ns": begin, "end_ns": end, "response": response,
                  "metrics": {key: after[key] - before[key] for key in before}}
        (dest / f"{case}-{number}.json").write_text(json.dumps(record, indent=2))
        if "choices" not in response:
            raise RuntimeError("completion request failed")
        return record

    def rows(self, dest, begin):
        text = (dest / "rounds.jsonl").read_text()
        lines = text[:text.rfind("\n") + 1].splitlines()
        return [row for line in lines if (row := json.loads(line))["phases"][0]["begin_ns"] >= begin]

    def collect(self, dest, case, label):
        begin = time.monotonic_ns()
        total = 0
        records = []
        for number in range(1, 21):
            record = self.request(dest, case, number if label == "pre" else number + 100)
            records.append(record)
            total += record["metrics"]["drafts"]
            rows = self.rows(dest, begin)
            result = analyze(rows, self.args.k, skip=3)
            if result["rounds_M"] >= self.args.rounds:
                (dest / f"{label}-{case}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                (dest / f"{label}-{case}-attrib.json").write_text(json.dumps(result, indent=2))
                return records
        raise RuntimeError(f"insufficient steady {case} rounds; authoritative rounds={total}")


def identity(record):
    choices = []
    for choice in record["response"]["choices"]:
        message = dict(choice["message"])
        if message.get("tool_calls"):
            message["tool_calls"] = [{key: value for key, value in call.items() if key != "id"}
                                     for call in message["tool_calls"]]
        choices.append({"message": message, "finish_reason": choice.get("finish_reason")})
    return choices, record["metrics"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("trace", "ab"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trace-patch", action="store_true", help="AB: measure total trace-on overhead against up")
    p.add_argument("--up", type=Path, default=ROOT / "codex/frozen/up")
    p.add_argument("--patch", type=Path, default=ROOT / "codex/frozen/trace")
    p.add_argument("--out", type=Path)
    p.add_argument("--k", type=int, default=int(os.environ.get("K", "3")))
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--budget", type=int, default=1200)
    p.add_argument("--port", type=int, default=8398)
    args = p.parse_args()
    if args.k < 1 or args.rounds < 40 or args.budget <= 120:
        raise ValueError("invalid K, round minimum, or budget")
    frozen(args.patch)
    if args.mode == "ab":
        frozen(args.up)
    print(json.dumps({"mode": args.mode, "boots": ["trace"] if args.mode == "trace" else ["up", "patch", "patch", "up"],
                      "command": command(args.patch / "bin/round-trace", args.k, args.port), "minimum_steady_rounds": args.rounds,
                      "budget_s_G": args.budget, "capture": "one boot: warmup, unprofiled think/tool, nsys start, think/tool, stop, export"}, indent=2))
    if not args.execute:
        return
    if os.environ.get("RUN_GPU") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("parent execute needs RUN_GPU=1 and selected CUDA_VISIBLE_DEVICES")
    out = (args.out or ROOT / "codex/runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("OUT must stay inside this worktree")
    out.mkdir(parents=True, exist_ok=False)
    h = Harness(args, out)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if sha(MODEL) != MODEL_SHA:
            raise ValueError("model SHA changed")
        pins = {str(p): sha(p) for p in [VOCAB, *INPUTS.glob("*.json"), INPUTS / "chat-template.jinja"]}
        (out / "inputs.json").write_text(json.dumps(pins, indent=2))
        if args.mode == "trace":
            dest = h.boot(args.patch, "trace", trace=True, nsys=True)
            for case in ("think", "tool"):
                h.collect(dest, case, "pre")
            h.run(["nsys", "start", "--session=" + h.session, "--sample=none", "--cpuctxsw=none", "-o", str(dest / "capture")])
            h.capturing = True
            for case in ("think", "tool"):
                h.collect(dest, case, "capture")
            h.run(["nsys", "stop", "--session=" + h.session], limit=120)
            h.capturing = False
            h.stop()
            h.xid()
            h.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", str(dest / "capture.sqlite"), str(dest / "capture.nsys-rep")], limit=180)
            for case in ("think", "tool"):
                rows = [json.loads(line) for line in (dest / f"capture-{case}.jsonl").read_text().splitlines()]
                result = analyze(rows, args.k, skip=3, min_rounds=args.rounds, sqlite=dest / "capture.sqlite")
                if result["nsys"]["gap_samples_M"] < args.rounds * (args.k + 2):
                    raise ValueError("insufficient complete paired-GPU nsys gaps")
                (dest / f"capture-{case}-gpu-attrib.json").write_text(json.dumps(result, indent=2))
        else:
            results = []
            for index, arm in enumerate(("up", "patch", "patch", "up"), 1):
                root = args.up if arm == "up" else args.patch
                dest = h.boot(root, f"{index}-{arm}", trace=args.trace_patch and arm == "patch")
                results.append({case: h.request(dest, case, 1) for case in ("tool", "think")})
                h.stop()
                h.xid()
                frozen(dest)
            for case in ("tool", "think"):
                expected = identity(results[0][case])
                if any(identity(result[case]) != expected for result in results[1:]):
                    raise ValueError("A/B completion or authoritative round/counter identity failed: " + case)
            (out / "ab-results.json").write_text(json.dumps(results, indent=2))
            summary = []
            for index, result in enumerate(results):
                for case, record in result.items():
                    rounds = record["metrics"]["drafts"]
                    if rounds <= 0:
                        raise ValueError("no authoritative MTP rounds")
                    ms = record["response"]["timings"]["predicted_ms"]
                    summary.append({"boot": index + 1, "arm": "up" if index in (0, 3) else "patch", "case": case,
                                    "rounds_M": rounds, "predicted_ms_M": ms, "ms_per_round_D": ms / rounds})
            (out / "ab-summary.json").write_text(json.dumps(summary, indent=2))
        if pins != {path: sha(Path(path)) for path in pins}:
            raise ValueError("inputs changed during run")
        frozen(args.patch)
        if args.mode == "ab":
            frozen(args.up)
        if time.monotonic() >= h.deadline:
            raise TimeoutError("queue wall budget exceeded")
    finally:
        try:
            if h.capturing and h.session:
                subprocess.run(["nsys", "stop", "--session=" + h.session], env=h.env, timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            h.stop()


if __name__ == "__main__":
    main()
