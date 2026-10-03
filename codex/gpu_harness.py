#!/usr/bin/env python3
"""Parent-only GPU harness. Plan mode reads frozen files and never touches a GPU."""
import hashlib
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
MODEL = Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf"
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
    metadata = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        if name not in {"SOURCE", "bin/SHA256SUMS"} or name in metadata or sha(root / name) != digest:
            raise ValueError("frozen metadata changed")
        metadata[name] = digest
    if set(metadata) != {"SOURCE", "bin/SHA256SUMS"}:
        raise ValueError("incomplete frozen metadata manifest")
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
    if actual != set(expected) or "sweep-srv" not in actual:
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


def gpu_memory(text):
    rows = list(csv.DictReader(io.StringIO(text), skipinitialspace=True))
    if len(rows) != 2 or len({row.get("uuid") for row in rows}) != 2:
        raise RuntimeError("selected GPUs are not a pair")
    try:
        return [int(re.fullmatch(r"\s*(\d+)(?: MiB)?\s*", row["memory.used [MiB]"]).group(1)) for row in rows]
    except (KeyError, AttributeError, ValueError) as error:
        raise RuntimeError("invalid GPU memory query") from error


class Harness:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.deadline = time.monotonic() + args.budget
        self.child = None
        self.pid = None
        self.binary = None
        self.env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
                        PYTHONDONTWRITEBYTECODE="1")
        (out / "tmp").mkdir()
        self.env["TMPDIR"] = str(out / "tmp")
        self.env["CUDA_CACHE_PATH"] = str(out / "cuda-cache")
        self.env.pop("LD_PRELOAD", None)
        self.env.pop("GGML_BACKEND_PATH", None)
        self.env.pop("LLAMA_ROUND_TRACE", None)
        self.env.pop("LLAMA_GRAPH_REUSE_DISABLE", None)
        self.env.pop("LLAMA_SPEC_REPLAY_ACCEPT", None)
        self.env.pop("LLAMA_GRAPH_CACHE_SIZE", None)
        self.env.pop("GGML_META_PLAN_CACHE", None)

    def timeout(self, limit=300, reserve=75):
        left = self.deadline - time.monotonic() - reserve
        if left <= 0:
            raise TimeoutError("queue wall budget exhausted")
        return min(limit, left)

    def run(self, cmd, **kwargs):
        return subprocess.run(cmd, check=True, timeout=self.timeout(kwargs.pop("limit", 60)),
                              env=self.env, **kwargs)

    def xid(self):
        since = self.env.get("XID_SINCE")
        if not since:
            raise RuntimeError("parent execution requires XID_SINCE")
        journal = self.run(["journalctl", "-k", "-b", "--since", since, "--no-pager"], capture_output=True, text=True).stdout
        if not journal.strip():
            raise RuntimeError("empty/unreadable kernel journal")
        if re.search(r"\bXid\b|fallen off the bus", journal, re.I):
            raise RuntimeError("Xid since XID_SINCE")

    def idle(self):
        self.xid()
        while True:
            data = self.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                             "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv"],
                            capture_output=True, text=True).stdout
            if sum(gpu_memory(data)) <= 1500:
                break
            self.timeout()
            time.sleep(1)
            self.xid()
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.args.port))

    def http(self, path, body=None, raw=False):
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

    def configure_command(self, cmd):
        return cmd

    def boot(self, root, label):
        self.idle()
        frozen(root)
        dest = self.out / label
        dest.mkdir()
        shutil.copytree(root / "bin", dest / "bin", symlinks=True)
        shutil.copy2(root / "SOURCE", dest / "SOURCE")
        shutil.copy2(root / "SHA256SUMS", dest / "SHA256SUMS")
        frozen(dest)
        binary = dest / "bin/sweep-srv"
        self.binary = binary
        self.env["LD_LIBRARY_PATH"] = str(dest / "bin")
        self.env["LLAMA_ROUND_TRACE"] = ""
        cmd = command(binary, self.args.k, self.args.port)
        cmd = self.configure_command(cmd)
        (dest / "command.json").write_text(json.dumps({"command": cmd, "CUDA_VISIBLE_DEVICES": self.env["CUDA_VISIBLE_DEVICES"],
                                                       "LLAMA_ROUND_TRACE": self.env["LLAMA_ROUND_TRACE"],
                                                       "LLAMA_TRACE": self.env.get("LLAMA_TRACE", "0"),
                                                       "XID_SINCE": self.env.get("XID_SINCE")}, indent=2))
        with (dest / "server.log").open("w") as log:
            self.child = subprocess.Popen(cmd, start_new_session=True, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        until = time.monotonic() + self.timeout()
        while time.monotonic() < until:
            if self.child.poll() is not None:
                raise RuntimeError("server exited during load; see server.log")
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
        warmup = self.request(dest, "warmup", 0)
        if "error" in warmup:
            raise ValueError("warmup failed after recording: " + json.dumps(warmup["error"]))
        return dest

    def request(self, dest, case, number, body=None, stream=False):
        body = json.loads((INPUTS / f"{case}.json").read_text()) if body is None else body
        body = dict(body)
        if stream:
            body.update(stream=True, timings_per_token=True, stream_options={"include_usage": True})
        (dest / f"{case}-{number}-input.json").write_text(json.dumps(body, indent=2))
        chunks = []
        begin = time.monotonic_ns()
        try:
            before = self.metrics()
            begin = time.monotonic_ns()
            if stream:
                request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/v1/chat/completions",
                    json.dumps(body).encode(), {"Content-Type": "application/json"})
                done = False
                with urllib.request.urlopen(request, timeout=self.timeout()) as response:
                    for line in response:
                        self.timeout()
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:].strip()
                        if data == b"[DONE]":
                            done = True
                            break
                        chunks.append(json.loads(data))
                if not done:
                    raise ValueError("incomplete SSE response")
                response = completion(chunks)
            else:
                response = self.http("/v1/chat/completions", body)
            end = time.monotonic_ns()
            after = self.metrics()
            if "choices" not in response:
                raise ValueError("completion request failed")
            record = {"case": case, "max_tokens": body["max_tokens"], "begin_ns": begin, "end_ns": end, "response": response,
                      "metrics": {key: after[key] - before[key] for key in before}, "chunks": chunks}
        except (OSError, ValueError, RuntimeError) as error:
            details = {"type": type(error).__name__, "message": str(error)}
            if isinstance(error, urllib.error.HTTPError):
                details.update(status=error.code, body=error.read().decode(errors="replace"))
            if isinstance(error, TimeoutError) and time.monotonic() >= self.deadline - 75:
                details["message"] = "queue wall budget exhausted"
            record = {"case": case, "max_tokens": body["max_tokens"], "begin_ns": begin,
                      "end_ns": time.monotonic_ns(), "chunks": chunks, "error": details}
            (dest / f"{case}-{number}-error.json").write_text(json.dumps(record, indent=2))
        (dest / f"{case}-{number}.json").write_text(json.dumps(record, indent=2))
        return record


def completion(chunks):
    message = {"role": "assistant", "content": ""}
    calls = {}
    result = {"choices": [{"message": message, "finish_reason": None}]}
    for chunk in chunks:
        if "error" in chunk:
            raise ValueError("SSE completion failed: " + str(chunk["error"]))
        for key in ("timings", "usage"):
            if isinstance(chunk.get(key), dict):
                result.setdefault(key, {}).update({name: value for name, value in chunk[key].items() if value is not None})
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("unexpected completion index")
            if choice.get("finish_reason"):
                result["choices"][0]["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content"):
                if delta.get(key):
                    message[key] = message.get(key, "") + delta[key]
            for part in delta.get("tool_calls", []):
                call = calls.setdefault(part["index"], {"type": "function", "function": {"name": "", "arguments": ""}})
                if part.get("id"):
                    call["id"] = part["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += part.get("function", {}).get(key, "")
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    if not result["choices"][0]["finish_reason"] or "timings" not in result:
        raise ValueError("missing final completion or timings")
    return result
