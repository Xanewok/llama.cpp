#!/usr/bin/env python3
"""Parent-only GPU harness. Plan mode reads frozen files and never touches a GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import time
import sys
import urllib.error
import urllib.request

from attrib import analyze, union_ns

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf"
MODEL_SHA = "e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df"
VOCAB = Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
INPUTS = ROOT / "codex/inputs"
STACK2 = "2b645a655e59d584c66fb491253d5b9f2c2867e2"


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
        (out / "tmp").mkdir()
        self.env["TMPDIR"] = str(out / "tmp")
        self.env["CUDA_CACHE_PATH"] = str(out / "cuda-cache")
        self.env.pop("LD_PRELOAD", None)
        self.env.pop("GGML_BACKEND_PATH", None)
        self.env.pop("LLAMA_ROUND_TRACE", None)
        self.env.pop("LLAMA_GRAPH_REUSE_DISABLE", None)

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
        if not journal.strip() or re.search(r"Xid|fallen off", journal):
            raise RuntimeError("empty/unreadable journal or Xid since XID_SINCE")

    def idle(self):
        self.xid()
        while True:
            data = self.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                             "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True).stdout
            used = [tuple(int(value.strip()) for value in line.split(",")) for line in data.splitlines()]
            if len(used) != 2 or any(len(row) != 2 for row in used):
                raise RuntimeError("selected GPUs are not a pair")
            if sum(row[0] for row in used) <= 1500 and all(row[1] == 0 for row in used):
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
        shutil.copy2(root / "SHA256SUMS", dest / "SHA256SUMS")
        frozen(dest)
        binary = dest / "bin/round-trace"
        self.binary = binary
        self.env["LD_LIBRARY_PATH"] = str(dest / "bin")
        self.env["LLAMA_ROUND_TRACE"] = str(dest / "rounds.jsonl") if trace else ""
        cmd = command(binary, self.args.k, self.args.port)
        (dest / "command.json").write_text(json.dumps({"command": cmd, "CUDA_VISIBLE_DEVICES": self.env["CUDA_VISIBLE_DEVICES"],
                                                       "LLAMA_ROUND_TRACE": self.env["LLAMA_ROUND_TRACE"],
                                                       "XID_SINCE": self.env.get("XID_SINCE")}, indent=2))
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


def steady(record, k, minimum):
    metrics = record["metrics"]
    rounds, tokens = metrics["drafts"], metrics["draft_tokens"]
    final = record["response"]["timings"]
    if rounds <= 0 or rounds != int(rounds) or tokens != final.get("draft_n") or \
            metrics["accepted_tokens"] != final.get("draft_n_accepted") or not 0 <= k * rounds - tokens <= k * (k - 1) // 2:
        raise ValueError("draft counters cannot identify full fixed-K verification rounds")
    points = {}
    last_tokens = last_ms = 0
    for chunk in record["chunks"]:
        timing = chunk.get("timings") or {}
        n = timing.get("draft_n")
        if n is None:
            continue
        ms = timing.get("predicted_ms")
        if ms is None or timing.get("predicted_n") is None:
            continue
        if n < last_tokens or ms < last_ms or n > tokens or ms > final["predicted_ms"]:
            raise ValueError("reversed or inconsistent cumulative SSE timing")
        last_tokens, last_ms = n, ms
        if n > 0 and n % k == 0 and timing.get("predicted_n", record["max_tokens"]) <= min(record["max_tokens"], final["predicted_n"]) - k - 1:
            points[n // k] = ms
    # Drafts shrink near the token limit. Keep full-K counters before that tail and after the first three rounds.
    intervals = [(b, points[b] - points[a]) for a, b in zip(sorted(points), sorted(points)[1:])
                 if a >= 3 and b == a + 1 and b < rounds and points[b] > points[a]]
    if len(intervals) < minimum:
        raise ValueError(f"insufficient steady rounds: {len(intervals)} < {minimum}")
    values = [ms for _, ms in intervals]
    rate = statistics.mean(values)
    mid = len(values) // 2
    return {"steady_rounds_M": len(values), "steady_ms_per_round_D": rate,
            "steady_median_ms_per_round_D": statistics.median(values), "steady_sd_ms_M": statistics.pstdev(values),
            "early_half_ms_per_round_D": statistics.mean(values[:mid]),
            "late_half_ms_per_round_D": statistics.mean(values[mid:]),
            "generation_excess_ms_D": final["predicted_ms"] - rounds * rate,
            "intervals": [{"round_D": r, "ms_M": ms} for r, ms in intervals]}


def suite(h, dest):
    cold = json.loads((INPUTS / "tool.json").read_text())
    results = {"cold-tool": h.request(dest, "cold-tool", 1, body=cold)}
    for seed in ((1,) if getattr(h.args, "canary", False) else (1, 2)):
        for case in ("tool", "think"):
            body = json.loads((INPUTS / f"long-{case}.json").read_text())
            body["seed"] = seed
            label = f"long-{case}-s{seed}"
            record = h.request(dest, label, 1, body=body, stream=True)
            results[label] = record
    body = json.loads((INPUTS / "tool.json").read_text())
    # A tool call cut by max_tokens makes the server answer 500 (unparsable arguments); follow-ups need room to finish.
    body.update(cache_prompt=True, max_tokens=768)
    for number in range(1, (2 if getattr(h.args, "canary", False) else h.args.short_requests) + 1):
        label = f"short-tool-{number}"
        record = h.request(dest, label, 1, body=body)
        results[label] = record
        if "error" in record:
            break
        message = dict(record["response"]["choices"][0]["message"])
        calls = message.get("tool_calls", [])
        message["tool_calls"] = [dict(call, id=f"followup_{number}_{i}") for i, call in enumerate(calls)]
        if not calls:
            message.pop("tool_calls", None)
        body["messages"].append(message)
        for call in message.get("tool_calls", []):
            body["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                     "content": "Fixture result: the command was not executed. Continue with the next step."})
        body["messages"].append({"role": "user", "content": "Continue fixing the parser. Give the next small tool call."})
    return results


def validate_boot(results, dest, k, minimum):
    failures = {}
    for case, record in results.items():
        if "error" in record:
            failures[case] = [record["error"]["message"]]
            continue
        timings = record["response"].get("timings", {})
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in [*record["metrics"].values(), *[timings.get(key) for key in ("predicted_n", "predicted_ms", "prompt_ms", "cache_n")]]):
            failures[case] = ["invalid timing or metric value"]
            continue
        if timings["predicted_n"] >= 32768 or record["metrics"]["drafts"] <= 0:
            failures[case] = ["runaway response or missing authoritative rounds"]
            continue
        metrics = record["metrics"]
        if metrics["drafts"] != int(metrics["drafts"]) or metrics["draft_tokens"] != timings.get("draft_n") or \
                metrics["accepted_tokens"] != timings.get("draft_n_accepted") or not 0 <= metrics["accepted_tokens"] <= metrics["draft_tokens"]:
            failures[case] = ["authoritative counters disagree with final timings"]
            continue
        if (case == "cold-tool" or case.startswith("long-")) and timings["cache_n"] != 0:
            failures[case] = ["cold request reused a prompt cache"]
            continue
        if case.startswith("long-"):
            errors = []
            try:
                record["steady"] = steady(record, k, minimum)
            except ValueError as error:
                errors.append(str(error))
            if record["response"]["timings"]["predicted_n"] < 2048:
                errors.append("long request generated fewer than 2048 tokens")
            calls = record["response"]["choices"][0]["message"].get("tool_calls", [])
            if not any(call["function"]["name"] == "write_file" and call["function"]["arguments"] for call in calls):
                errors.append("long request did not stream visible write_file arguments")
            if errors:
                failures[case] = errors
            (dest / f"{case}-steady.json").write_text(json.dumps({"errors": errors, **record.get("steady", {})}, indent=2))
        elif case.startswith("short-tool-") and case != "short-tool-1" and record["response"]["timings"]["cache_n"] <= 0:
            failures[case] = ["agent follow-up did not reuse the prompt cache"]
    (dest / "boot-validation.json").write_text(json.dumps({"passed": not failures, "failures": failures,
                                                          "minimum_steady_rounds": minimum, "minimum_completion_tokens": 2048}, indent=2))
    return failures


def summarize(results, k, arms=None):
    arms = arms or ["up", "patch", "patch", "up"]
    if len(arms) != len(results):
        raise ValueError("boot labels and results differ")
    summary = []
    for index, result in enumerate(results):
        reference = statistics.mean(record["steady"]["steady_ms_per_round_D"]
                                    for case, record in result.items() if case.startswith("long-tool-"))
        for case, record in result.items():
            timings = record["response"]["timings"]
            rounds = record["metrics"]["drafts"]
            if rounds <= 0:
                raise ValueError("no authoritative MTP rounds")
            row = {"boot": index + 1, "arm": arms[index], "case": case,
                   "rounds_M": rounds, "predicted_ms_M": timings["predicted_ms"],
                   "whole_request_ms_per_round_D": timings["predicted_ms"] / rounds,
                   "prompt_ms_M": timings["prompt_ms"], "cached_tokens_M": timings["cache_n"],
                   "wall_ms_M": (record["end_ns"] - record["begin_ns"]) / 1e6}
            row["outside_prompt_generation_ms_D"] = row["wall_ms_M"] - timings["prompt_ms"] - timings["predicted_ms"]
            if "steady" in record:
                row.update({key: value for key, value in record["steady"].items() if key != "intervals"})
            else:
                row.update(reference_steady_ms_per_round_D=reference,
                           generation_excess_ms_D=timings["predicted_ms"] - rounds * reference)
            row["request_fixed_ms_estimate_D"] = row["outside_prompt_generation_ms_D"] + row["generation_excess_ms_D"]
            summary.append(row)
    paired = []
    per_arm = []
    for case in results[0]:
        rows = [r for r in summary if r["case"] == case]
        keys = [key for key in rows[0] if "ms" in key and key.endswith(("_M", "_D"))]
        upstream = [row for row in rows if row["arm"] == "up"]
        for arm in dict.fromkeys(arms):
            selected = [row for row in rows if row["arm"] == arm]
            per_arm.append({"arm": arm, "case": case, "boots_M": len(selected),
                            "mean_D": {key: statistics.mean(row[key] for row in selected) for key in keys},
                            "boot_range_M": {key: [min(row[key] for row in selected), max(row[key] for row in selected)] for key in keys},
                            "boot_sd_D": {key: statistics.pstdev(row[key] for row in selected) for key in keys}})
            if arm == "up":
                continue
            if len(selected) != len(upstream):
                raise ValueError("unbalanced arm boots")
            pairs = [{key: row[key] - base[key] for key in keys} for base, row in zip(upstream, selected)]
            paired.append({"case": case, "arm": arm, "pair_differences_D": pairs,
                           "patch_minus_up_D": {key: statistics.mean(pair[key] for pair in pairs) for key in keys},
                           "difference_range_D": {key: [min(pair[key] for pair in pairs), max(pair[key] for pair in pairs)] for key in keys}})
    classes = []
    for request_class, cases in (
        ("cold-1k", ["cold-tool"]),
        ("cold-10k-tool", [case for case in results[0] if case.startswith("long-tool-")]),
        ("cold-10k-think", [case for case in results[0] if case.startswith("long-think-")]),
        ("cached-followup", [case for case in results[0] if case.startswith("short-tool-") and case != "short-tool-1"]),
    ):
        boot_rows = []
        for boot, arm in enumerate(arms, 1):
            selected = [row for row in summary if row["boot"] == boot and row["case"] in cases]
            keys = ("wall_ms_M", "prompt_ms_M", "generation_excess_ms_D", "request_fixed_ms_estimate_D")
            boot_rows.append({"boot": boot, "arm": arm, **{key: statistics.mean(row[key] for row in selected) for key in keys}})
        base = [row for row in boot_rows if row["arm"] == "up"]
        for arm in dict.fromkeys(arms):
            selected = [row for row in boot_rows if row["arm"] == arm]
            deltas = [{key: row[key] - ref[key] for key in keys} for ref, row in zip(base, selected)]
            classes.append({"class": request_class, "arm": arm, "per_boot_M_D": selected,
                            "mean_D": {key: statistics.mean(row[key] for row in selected) for key in keys},
                            "minus_up_D": {key: statistics.mean(row[key] for row in deltas) for key in keys},
                            "difference_range_D": {key: [min(row[key] for row in deltas), max(row[key] for row in deltas)] for key in keys}})
    return {"requests": summary, "per_arm": per_arm, "paired": paired, "request_classes": classes,
            "pairing_definition": "Pair each arm's first boot with the first up and second boot with the second up; mirrored order balances early/late positions. Ranges describe boot spread, not confidence intervals.",
            "prompt_classes": "10k/1k are workload names; actual ab3 prompts are about 8.4k/602 uncached tokens. Cached follow-ups exclude short-tool-1; raw prompt/cache token counts are retained.",
            "fixed_cost_definition": "request_fixed_ms_estimate = wall_ms - prompt_ms - rounds * steady_ms_per_round. Generation excess = predicted_ms - rounds * steady_ms_per_round. These include ramp/tail, HTTP/dispatch/release and content/context effects, not pure graph setup. Short requests use the same boot's long-tool mean as an extrapolation; negative estimates are retained.",
            "steady_definition": "Server cumulative SSE predicted_ms differences at adjacent full-K draft counters, after round 3 and before the last K+1 output tokens; no time-based trimming.",
            "k": k}


def builds(rows, record, k):
    full = [row for row in rows if record["begin_ns"] <= row["phases"][0]["begin_ns"] < record["end_ns"]
            and 0 < row["next_begin_ns"] < record["end_ns"]
            and [p["phase"] for p in row["phases"]] == ["V", "C", *[f"D{i}" for i in range(1, k + 1)]]]
    result = {}
    for label, selected in (("first_three", full[:3]), ("later", full[3:])):
        counts = {}
        for row in selected:
            for phase in row["phases"]:
                for counter in phase["counters"]:
                    key = phase["phase"] + ":" + counter["kind"]
                    counts[key] = counts.get(key, 0) + counter["count"]
        result[label] = {"rounds_M": len(selected), "counts_M": counts,
                         "build_union_ms_D": union_ns([(e["begin_ns"], e["end_ns"]) for row in selected for e in row["events"]
                                                       if e["kind"] in ("llama_build", "meta_build", "cuda_capture", "cuda_instantiate")]) / 1e6}
    return result


def identity(record):
    choices = []
    for choice in record["response"]["choices"]:
        message = dict(choice["message"])
        if message.get("tool_calls"):
            message["tool_calls"] = [{key: value for key, value in call.items() if key != "id"}
                                     for call in message["tool_calls"]]
        choices.append({"message": message, "finish_reason": choice.get("finish_reason")})
    timings = record["response"].get("timings", {})
    return choices, record["metrics"], {key: timings[key] for key in ("predicted_n", "prompt_n", "cache_n") if key in timings}


def canary_references(k):
    run = "20261003-173851" if k == 5 else "20261003-174710"
    source = ROOT / "codex/runs" / run / "1-up"
    return {case: source / name for case, name in (
        ("cold-tool", "cold-tool-1.json"), ("long-tool-s1", "long-tool-s1-1.json"),
        ("long-think-s1", "long-think-s1-1.json"))}


def canary_identity(results, k):
    return [case for case, path in canary_references(k).items()
            if identity(results[case]) != identity(json.loads(path.read_text()))]


UP_SOURCE = "0dba27b9002b8347c6f14a9f4a1651b3478065ef"
TANDEM_ORDER = ("up", "graph-prefill", "graph-tandem", "graph-tandem", "graph-prefill", "up")


def tandem_pins(roots):
    paths = [MODEL, VOCAB, ROOT / "codex/gpu_ab.sh", Path(__file__).resolve(), ROOT / "codex/attrib.py",
             *sorted(INPUTS.glob("*.json")), INPUTS / "chat-template.jinja"]
    for root in roots.values():
        paths.extend([root / "SOURCE", root / "SHA256SUMS", root / "bin/SHA256SUMS"])
        paths.extend(root / "bin" / name for name in frozen(root))
    return sorted(set(path.resolve() for path in paths))


def stage_estimate(canary, short_requests):
    stages = {"idle_load_snapshot_s_G": 20, "warmup_s_G": 5, "cold_1k_s_G": 3,
              "long_request_s_G": 30, "long_requests_D": 2 if canary else 4,
              "followup_s_G": 3, "followups_D": 2 if canary else short_requests,
              "stop_xid_hash_s_G": 10}
    stages["boot_total_s_G"] = 20 + 5 + 3 + 30 * stages["long_requests_D"] + 3 * stages["followups_D"] + 10
    return stages


def tandem_main():
    p = argparse.ArgumentParser(description="Frozen three-arm graph reuse/input staging A/B; plan by default")
    p.add_argument("mode", choices=("tandem",))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--canary", action="store_true")
    p.add_argument("--k", choices=("3", "5", "both"))
    p.add_argument("--budget", type=int)
    p.add_argument("--port", type=int, default=8398)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--short-requests", type=int, default=8)
    p.add_argument("--up", type=Path, default=ROOT / "codex/frozen/up")
    p.add_argument("--prefill", type=Path, default=ROOT / "codex/frozen/graph-prefill")
    p.add_argument("--tandem", type=Path, default=ROOT / "codex/frozen/graph-tandem")
    p.add_argument("--out", type=Path)
    p.add_argument("--pins", action="store_true", help="print the complete hash/pin file inventory without executing")
    args = p.parse_args()
    args.k = args.k or ("3" if args.canary else "both")
    args.budget = args.budget or (600 if args.canary else 3600)
    if args.rounds < 40 or not 8 <= args.short_requests <= 16 or not 120 < args.budget <= 7200:
        raise ValueError("invalid round minimum, follow-up count or budget")
    if args.canary and (args.k == "both" or args.budget > 600):
        raise ValueError("canary uses one K and at most 600 seconds")
    ks = (5, 3) if args.k == "both" else (int(args.k),)
    boots = ("up", "graph-prefill", "graph-tandem") if args.canary else TANDEM_ORDER
    stages = stage_estimate(args.canary, args.short_requests)
    stages["initial_input_hash_s_G"] = 20
    estimated = stages["initial_input_hash_s_G"] + stages["boot_total_s_G"] * len(boots) * len(ks)
    if estimated > 0.7 * args.budget:
        raise ValueError("stage estimates exceed 70% of the wall budget")
    roots = {"up": args.up.resolve(), "graph-prefill": args.prefill.resolve(), "graph-tandem": args.tandem.resolve()}
    paths = tandem_pins(roots)
    if (args.up / "SOURCE").read_text().splitlines()[0] != UP_SOURCE:
        raise ValueError("up must be the supplied codex/frozen/up snapshot")
    for case in ("tool", "think"):
        body = json.loads((INPUTS / f"long-{case}.json").read_text())
        if body["max_tokens"] < max(2048, (max(ks) + 1) * (args.rounds + 5)) or body["cache_prompt"]:
            raise ValueError("long workload token budget is too short or cache is enabled")
    if args.pins:
        print(json.dumps([str(path) for path in paths], indent=2))
        return
    print(json.dumps({"mode": "tandem", "canary": args.canary, "k": list(ks), "boots": list(boots),
                      "stage_estimates": stages, "stage_count_D": len(boots) * len(ks),
                      "stage_estimate_s_G": estimated, "budget_s_G": args.budget, "minimum_steady_rounds": args.rounds,
                      "workloads": "ab3 unchanged: cold-tool, long-tool/think seeds 1/2 (seed 1 in canary), cached tool follow-ups with 768-token headroom",
                      "pins": [str(path) for path in paths], "capture": "trace-off"}, indent=2))
    if not args.execute:
        return
    if os.environ.get("RUN_GPU") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("parent execute needs RUN_GPU=1 and selected CUDA_VISIBLE_DEVICES")
    out = (args.out or ROOT / "codex/runs" / ("tandem-" + time.strftime("%Y%m%d-%H%M%S"))).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("OUT must stay inside this worktree")
    out.mkdir(parents=True, exist_ok=False)
    h = Harness(args, out)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        pins = {str(path): sha(path) for path in paths}
        if pins[str(MODEL.resolve())] != MODEL_SHA:
            raise ValueError("model SHA changed")
        (out / "inputs.json").write_text(json.dumps(pins, indent=2))
        reports = []
        for k in ks:
            args.k = k
            results = []
            for index, arm in enumerate(boots, 1):
                dest = h.boot(roots[arm], f"k{k}-{index}-{arm}")
                result = suite(h, dest)
                failures = validate_boot(result, dest, k, args.rounds)
                results.append(result)
                print(f"M: K={k} boot={index} arm={arm}: recorded {len(result)} requests, validation failures={len(failures)}", flush=True)
                (out / f"k{k}-ab-results.json").write_text(json.dumps(results, indent=2))
                h.stop()
                h.xid()
                frozen(dest)
                if failures:
                    raise ValueError("boot validation failed after writing results: " + json.dumps(failures))
            mismatches = [case for case in results[0]
                          if any(identity(result[case]) != identity(results[0][case]) for result in results[1:])]
            (out / f"k{k}-identity.json").write_text(json.dumps({"passed": not mismatches, "failures": mismatches}, indent=2))
            if mismatches:
                raise ValueError("three-arm completion or authoritative counter identity failed: " + ", ".join(mismatches))
            reports.append(summarize(results, k, list(boots)))
        if pins != {path: sha(Path(path)) for path in pins}:
            raise ValueError("inputs changed during run")
        h.timeout(limit=1, reserve=0)
        (out / ("canary-summary.json" if args.canary else "ab-summary.json")).write_text(json.dumps(reports, indent=2))
        (out / "verdict.json").write_text(json.dumps({"verdict": "PASS", "canary": args.canary, "k": list(ks)}, indent=2))
    except BaseException as error:
        message = str(error)
        verdict = "hazard" if "Xid" in message or "fallen off" in message else "budget" if "budget" in message else "INVALID"
        (out / "verdict.json").write_text(json.dumps({"verdict": verdict, "error": message}, indent=2))
        raise
    finally:
        h.stop()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "tandem":
        tandem_main()
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("trace", "ab"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--canary", action="store_true", help="one patch boot, one long request per kind and two cached follow-ups; budget <=600 seconds")
    p.add_argument("--trace-patch", action="store_true", help="unsupported for graph reuse AB; use trace mode separately")
    p.add_argument("--up", type=Path, default=ROOT / "codex/frozen/graph-up")
    p.add_argument("--patch", type=Path, default=ROOT / "codex/frozen/graph-patch")
    p.add_argument("--out", type=Path)
    p.add_argument("--k", type=int, default=int(os.environ.get("K", "5")))
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--budget", type=int, default=2400)
    p.add_argument("--port", type=int, default=8398)
    p.add_argument("--short-requests", type=int, default=8)
    p.add_argument("--diagnostic-trace", action="store_true", help="trace both A/B arms to attribute startup builds; timings include trace overhead")
    args = p.parse_args()
    if args.k not in (3, 5) or args.rounds < 40 or not 120 < args.budget <= 2400 or not 8 <= args.short_requests <= 16:
        raise ValueError("invalid K, round minimum, or budget")
    if args.canary and (args.mode != "ab" or args.budget > 600):
        raise ValueError("canary needs ab mode and budget <=600")
    if args.mode == "ab" and args.trace_patch:
        raise ValueError("graph reuse A/B requires trace-off arms")
    estimated_boot_s = 45 + 15 + 15 + 35 * (2 if args.canary else 4) + 20 * (2 if args.canary else args.short_requests)
    estimated_s = estimated_boot_s * (1 if args.canary else 4)
    if args.mode == "ab" and estimated_s > 0.7 * args.budget:
        raise ValueError("stage estimates exceed 70% of the wall budget")
    frozen(args.patch)
    if args.mode == "ab":
        frozen(args.up)
        if (args.up / "SOURCE").read_text().splitlines()[0] != STACK2:
            raise ValueError("up must be the frozen stack2 base")
        for case in ("tool", "think"):
            body = json.loads((INPUTS / f"long-{case}.json").read_text())
            if body["max_tokens"] < max(2048, (args.k + 1) * (args.rounds + 5)) or body["cache_prompt"]:
                raise ValueError("long workload token budget is too short or cache is enabled")
    boots = ("patch",) if args.canary else ("up", "patch", "patch", "up")
    print(json.dumps({"mode": args.mode, "canary": args.canary, "boots": ["trace"] if args.mode == "trace" else list(boots),
                      "command": command(args.patch / "bin/round-trace", args.k, args.port), "minimum_steady_rounds": args.rounds,
                      "budget_s_G": args.budget, "stage_estimate_s_G": estimated_s if args.mode == "ab" else None, "capture": ("trace-both" if args.diagnostic_trace else "trace-off") + " " + "/".join(boots) if args.mode == "ab" else "warmup, think/tool, trace",
                      "ab_workload": {"long": "long-tool, long-think; " + ("seed 1" if args.canary else "seeds 1 and 2") + "; one write_file content with 256 explicit numbered lines; 2048 max tokens each; SSE server timings",
                                      "cold": "original tool request immediately after warmup; seed 1; cache_prompt=false",
                                      "short": f"{2 if args.canary else args.short_requests} sequential agent follow-ups; cache_prompt=true; 768 max tokens",
                                      "steady_gate": "at least 40 adjacent full-K interior SSE intervals and 2048 completion tokens per long request; visible write_file arguments; validate after all boot requests are recorded",
                                      "fixed_cost": "report prompt, generation excess, and outside prompt/generation separately"}}, indent=2))
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
        references = list(canary_references(args.k).values()) if args.canary else []
        pins = {str(p): sha(p) for p in [VOCAB, *INPUTS.glob("*.json"), INPUTS / "chat-template.jinja", *references]}
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
            for index, arm in enumerate(boots, 1):
                root = args.up if arm == "up" else args.patch
                dest = h.boot(root, f"{index}-{arm}", trace=args.diagnostic_trace)
                result = suite(h, dest)
                failures = validate_boot(result, dest, args.k, args.rounds)
                results.append(result)
                (out / "ab-results.json").write_text(json.dumps(results, indent=2))
                h.stop()
                h.xid()
                frozen(dest)
                if args.diagnostic_trace:
                    rows = h.rows(dest, 0)
                    (dest / "request-builds.json").write_text(json.dumps({case: builds(rows, record, args.k) for case, record in result.items()}, indent=2))
                if failures:
                    raise ValueError("boot validation failed after writing results: " + json.dumps(failures))
            for case in results[0]:
                expected = identity(results[0][case])
                if any(identity(result[case]) != expected for result in results[1:]):
                    raise ValueError("A/B completion or authoritative round/counter identity failed: " + case)
            (out / "ab-results.json").write_text(json.dumps(results, indent=2))
            if args.canary:
                mismatch = canary_identity(results[0], args.k)
                (out / "canary-validation.json").write_text(json.dumps({"passed": not mismatch, "k": args.k,
                    "cases": list(results[0]), "identity_failures": mismatch}, indent=2))
                if mismatch:
                    raise ValueError("canary differs from recorded frozen up: " + ", ".join(mismatch))
            else:
                (out / "ab-summary.json").write_text(json.dumps(summarize(results, args.k), indent=2))
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
