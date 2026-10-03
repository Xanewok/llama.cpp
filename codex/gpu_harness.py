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
import statistics
import subprocess
import time

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
        data = self.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                         "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        used = [int(line.strip()) for line in data.splitlines()]
        if len(used) != 2 or sum(used) > 1500:
            raise RuntimeError("selected GPUs are busy or not a pair")
        utilization = self.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                                "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        if any(int(line.strip()) != 0 for line in utilization.splitlines()) or len(utilization.splitlines()) != 2:
            raise RuntimeError("selected GPUs are not idle")
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
        self.request(dest, "warmup", 0)
        return dest

    def request(self, dest, case, number, body=None, stream=False):
        body = json.loads((INPUTS / f"{case}.json").read_text()) if body is None else body
        body = dict(body)
        if stream:
            body.update(stream=True, timings_per_token=True, stream_options={"include_usage": True})
        (dest / f"{case}-{number}-input.json").write_text(json.dumps(body, indent=2))
        before = self.metrics()
        begin = time.monotonic_ns()
        if stream:
            import urllib.request
            request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/v1/chat/completions",
                json.dumps(body).encode(), {"Content-Type": "application/json"})
            chunks = []
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
            chunks = []
            response = self.http("/v1/chat/completions", body)
        end = time.monotonic_ns()
        after = self.metrics()
        record = {"case": case, "max_tokens": body["max_tokens"], "begin_ns": begin, "end_ns": end, "response": response,
                  "metrics": {key: after[key] - before[key] for key in before}, "chunks": chunks}
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


def completion(chunks):
    message = {"role": "assistant", "content": ""}
    calls = {}
    result = {"choices": [{"message": message, "finish_reason": None}]}
    for chunk in chunks:
        if "error" in chunk:
            raise ValueError("SSE completion failed: " + str(chunk["error"]))
        for key in ("timings", "usage"):
            if key in chunk:
                result[key] = chunk[key]
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
        timing = chunk.get("timings", {})
        n = timing.get("draft_n")
        if n is None:
            continue
        ms = timing.get("predicted_ms", 0)
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
    for seed in (1, 2):
        for case in ("tool", "think"):
            body = json.loads((INPUTS / f"long-{case}.json").read_text())
            body["seed"] = seed
            label = f"long-{case}-s{seed}"
            record = h.request(dest, label, 1, body=body, stream=True)
            record["steady"] = steady(record, h.args.k, h.args.rounds)
            (dest / f"{label}-steady.json").write_text(json.dumps(record["steady"], indent=2))
            results[label] = record
    body = json.loads((INPUTS / "tool.json").read_text())
    body.update(cache_prompt=True, max_tokens=128)
    for number in range(1, h.args.short_requests + 1):
        label = f"short-tool-{number}"
        record = h.request(dest, label, 1, body=body)
        if number > 1 and record["response"]["timings"]["cache_n"] <= 0:
            raise ValueError("agent follow-up did not reuse the prompt cache")
        results[label] = record
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


def summarize(results, k):
    summary = []
    for index, result in enumerate(results):
        reference = statistics.mean(result[f"long-tool-s{s}"]["steady"]["steady_ms_per_round_D"] for s in (1, 2))
        for case, record in result.items():
            timings = record["response"]["timings"]
            rounds = record["metrics"]["drafts"]
            if rounds <= 0:
                raise ValueError("no authoritative MTP rounds")
            row = {"boot": index + 1, "arm": "up" if index in (0, 3) else "patch", "case": case,
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
    for case in results[0]:
        rows = [r for r in summary if r["case"] == case]
        keys = [key for key in rows[0] if "ms" in key and key.endswith(("_M", "_D"))]
        pairs = [{key: rows[p][key] - rows[u][key] for key in keys} for u, p in ((0, 1), (3, 2))]
        paired.append({"case": case, "pair_differences_D": pairs,
                       "patch_minus_up_D": {key: statistics.mean(pair[key] for pair in pairs) for key in keys}})
    return {"requests": summary, "paired": paired,
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("trace", "ab"))
    p.add_argument("--execute", action="store_true")
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
    if args.mode == "ab" and args.trace_patch:
        raise ValueError("graph reuse A/B requires trace-off arms")
    frozen(args.patch)
    if args.mode == "ab":
        frozen(args.up)
        if (args.up / "SOURCE").read_text().splitlines()[0] != STACK2:
            raise ValueError("up must be the frozen stack2 base")
        for case in ("tool", "think"):
            body = json.loads((INPUTS / f"long-{case}.json").read_text())
            if body["max_tokens"] < (args.k + 1) * (args.rounds + 5) or body["cache_prompt"]:
                raise ValueError("long workload token budget is too short or cache is enabled")
    print(json.dumps({"mode": args.mode, "boots": ["trace"] if args.mode == "trace" else ["up", "patch", "patch", "up"],
                      "command": command(args.patch / "bin/round-trace", args.k, args.port), "minimum_steady_rounds": args.rounds,
                      "budget_s_G": args.budget, "capture": ("trace-both" if args.diagnostic_trace else "trace-off") + " up/patch/patch/up" if args.mode == "ab" else "warmup, think/tool, trace",
                      "ab_workload": {"long": "long-tool, long-think; seeds 1 and 2; 2048 max tokens each; SSE server timings",
                                      "cold": "original tool request immediately after warmup; seed 1; cache_prompt=false",
                                      "short": f"{args.short_requests} sequential agent follow-ups; cache_prompt=true; 128 max tokens",
                                      "steady_gate": "at least 40 adjacent full-K interior SSE intervals per long request",
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
                dest = h.boot(root, f"{index}-{arm}", trace=args.diagnostic_trace)
                result = suite(h, dest)
                results.append(result)
                (out / "ab-results.json").write_text(json.dumps(results, indent=2))
                h.stop()
                h.xid()
                frozen(dest)
                if args.diagnostic_trace:
                    rows = h.rows(dest, 0)
                    (dest / "request-builds.json").write_text(json.dumps({case: builds(rows, record, args.k) for case, record in result.items()}, indent=2))
            for case in results[0]:
                expected = identity(results[0][case])
                if any(identity(result[case]) != expected for result in results[1:]):
                    raise ValueError("A/B completion or authoritative round/counter identity failed: " + case)
            (out / "ab-results.json").write_text(json.dumps(results, indent=2))
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
