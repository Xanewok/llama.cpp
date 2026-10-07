"""GPU proof: package 16's candidate path is token-identical to host sampling under -sm tensor, np1 and np2.

Usage: bg_gpu_check.py {--preflight|--canary|--run} --out DIR [--port N]
Same frozen/stack7 binary for both arms. cand = swift15-j5q(-duo) serving flags; host = the same without
--backend-sampling (draft settings unchanged so drafts, verify batches and rounds stay comparable).
Production requests: tools + lazy grammar + reasoning budget, no logit_bias, no ignore_eos.

Gates (summary.json, then exit code):
- identity (rc 8 "cross-arm identity failed"): per request/stream, reasoning, content, tool calls and completion
  tokens equal across arms. np2 pairs are decidable only when the schedule guard holds in both arms (B posted
  >= 30 ms after A and >= 50 ms before A's prefill ends), else excluded; < 4 decidable pairs -> rc 5.
- engagement (rc 6 "path-engaged proof failed"): round-trace "sampling" host time per verify round, median:
  cand < 150 us and host >= 4x cand; in-boot negative control (top_k = 0) >= 4x cand and logs the disable warning;
  cand boots log no disable or SPLIT_MODE_TENSOR refusal warning.
- comparator: cand(seed0, auto) != host(seed1, auto), or the identity check is blind (rc 6).
- grammar engaged: >= 4 of the 6 cand np1 requests end in a parsed tool call, else rc 5 "proof inconclusive".
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

FROZEN = Path("/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack7")
FROZEN_COMMIT = "48b82b0d1559e936dbf612c871f2583cd4c79f3e"
SERVER = FROZEN / "build/bin/llama-server"
MODEL = "/home/xanewok/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf"
TEMPLATE = ("/home/xanewok/repos/qwen38-eval/results/session2/pair-5060ti-2060s/campaign-swift-fullsuite-20260915/"
            "swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja")
VOCAB = "/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
GPUS = "GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f"

# `PROFILE=swift15-j5q[-duo] serve.sh command` (2026-10-08), minus --host/--port
COMMON = ["-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-fa", "on", "--spec-type", "draft-mtp",
          "--spec-draft-n-max", "5", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
          "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "-sm", "tensor", "--tensor-split", "50,50",
          "-b", "2048", "--jinja", "--chat-template-file", TEMPLATE,
          "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05", "--metrics",
          "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling"]
TAIL = ["-t", "8", "-tb", "8", "--spec-draft-vocab", VOCAB]
PROFILES = {
    1: {"args": ["-c", "102400", "-np", "1", "-ub", "512"], "env": {"LLAMA_MTP_CHAIN": "1", "LLAMA_MTP_FOLD_C": "1"}},
    2: {"args": ["-c", "204800", "-np", "2", "-ub", "128"], "env": {"LLAMA_MTP_CHAIN": "0", "LLAMA_MTP_FOLD_C": "0"}},
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "exec_command",
        "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "required": ["cmd"],
                       "properties": {"cmd": {"type": "string"}, "workdir": {"type": "string"}}},
    },
}, {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": "Record the current plan.",
        "parameters": {"type": "object", "required": ["plan"], "properties": {"plan": {"type": "string"}}},
    },
}]
SYSTEM = "You are a coding agent working in /work. Act through tools."
TASKS = [
    "List the files in /tmp, sorted by size, using the exec_command tool.",
    "Find every Python file under /work that imports asyncio, using the exec_command tool.",
]
SEEDS = [1, 7, 1729]
MAX_TOKENS = 2048
DEADLINE = [float("inf")]
DISABLE = ("backend sampling is not compatible with", "backend sampling with SPLIT_MODE_TENSOR needs",
           "backend sampling not supported")


def log(msg):
    print(msg, flush=True)


def gpu_mem_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        uuid, mib = [x.strip() for x in line.split(",")]
        used[uuid] = int(mib)
    return [used.get(u, 0) for u in GPUS.split(",")]


def wait_gpus_free(limit_s=180):
    end = time.time() + limit_s
    while time.time() < end:
        if max(gpu_mem_used()) < 1500:
            return
        time.sleep(3)
    log(f"GPU memory still held after {limit_s} s: {gpu_mem_used()}")


def post(port, body, timeout=1200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:2000].decode(errors='replace')}"
    except Exception as e:  # recorded, gates decide later
        return None, repr(e)


class Boot:
    def __init__(self, out, name, np, cand, port):
        self.dir = out / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name, self.np, self.cand, self.port = name, np, cand, port
        self.log_path = self.dir / "server.log"
        self.trace = self.dir / "round-trace.jsonl"
        args = [str(SERVER), "-m", MODEL] + PROFILES[np]["args"] + COMMON
        if cand:
            args.append("--backend-sampling")
        args += TAIL + ["--host", "127.0.0.1", "--port", str(port)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS, GGML_SCHED_INPUT_STAGING="1", LLAMA_ARG_KV_PAD="4096",
                   LLAMA_ROUND_TRACE=str(self.trace), **PROFILES[np]["env"])
        (self.dir / "cmd.json").write_text(json.dumps({"argv": args, "env": {k: env[k] for k in (
            "CUDA_VISIBLE_DEVICES", "GGML_SCHED_INPUT_STAGING", "LLAMA_ARG_KV_PAD", "LLAMA_MTP_CHAIN",
            "LLAMA_MTP_FOLD_C", "LLAMA_ROUND_TRACE")}}, indent=1))
        wait_gpus_free()
        self.logf = open(self.log_path, "wb")
        self.proc = subprocess.Popen(args, stdout=self.logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        (self.dir / "server.pid").write_text(f"{self.proc.pid}\n")
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"{name}: server exited rc={self.proc.returncode} during load")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"{name}: server not healthy after 600 s")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.logf.close()

    def log_slice(self, start):
        return self.log_path.read_bytes()[start:].decode(errors="replace")


def request(boot, key, task, seed, choice, extra=None):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}],
            "tools": TOOLS, "tool_choice": choice, "seed": seed, "max_tokens": MAX_TOKENS, "cache_prompt": False,
            **(extra or {})}
    t0 = time.monotonic_ns()
    res, err = post(boot.port, body)
    t1 = time.monotonic_ns()
    row = {"boot": boot.name, "key": key, "seed": seed, "tool_choice": choice, "extra": extra or {},
           "t0_ns": t0, "t1_ns": t1, "error": err}
    if res:
        msg = res["choices"][0]["message"]
        tm = res.get("timings", {})
        row.update({
            "finish": res["choices"][0].get("finish_reason"),
            "completion_tokens": res.get("usage", {}).get("completion_tokens"),
            "reasoning": msg.get("reasoning_content"), "content": msg.get("content"),
            "tool_calls": [[c["function"]["name"], c["function"]["arguments"]] for c in msg.get("tool_calls") or []],
            "prompt_ms": tm.get("prompt_ms"), "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
        })
    return row


def run_np1(boot, rows, seeds, choices, neg):
    for choice in choices:
        for seed in seeds:
            if time.time() > DEADLINE[0]:
                return
            start = boot.log_path.stat().st_size
            r = request(boot, f"np1/{choice}/{seed}", TASKS[0], seed, choice)
            r["disable_warnings"] = sum(boot.log_slice(start).count(d) for d in DISABLE)
            rows.append(r)
            log(json.dumps({k: r.get(k) for k in ("boot", "key", "finish", "completion_tokens", "tool_calls", "error")}))
    if neg:
        start = boot.log_path.stat().st_size
        r = request(boot, "neg-topk0", TASKS[0], seeds[0], "auto", {"top_k": 0})
        r["disable_warnings"] = sum(boot.log_slice(start).count(d) for d in DISABLE)
        rows.append(r)
        log(json.dumps({k: r.get(k) for k in ("boot", "key", "finish", "completion_tokens", "disable_warnings")}))


def run_np2(boot, rows, seeds, choices):
    for choice in choices:
        for seed in seeds:
            if time.time() > DEADLINE[0]:
                return
            start = boot.log_path.stat().st_size
            out = {}

            def go(stream, task, s, delay):
                time.sleep(delay)
                out[stream] = request(boot, f"np2/{choice}/{seed}/{stream}", task, s, choice)

            ta = threading.Thread(target=go, args=("A", TASKS[0], seed, 0.0))
            tb = threading.Thread(target=go, args=("B", TASKS[1], seed + 1000, 0.05))
            ta.start(); tb.start(); ta.join(); tb.join()
            warn = sum(boot.log_slice(start).count(d) for d in DISABLE)
            a, b = out["A"], out["B"]
            gap_ms = (b["t0_ns"] - a["t0_ns"]) / 1e6
            guard = a.get("prompt_ms") is not None and gap_ms >= 30 and gap_ms <= a["prompt_ms"] - 50
            for r in (a, b):
                r.update({"pair_gap_ms": round(gap_ms, 2), "schedule_guard": guard, "disable_warnings": warn})
                rows.append(r)
            log(json.dumps({"boot": boot.name, "pair": f"{choice}/{seed}", "gap_ms": round(gap_ms, 1),
                            "prompt_ms_A": a.get("prompt_ms"), "guard": guard,
                            "tokens": [a.get("completion_tokens"), b.get("completion_tokens")],
                            "errors": [a["error"], b["error"]]}))


def trace_rounds(boot):
    """(begin_ns, sampling_us) per verify round."""
    rounds = []
    if not boot.trace.exists():
        return rounds
    for line in boot.trace.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ph = r.get("phases", [])
        if not any(p["phase"] == "V" for p in ph):
            continue
        us = sum(c["host_us"] for p in ph for c in p["counters"] if c["kind"] == "sampling")
        rounds.append((ph[0]["begin_ns"], us))
    return rounds


def sampling_median(rounds, rows):
    spans = [(r["t0_ns"], r["t1_ns"]) for r in rows]
    v = [us for b, us in rounds if any(t0 <= b <= t1 for t0, t1 in spans)]
    return (round(statistics.median(v), 1) if v else None), len(v)


def verify_frozen():
    src = (FROZEN / "SOURCE").read_text().split()[0]
    if src != FROZEN_COMMIT:
        log(f"frozen bytes changed: SOURCE {src}")
        sys.exit(6)
    r = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=FROZEN / "build/bin", capture_output=True, text=True)
    if r.returncode:
        log("frozen bytes changed:\n" + r.stdout + r.stderr)
        sys.exit(6)


def ident(r):
    return (r.get("reasoning"), r.get("content"), r.get("tool_calls"), r.get("completion_tokens"), r.get("finish"))


def evaluate(rows, boots, canary):
    by = {(r["boot"], r["key"]): r for r in rows}
    res = {"identity": [], "excluded": [], "engagement": {}, "failures": [], "insufficient": []}
    errors = [f"{r['boot']} {r['key']}: {r['error']}" for r in rows if r["error"]]
    if errors:
        res["insufficient"] += errors
    pairs = []
    for (bname, key), r in by.items():
        if bname.startswith("cand"):
            h = by.get((bname.replace("cand", "host"), key))
            if h and not key.startswith("neg"):
                pairs.append((r, h))
    np2_decidable = 0
    for c, h in pairs:
        if c["key"].startswith("np2") and not (c["schedule_guard"] and h["schedule_guard"]):
            res["excluded"].append(c["key"])
            continue
        if c["key"].startswith("np2") and c["key"].endswith("/A"):
            np2_decidable += 1
        same = ident(c) == ident(h)
        res["identity"].append({"key": c["key"], "np": c["boot"][-1], "equal": same,
                                "tokens": [c.get("completion_tokens"), h.get("completion_tokens")],
                                "draft": [[c.get("draft_n"), c.get("draft_n_accepted")], [h.get("draft_n"), h.get("draft_n_accepted")]]})
        if not same:
            res["failures"].append(f"cross-arm identity failed: {c['boot']} {c['key']}")
    if not canary and any(b.endswith("2") for b in boots) and np2_decidable < 4:
        res["insufficient"].append(f"np2 decidable pairs {np2_decidable} < 4")
    # comparator: the identity check must see a seed change
    c0 = by.get(("cand-np1", f"np1/auto/{SEEDS[0]}"))
    h1 = by.get(("host-np1", f"np1/auto/{SEEDS[1]}"))
    if c0 and h1 and ident(c0) == ident(h1):
        res["failures"].append("path-engaged proof failed: comparator pair (different seeds) compares equal")
    # engagement
    for np in (1, 2):
        cb, hb = boots.get(f"cand-np{np}"), boots.get(f"host-np{np}")
        if not cb:
            continue
        prod = [r for r in rows if r["boot"] == cb.name and not r["key"].startswith("neg")]
        cm, cn = sampling_median(trace_rounds(cb), prod)
        e = {"cand_median_us": cm, "cand_rounds": cn,
             "cand_disable_warnings": sum(r["disable_warnings"] for r in prod)}
        if hb:
            hm, hn = sampling_median(trace_rounds(hb), [r for r in rows if r["boot"] == hb.name])
            e.update({"host_median_us": hm, "host_rounds": hn})
            if hm is None or cm is None or not (cm < 150 and hm >= 4 * cm):
                res["failures"].append(f"path-engaged proof failed: np{np} sampling medians cand {cm} host {hm}")
        elif cm is None or cm >= 150:
            res["failures"].append(f"path-engaged proof failed: np{np} cand median {cm}")
        if e["cand_disable_warnings"]:
            res["failures"].append(f"path-engaged proof failed: np{np} cand boot logged a disable/refusal warning")
        neg = [r for r in rows if r["boot"] == cb.name and r["key"].startswith("neg")]
        if neg:
            nm, nn = sampling_median(trace_rounds(cb), neg)
            e.update({"neg_median_us": nm, "neg_rounds": nn, "neg_disable_warnings": neg[0]["disable_warnings"]})
            if not neg[0]["disable_warnings"] or nm is None or cm is None or nm < 4 * cm:
                res["failures"].append(f"path-engaged proof failed: np{np} negative control (top_k 0) warn "
                                       f"{neg[0]['disable_warnings']} median {nm}")
        res["engagement"][f"np{np}"] = e
    calls = [r for r in rows if r["boot"] == "cand-np1" and r["key"].startswith("np1")]
    n_calls = sum(1 for r in calls if r.get("tool_calls"))
    res["cand_np1_tool_calls"] = f"{n_calls}/{len(calls)}"
    if n_calls < (1 if canary else 4):
        res["insufficient"].append(f"grammar engaged in only {n_calls}/{len(calls)} cand np1 requests")
    return res


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--canary", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18450)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in (SERVER, MODEL, TEMPLATE, VOCAB):
        if not Path(p).is_file():
            log(f"missing input {p}")
            sys.exit(6)
    verify_frozen()
    if a.preflight:
        h = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (out / "preflight.json").write_text(json.dumps({"ok": True, "script_sha256": h}))
        log("preflight ok")
        return

    if a.canary:
        plan = [("cand-np1", 1, True, SEEDS[:1], ("auto",), True), ("cand-np2", 2, True, SEEDS[:1], ("auto",), False)]
    else:
        plan = []
        for np in (1, 2):
            for cand in (True, False):
                plan.append((f"{'cand' if cand else 'host'}-np{np}", np, cand, SEEDS, ("auto", "required"), cand and np == 1))
    rows, boots = [], {}
    deadline = time.time() + (520 if a.canary else 1650)
    DEADLINE[0] = deadline
    for name, np, cand, seeds, choices, neg in plan:
        if time.time() > deadline:
            log(f"budget: skipping {name}")
            break
        try:
            boot = Boot(out, name, np, cand, a.port)
        except RuntimeError as e:
            log(str(e))
            (out / "summary.json").write_text(json.dumps({"pass": False, "boot_error": str(e), "rows": rows}, indent=1))
            sys.exit(4)
        boots[name] = boot
        try:
            if np == 1:
                run_np1(boot, rows, seeds, choices, neg)
            else:
                run_np2(boot, rows, seeds, choices)
        finally:
            boot.stop()
            time.sleep(2)

    res = evaluate(rows, boots, a.canary)
    expected = sum(len(p[3]) * len(p[4]) * p[1] + (1 if p[5] else 0) for p in plan)
    complete = len(boots) == len(plan) and len(rows) == expected
    if not complete:
        res["insufficient"].append(f"budget: {len(boots)}/{len(plan)} boots, {len(rows)}/{expected} requests")
    res["pass"] = not res["failures"] and not res["insufficient"]
    res["rows"] = rows
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    if res["failures"]:
        identity = any(f.startswith("cross-arm identity failed") for f in res["failures"])
        sys.exit(8 if identity else 6)
    if res["insufficient"]:
        log("proof inconclusive")
        sys.exit(5)


if __name__ == "__main__":
    main()
