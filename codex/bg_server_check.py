"""CPU llama-server check: tool-call grammar + reasoning budget keep the backend candidate path, token-identical to host.

Usage: bg_server_check.py SERVER_BIN MODEL OUT_DIR
One boot with --backend-sampling; per request backend_sampling true (candidate path) vs false (host path), same seed.
Gates: identity A == B per (seed, tool_choice); engagement (candidate rows only in A, none in B; no disable warning in A);
negative control top_k = 0 must log the disable warning; comparator control: A(seed) != B(other seed).
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

SERVER, MODEL, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
SEEDS = [int(s) for s in os.environ.get("BG_SEEDS", "1 7 1729 31337").split()]
MAX_TOKENS = int(os.environ.get("BG_MAX_TOKENS", "1024"))
TOOLS = [{
    "type": "function",
    "function": {
        "name": "exec_command",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}, "workdir": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": "Record the current plan.",
        "parameters": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]},
    },
}]
MESSAGES = [
    {"role": "system", "content": "You are a coding agent. Act through tools. Keep reasoning short."},
    {"role": "user", "content": "List the files in /tmp, sorted by size, using the exec_command tool."},
]
DISABLE = "backend sampling is not compatible with"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "server.log"
    port = free_port()
    cmd = [SERVER, "-m", MODEL, "-c", "8192", "-np", "1", "-t", "4", "-tb", "4", "--jinja",
           "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05", "--backend-sampling",
           "--cache-ram", "0", "-lv", "4", "--host", "127.0.0.1", "--port", str(port)]
    (OUT / "server-cmd.txt").write_text(" ".join(cmd) + "\n")
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if proc.poll() is not None:
                raise SystemExit(f"server died rc={proc.returncode}")
            time.sleep(1)
        else:
            raise SystemExit("server not healthy")

        rows = []

        def run(name, seed, choice, extra):
            body = {"messages": MESSAGES, "tools": TOOLS, "tool_choice": choice, "seed": seed,
                    "max_tokens": MAX_TOKENS, "cache_prompt": False, **extra}
            log.flush()
            start = log_path.stat().st_size
            t0 = time.time()
            res = post(port, body)
            dt = time.time() - t0
            time.sleep(0.3)
            seg = log_path.read_bytes()[start:].decode(errors="replace")
            msg = res["choices"][0]["message"]
            row = {
                "arm": name, "seed": seed, "tool_choice": choice, "extra": extra, "seconds": round(dt, 1),
                "finish": res["choices"][0].get("finish_reason"),
                "completion_tokens": res.get("usage", {}).get("completion_tokens"),
                "reasoning": msg.get("reasoning_content"), "content": msg.get("content"),
                "tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in msg.get("tool_calls") or []],
                "candidate_rows": seg.count("candidate row, n ="),
                "grammar_resamples": seg.count("grammar rejected token"),
                "disable_warnings": seg.count(DISABLE),
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("arm", "seed", "tool_choice", "seconds", "finish", "completion_tokens",
                                                  "tool_calls", "candidate_rows", "grammar_resamples",
                                                  "disable_warnings")}), flush=True)
            return row

        for choice in ("auto", "required"):
            for seed in SEEDS:
                run("cand", seed, choice, {"backend_sampling": True})
                run("host", seed, choice, {"backend_sampling": False})
        neg = run("neg-topk0", SEEDS[0], "auto", {"backend_sampling": True, "top_k": 0})
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    key = lambda r: (r["reasoning"], r["content"], r["tool_calls"], r["completion_tokens"])
    by = {(r["arm"], r["seed"], r["tool_choice"]): r for r in rows}
    failures = []
    for choice in ("auto", "required"):
        for seed in SEEDS:
            a, b = by[("cand", seed, choice)], by[("host", seed, choice)]
            if key(a) != key(b):
                failures.append(f"identity {choice} seed {seed}")
            if a["candidate_rows"] == 0 or a["disable_warnings"]:
                failures.append(f"engagement cand {choice} seed {seed}")
            if b["candidate_rows"]:
                failures.append(f"host arm read candidates {choice} seed {seed}")
    a, b = by[("cand", SEEDS[0], "auto")], by[("host", SEEDS[1], "auto")]
    if key(a) == key(b):
        failures.append("comparator control: different seeds compare equal")
    if not neg["disable_warnings"] or neg["candidate_rows"]:
        failures.append("negative control top_k=0 did not disable the candidate path")
    summary = {
        "pass": not failures, "failures": failures, "seeds": SEEDS,
        "tool_calls_cand": sum(len(r["tool_calls"]) for r in rows if r["arm"] == "cand"),
        "candidate_rows_cand": sum(r["candidate_rows"] for r in rows if r["arm"] == "cand"),
        "grammar_resamples_cand": sum(r["grammar_resamples"] for r in rows if r["arm"] == "cand"),
        "grammar_resamples_host": sum(r["grammar_resamples"] for r in rows if r["arm"] == "host"),
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
