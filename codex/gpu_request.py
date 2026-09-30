#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import pathlib
import sys
import urllib.request

smoke, port, kind, output, timeout = sys.argv[1:]
root = pathlib.Path(output)
base = f"http://127.0.0.1:{port}"
spec = importlib.util.spec_from_file_location("codex_smoke", smoke)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def call(path, body):
    return cs.call(base + path, body, timeout=float(timeout))


def metrics():
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        text = response.read().decode()
    wanted = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total")
    values = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, value = line.split()[:2]
            for key in wanted:
                if name == "llamacpp:" + key:
                    values[key] = float(value)
    if len(values) != len(wanted):
        raise RuntimeError("missing speculative round counters")
    return values


prompt_tokens = None
if kind == "long":
    prompt_file = root / "long-prompt.json"
    if not prompt_file.exists():
        block = "Read this log entry and retain its order: alpha beta gamma delta.\n"
        lo, hi = 1, 1200
        while lo < hi:
            mid = (lo + hi) // 2
            prompt = block * mid + "Summarize the pattern in a few sentences."
            count = len(call("/tokenize", {"content": prompt, "add_special": False})["tokens"])
            if count < 8000:
                lo = mid + 1
            else:
                hi = mid
        prompt = block * lo + "Summarize the pattern in a few sentences."
        count = len(call("/tokenize", {"content": prompt, "add_special": False})["tokens"])
        assert 7900 <= count <= 8100, count
        prompt_file.write_text(json.dumps({"text": prompt, "tokens": count}))
    data = json.loads(prompt_file.read_text())
    messages = [{"role": "user", "content": data["text"]}]
    prompt_tokens = data["tokens"]
else:
    messages = dict(cs.conversations())["fresh" if kind == "warmup" else "follow"]
body = {"messages": messages, "tools": cs.TOOLS, "tool_choice": "auto", "seed": 1,
        "cache_prompt": False, "max_tokens": {"warmup": 128, "tool": 384, "think": 1024, "long": 128}[kind],
        "chat_template_kwargs": {"enable_thinking": kind in ("warmup", "think")}}
before = metrics()
r = call("/v1/chat/completions", body)
after = metrics()
m = r["choices"][0]["message"]
message = {"content": m.get("content"), "reasoning_content": m.get("reasoning_content"),
           "tool_calls": [c["function"] for c in m.get("tool_calls") or []]}
digest = hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()
print(json.dumps({"timings": r["timings"], "sha256": digest, "message": message,
                  "counters": {key: after[key] - before[key] for key in before},
                  "long_content_tokens": prompt_tokens}))
