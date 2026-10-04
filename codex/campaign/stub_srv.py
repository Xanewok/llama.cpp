#!/usr/bin/env python3
"""CPU stand-in for the MTP server: llama-server-style argv, /health /props /metrics, and /stub/generate that simulates
draft rounds. Never named llama-server. Defect injection for negative controls (env):
  STUB_IGNORE_PMIN=1  accept --spec-draft-p-min but never act on it
  STUB_FORCE_NMAX=N   draft N regardless of --spec-draft-n-max
  STUB_LIE=1          the startup echo reports the requested params instead of the effective ones
  STUB_NO_ECHO=1      print no params echo (the real build prints none at its default verbosity)
Canary surface (/tokenize, /completion with seeded tokens and timings, SPLIT debug lines, round-trace counters), arm from STUB_ARM:
  STUB_B_FULL_MASK=1  arm B uploads full decode masks (compact path not engaged)
  STUB_NO_STAGED=1    arm B's round trace never counts input_fence_staged
  STUB_DIVERGE=1      arm B's tokens differ from A's;  STUB_DRAFT_DIVERGE=1  its draft counter differs
"""
import zlib
import json
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse(argv):
    values, i = {}, 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            values.setdefault(token, []).append(argv[i + 1])
            i += 2
        else:
            values.setdefault(token, []).append(True)
            i += 1
    return values


class State:
    def __init__(self, args):
        one = lambda flag, default: args.get(flag, [default])[-1]
        self.port = int(one("--port", 8081))
        self.slots = int(one("-np", 1))
        self.ctx = int(one("-c", 4096)) // self.slots
        self.req_k, self.req_p = int(one("--spec-draft-n-max", 3)), float(one("--spec-draft-p-min", 0))
        self.k = int(os.environ["STUB_FORCE_NMAX"]) if os.environ.get("STUB_FORCE_NMAX") else self.req_k
        self.p = 0.0 if os.environ.get("STUB_IGNORE_PMIN") == "1" else self.req_p
        self.model = one("-m", "stub.gguf")
        self.lock = threading.Lock()
        self.c = dict(predicted=0.0, predicted_s=0.0, drafts=0.0, draft_tokens=0.0, accepted=0.0, prompt=0.0, prompt_s=0.0, nmax=0.0, rounds=0.0, plain=0.0)
        self.task = 0
        self.arm = os.environ.get("STUB_ARM", "A")
        self.staging = os.environ.get("GGML_SCHED_INPUT_STAGING") == "1"
        self.debug = os.environ.get("GGML_SCHED_DEBUG") == "1" and "5" in args.get("--verbosity", [])
        self.trace = os.environ.get("LLAMA_ROUND_TRACE")
        self.cached, self.requests = [], 0

    @staticmethod
    def tokenize(text):
        return [zlib.crc32(text[i:i + 3].encode()) % 150000 for i in range(0, len(text), 3)]

    def split_line(self, n_tokens, depth):
        full = depth * n_tokens * 2 // 1024
        compact = self.arm == "B" and os.environ.get("STUB_B_FULL_MASK") != "1" and depth >= 32768 and n_tokens <= 16
        kib = n_tokens // 2 if compact else full
        size = f"{kib // 1024}M" if kib >= 1024 else f"{kib}K"
        return f"## SPLIT #3: Meta [Meta(CUDA0,CUDA1)] # 2 inputs: [attn_inp_kq_mask ({size:>4})] [inp_tokens (  0K)]"

    def completion(self, prompt, n_predict, seed, cache_prompt):
        toks = list(prompt) if isinstance(prompt, list) else self.tokenize(prompt)
        with self.lock:
            self.requests += 1
            common = 0
            for a, b in zip(self.cached, toks):
                if a != b:
                    break
                common += 1
            cache_n = min(common, len(toks) - 1) if cache_prompt else 0
            prompt_n = len(toks) - cache_n
            self.cached = toks
            rng = random.Random(f"{seed}-{zlib.crc32(json.dumps(toks).encode())}-{self.requests if self.requests > 1 else 1}")
            if self.debug:
                for _ in range(max(1, -(-prompt_n // 128))):
                    print(self.split_line(min(128, prompt_n), len(toks)), flush=True)
            out, drafted, accepted, rounds = [], 0, 0, 0
            while len(out) < n_predict:
                length = self.k
                got = 0
                while got < length and rng.random() < 0.7:
                    got += 1
                drafted += length
                accepted += got
                out += [rng.randrange(150000) for _ in range(got + 1)]
                rounds += 1
                if self.debug:
                    print(self.split_line(length + 1, len(toks) + len(out)), flush=True)
            out = out[:n_predict]
            if self.arm == "B" and os.environ.get("STUB_DIVERGE") == "1":
                out[50] += 1
            if self.arm == "B" and os.environ.get("STUB_DRAFT_DIVERGE") == "1":
                drafted += 1
            self.c["drafts"] += rounds
            self.c["draft_tokens"] += drafted
            self.c["accepted"] += accepted
            self.c["predicted"] += len(out)
            if self.trace:
                staged = rounds if self.staging and os.environ.get("STUB_NO_STAGED") != "1" else 0
                counters = [{"kind": "input_fence_staged", "device": 0, "count": staged, "bytes": 0, "host_us": 0},
                            {"kind": "input_fence_sync", "device": 0, "count": 0 if staged else rounds, "bytes": 0, "host_us": 0}]
                with open(self.trace, "a") as stream:
                    stream.write(json.dumps({"phases": [{"counters": counters}]}) + "\n")
        return {"tokens": out, "timings": {"predicted_n": len(out), "draft_n": drafted, "draft_n_accepted": accepted,
                                           "prompt_n": prompt_n, "cache_n": cache_n}}

    def generate(self, n, seed, top1_mean=0.58):
        rng = random.Random(seed)
        made = rounds = 0
        acc_total = gen_total = 0
        secs = 0.0
        with self.lock:
            while made < n:
                length = 0
                for _ in range(self.k):
                    p = min(0.999, max(0.01, rng.betavariate(2.6 * top1_mean / 0.58, 1.9)))
                    if self.p > 0 and p < self.p:
                        break
                    length += 1
                accepted = 0
                if length:
                    for _ in range(length):
                        if rng.random() < 0.55 + 0.4 * rng.random():
                            accepted += 1
                        else:
                            break
                    self.c["drafts"] += 1
                    self.c["draft_tokens"] += length
                    self.c["accepted"] += accepted
                    acc_total += accepted
                    gen_total += length
                if not length:
                    self.c["plain"] += 1
                self.c["rounds"] += 1
                made += accepted + 1
                rounds += 1
                secs += 0.0075 + 0.0009 * length
            self.c["predicted"] += made
            self.c["predicted_s"] += secs
            self.c["prompt"] += 800
            self.c["prompt_s"] += 1.1
            self.c["nmax"] = max(self.c["nmax"], 800 + made)
            self.task += 1
            task = self.task
        line = (f"slot print_timing: id  0 | task {task} |        eval time = {1000 * secs:10.2f} ms / {made:5d} tokens "
                f"({1000 * secs / made:8.2f} ms per token, {made / secs:8.2f} tokens per second)")
        print(line, flush=True)
        print(f"slot print_timing: id  0 | task {task} | draft acceptance = {acc_total / max(gen_total, 1):.5f} "
              f"({acc_total} accepted / {gen_total} generated)", flush=True)
        return made

    def metrics(self):
        c = self.c
        names = [("tokens_predicted_total", c["predicted"]), ("tokens_predicted_seconds_total", c["predicted_s"]),
                 ("prompt_tokens_total", c["prompt"]), ("prompt_seconds_total", c["prompt_s"]),
                 ("prompt_tokens_cached_total", 0), ("n_tokens_max", c["nmax"]),
                 ("spec_decode_num_draft_tokens_total", c["draft_tokens"]),
                 ("spec_decode_num_accepted_tokens_total", c["accepted"]), ("spec_decode_num_drafts_total", c["drafts"])]
        return "".join(f"# TYPE llamacpp:{n} counter\nllamacpp:{n} {v}\n" for n, v in names)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, kind="application/json"):
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send(json.dumps({"status": "ok"}))
            elif self.path == "/props":
                self.send(json.dumps({"total_slots": state.slots, "model_path": state.model,
                                      "default_generation_settings": {"n_ctx": state.ctx, "params": {"speculative.types": "none"}}}))
            elif self.path == "/stub/truth":
                self.send(json.dumps({"rounds": state.c["rounds"], "plain": state.c["plain"]}))
            elif self.path == "/metrics":
                self.send(state.metrics(), "text/plain")
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/stub/generate":
                made = state.generate(int(body.get("n", 100)), int(body.get("seed", 1)), float(body.get("top1_mean", 0.58)))
                self.send(json.dumps({"tokens": made}))
            elif self.path == "/tokenize":
                self.send(json.dumps({"tokens": state.tokenize(body.get("content", ""))}))
            elif self.path == "/completion":
                self.send(json.dumps(state.completion(body["prompt"], int(body["n_predict"]), int(body.get("seed", 1)), bool(body.get("cache_prompt")))))
            else:
                self.send_error(404)

    return Handler


def main():
    args = parse(sys.argv[1:])
    state = State(args)
    echo_k, echo_p = (state.req_k, state.req_p) if os.environ.get("STUB_LIE") == "1" else (state.k, state.p)
    if os.environ.get("STUB_NO_ECHO") != "1":
        print(f"common_speculative_init_result: stub draft params n_max={echo_k}, n_min=0, p_min={echo_p:.2f}", flush=True)
    print(f"srv  llama_server: listening on http://127.0.0.1:{state.port}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", state.port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
