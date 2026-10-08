#!/usr/bin/env python3
"""CPU stand-in for the MTP server (never a GPU process, never named llama-server): llama-server argv, /health /props
/metrics /tokenize, /completion (JSON, or SSE with per-token timings when "stream"), /stub/generate for campaign traffic,
SPLIT debug lines, round-trace counters, KV cache lines at verbosity 5.

Defects hit the arms named in STUB_ARMS (default "B"; arm from STUB_ARM):
  STUB_FULL_MASK=1      decode uploads full masks (compact path not engaged)
  STUB_NO_STAGED=1      the round trace never counts input_fence_staged
  STUB_DRAFT_KV_STUCK=1 the draft KV cache ignores --spec-draft-type-k/v (flag not engaged)
  STUB_DIVERGE=1        tokens differ;  STUB_DRAFT_DIVERGE=1  the draft counter differs
  STUB_DIVERGE_SLOT=N   with STUB_DIVERGE=1, only streams on slot N diverge
Concurrency: STUB_CHUNK_MS=N sleeps N ms per streamed chunk; STUB_SERIALIZE=1 streams one request at a time (no overlap).
STUB_EOS_AT=N         a depth stream whose request has ignore_eos false ends at N tokens (EOS); ignore_eos true logs the real
                      server's TP "using CPU" warning (an EOG logit_bias leaves the chain without a leading top-k)
STUB_ABORT_STREAM=1   every /v1/responses ends without response.completed (an aborted agent call); "stub": {"abort": true} aborts one
Slots: /completion SSE keeps one prompt cache per id_slot (cache_prompt replays all but 132 tokens of an equal prompt);
/v1/responses streams OpenAI Responses events for agent traffic ("stub" in the body sets input, cached, output tokens);
`--smi UUID...` prints `uuid, MiB free` rows like nvidia-smi -lms.
Global: STUB_FORCE_NMAX=N drafts N regardless of --spec-draft-n-max; STUB_IGNORE_PMIN=1; STUB_LIE=1 echoes the requested
params instead of the effective ones; STUB_NO_ECHO=1 prints no echo (the real build is silent at default verbosity).
"""
import json
import os
import random
import sys
import threading
import time
import zlib
import contextlib
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


def say(line):
    print(line, flush=True)


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
        self.arm = os.environ.get("STUB_ARM", "A")
        self.staging = os.environ.get("GGML_SCHED_INPUT_STAGING") == "1"
        verbose = "5" in args.get("--verbosity", []) or os.environ.get("LLAMA_ARG_LOG_VERBOSITY") == "5"
        self.debug = os.environ.get("GGML_SCHED_DEBUG") == "1" and verbose
        self.verbose = verbose
        self.trace = os.environ.get("LLAMA_ROUND_TRACE")
        self.kv = one("-ctk", "f16")
        self.draft_kv = "f16" if self.defect("STUB_DRAFT_KV_STUCK") else one("--spec-draft-type-k", "f16")
        self.lock = threading.Lock()
        self.c = dict(predicted=0.0, predicted_s=0.0, drafts=0.0, draft_tokens=0.0, accepted=0.0, prompt=0.0, prompt_s=0.0, nmax=0.0, rounds=0.0, plain=0.0)
        self.task, self.cached, self.requests = 0, [], 0
        self.slot_cache, self.serial = {}, threading.Lock()

    def defect(self, name):
        return os.environ.get(name) == "1" and self.arm in os.environ.get("STUB_ARMS", "B")

    @staticmethod
    def tokenize(text):
        return [zlib.crc32(text[i:i + 3].encode()) % 150000 for i in range(0, len(text), 3)]

    def compact(self, depth, rows):
        return not self.defect("STUB_FULL_MASK") and self.staging and depth >= 32768 and rows <= 16

    def split_line(self, rows, depth):
        kib = rows // 2 if self.compact(depth, rows) else depth * rows * 2 // 1024
        size = f"{kib // 1024}M" if kib >= 1024 else f"{kib}K"
        return f"## SPLIT #3: Meta [Meta(CUDA0,CUDA1)] # 2 inputs: [attn_inp_kq_mask ({size:>4})] [inp_tokens (  0K)]"

    def staged_counters(self, rounds):
        if not self.trace:
            return
        staged = rounds if self.staging and not self.defect("STUB_NO_STAGED") else 0
        counters = [{"kind": "input_fence_staged", "device": 0, "count": staged, "bytes": 0, "host_us": 0},
                    {"kind": "input_fence_sync", "device": 0, "count": 0 if staged else rounds, "bytes": 0, "host_us": 0},
                    {"kind": "graph_hit", "device": -1, "count": rounds, "bytes": 0, "host_us": 0},
                    {"kind": "ar_wait", "device": 0, "count": rounds, "bytes": 0, "host_us": 10.0 * rounds}]
        with open(self.trace, "a") as stream:
            stream.write(json.dumps({"phases": [{"begin_ns": time.monotonic_ns(), "counters": counters}]}) + "\n")

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
                    say(self.split_line(min(128, prompt_n), len(toks)))
            out, drafted, accepted, rounds = [], 0, 0, 0
            while len(out) < n_predict:
                got = 0
                while got < self.k and rng.random() < 0.7:
                    got += 1
                drafted += self.k
                accepted += got
                out += [rng.randrange(150000) for _ in range(got + 1)]
                rounds += 1
                if self.debug:
                    say(self.split_line(self.k + 1, len(toks) + len(out)))
            out = out[:n_predict]
            if self.defect("STUB_DIVERGE"):
                out[50] += 1
            if self.defect("STUB_DRAFT_DIVERGE"):
                drafted += 1
            self.c["drafts"] += rounds
            self.c["draft_tokens"] += drafted
            self.c["accepted"] += accepted
            self.c["predicted"] += len(out)
            self.staged_counters(rounds)
        return {"tokens": out, "timings": {"predicted_n": len(out), "draft_n": drafted, "draft_n_accepted": accepted,
                                           "prompt_n": prompt_n, "cache_n": cache_n}}

    def stream(self, write, req):
        """Depth-harness shape: two tokens per round, cumulative timings on every chunk, an empty final chunk."""
        depth, slot = len(req["prompt"]), req.get("id_slot", 0)
        with self.lock:
            cached, self.slot_cache[slot] = self.slot_cache.get(slot, []), list(req["prompt"])
        cache_n = max(depth - 132, 0) if req.get("cache_prompt") and cached == list(req["prompt"]) else 0
        diverge = self.defect("STUB_DIVERGE") and os.environ.get("STUB_DIVERGE_SLOT", str(slot)) == str(slot)
        rng = random.Random(depth * 7 + req["seed"] + (1 if diverge else 0))
        if self.debug:
            say("## SPLIT #0: CUDA0 # 1 inputs [attn_inp_kq_mask (  10M)]")
        with self.exclusive(req["n_predict"] > 1):
            return self.stream_rounds(write, req, rng, depth, cache_n, self.chunk_delay(req["n_predict"] > 1))

    def exclusive(self, decoding):
        return self.serial if decoding and os.environ.get("STUB_SERIALIZE") == "1" else contextlib.nullcontext()

    @staticmethod
    def chunk_delay(decoding):
        return float(os.environ.get("STUB_CHUNK_MS", "0")) / 1000 if decoding else 0

    def responses(self, write, req):
        """One agent turn: Responses-API events, the last carrying usage; the server's counters advance by what it generated."""
        spec = req.get("stub", {})
        made = self.generate(int(spec.get("output", 100)), int(spec.get("seed", 1)))
        with self.exclusive(True):
            for _ in range(int(spec.get("chunks", 20))):
                write({"type": "response.output_text.delta", "delta": "x"})
                time.sleep(self.chunk_delay(True))
        if os.environ.get("STUB_ABORT_STREAM") == "1" or spec.get("abort"):
            return
        write({"type": "response.completed", "response": {"usage": {
            "input_tokens": int(spec.get("input", 1000)), "output_tokens": made,
            "input_tokens_details": {"cached_tokens": int(spec.get("cached", 0))}}}})

    def stream_rounds(self, write, req, rng, depth, cache_n, delay):
        n = draft = acc = rounds = 0
        ms = 0.0
        limit = req["n_predict"]
        if req.get("ignore_eos", True):
            say("W set_sampler: seq_id = 0: backend sampling with SPLIT_MODE_TENSOR needs a sampler chain whose first active sampler is top-k with 1 <= k <= 256; using CPU")
        elif os.environ.get("STUB_EOS_AT"):
            limit = min(limit, int(os.environ["STUB_EOS_AT"]))
        while n < limit:
            step = min(2, limit - n)
            toks = [rng.randrange(1000, 150000) for _ in range(step)]
            n += step
            draft += self.k
            acc += step - 1
            rounds += 1
            ms += 40.0 if self.staging else 42.0
            if self.debug and depth > 32768:
                say(f"## SPLIT #1: CUDA0 # 2 inputs [attn_inp_kq_mask ({'   3K' if self.compact(depth, 6) else ' 176K'})] [inp_pos (   0K)]")
            write({"content": "x", "tokens": toks, "stop": False,
                   "timings": {"predicted_ms": ms, "predicted_n": n, "draft_n": draft, "draft_n_accepted": acc}})
            if delay:
                time.sleep(delay)
        write({"content": "", "tokens": [], "stop": True,
               "timings": {"predicted_ms": ms, "predicted_n": n, "draft_n": draft, "draft_n_accepted": acc,
                           "prompt_n": depth - cache_n, "cache_n": cache_n}})
        with self.lock:
            self.staged_counters(rounds)

    def generate(self, n, seed, top1_mean=0.58):
        rng = random.Random(seed)
        made = acc_total = gen_total = 0
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
                else:
                    self.c["plain"] += 1
                self.c["rounds"] += 1
                made += accepted + 1
                secs += 0.0075 + 0.0009 * length
            self.c["predicted"] += made
            self.c["predicted_s"] += secs
            self.c["prompt"] += 800
            self.c["prompt_s"] += 1.1
            self.c["nmax"] = max(self.c["nmax"], 800 + made)
            self.task += 1
            task = self.task
        say(f"slot print_timing: id  0 | task {task} |        eval time = {1000 * secs:10.2f} ms / {made:5d} tokens "
            f"({1000 * secs / made:8.2f} ms per token, {made / secs:8.2f} tokens per second)")
        say(f"slot print_timing: id  0 | task {task} | draft acceptance = {acc_total / max(gen_total, 1):.5f} "
            f"({acc_total} accepted / {gen_total} generated)")
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
            elif self.path == "/metrics":
                self.send(state.metrics(), "text/plain")
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/stub/generate":
                self.send(json.dumps({"tokens": state.generate(int(body.get("n", 100)), int(body.get("seed", 1)), float(body.get("top1_mean", 0.58)))}))
            elif self.path == "/tokenize":
                self.send(json.dumps({"tokens": state.tokenize(body.get("content", ""))}))
            elif self.path == "/v1/responses":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                state.responses(lambda chunk: (self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n"), self.wfile.flush()), body)
            elif self.path == "/completion" and body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def write(chunk):
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                state.stream(write, body)
            elif self.path == "/completion":
                self.send(json.dumps(state.completion(body["prompt"], int(body["n_predict"]), int(body.get("seed", 1)), bool(body.get("cache_prompt")))))
            else:
                self.send_error(404)

    return Handler


def smi(uuids):
    """nvidia-smi stand-in: the lowest free MiB of GPU j is 2900 - 100 j."""
    for i in range(10 ** 7):
        for j, uuid in enumerate(uuids):
            say(f"{uuid}, {2900 - 100 * j + 10 * (i % 5)}")
        time.sleep(0.05)


def main():
    if "--smi" in sys.argv:
        return smi(sys.argv[sys.argv.index("--smi") + 1:])
    state = State(parse(sys.argv[1:]))
    echo_k, echo_p = (state.req_k, state.req_p) if os.environ.get("STUB_LIE") == "1" else (state.k, state.p)
    if os.environ.get("STUB_NO_ECHO") != "1":
        say(f"common_speculative_init_result: stub draft params n_max={echo_k}, n_min=0, p_min={echo_p:.2f}")
    if state.verbose:
        say(f"llama_kv_cache: size = 3400.00 MiB ({state.ctx} cells,  16 layers,  1/1 seqs), K ({state.kv}): 1700.00 MiB, V ({state.kv}): 1700.00 MiB")
        say(f"llama_kv_cache: size =  400.00 MiB ({state.ctx} cells,   1 layers,  1/1 seqs), K ({state.draft_kv}):  200.00 MiB, V ({state.draft_kv}):  200.00 MiB")
    say(f"srv  llama_server: listening on http://127.0.0.1:{state.port}")
    server = ThreadingHTTPServer(("127.0.0.1", state.port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
