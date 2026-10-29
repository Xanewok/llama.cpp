"""CPU-only tests for scripts.instruments.runaway_guard: detector on recorded loops, proxy against a fake engine."""

import gzip
import http.client
import http.server
import itertools
import json
import os
from pathlib import Path
import random
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

from scripts import ROOT
from scripts.instruments import runaway_guard as g

FIX = Path(__file__).parent / "fixtures" / "runaway"
MANIFEST = {m["name"]: m for m in json.loads((FIX / "MANIFEST.json").read_text())}


def text(name):
    return gzip.open(FIX / f"{name}.txt.gz", "rt").read()


def chunks(s, n):
    return [s[i:i + n] for i in range(0, len(s), n)]  # a list: streams are rebuilt per request


def detect(s, step=24, **kw):
    d = g.LoopDetector(**kw)
    for piece in chunks(s, step):
        hit = d.feed(piece)
        if hit:
            return hit
    return None


LOOPS = ["loop_reasoning_letters_p12", "loop_reasoning_letters_p3", "loop_reasoning_letters_p195",
         "loop_args_table_p23", "loop_args_map_p21"]
NEGATIVES = ["legit_reasoning_long", "legit_args_longest", "nearmiss_letters_recovered", "nearmiss_paragraph_repeat",
             "nearmiss_table_recovered"]


class DetectorOnRecordedStreams(unittest.TestCase):
    def test_every_recorded_loop_fires_within_one_window_of_onset(self):
        for name in LOOPS:
            with self.subTest(name):
                hit = detect(text(name))
                self.assertIsNotNone(hit)
                onset = MANIFEST[name]["onset"]
                self.assertEqual(hit["period"] % MANIFEST[name]["period"], 0)
                self.assertLessEqual(hit["period"], g.P_MAX)
                self.assertGreaterEqual(hit["chars"], g.MIN_CHARS)
                # fires at onset + MIN_CHARS (+ the seeding gram), never later than +1 window
                self.assertLessEqual(hit["at"] - onset, g.MIN_CHARS + 2 * g.GRAM + MANIFEST[name]["period"])
                self.assertGreaterEqual(hit["at"] - onset, g.MIN_CHARS)

    def test_token_chunked_capture_fires(self):
        pieces = json.loads(gzip.open(FIX / "loop_tokens_table_catch.json.gz", "rt").read())
        d = g.LoopDetector()
        fired = None
        for i, piece in enumerate(pieces):
            fired = d.feed(piece)
            if fired:
                break
        self.assertIsNotNone(fired)
        self.assertEqual(fired["period"], 16)  # "\n        [0, 0, 0, 0],"
        self.assertLess(i, 6400)  # loop starts at token 92; 12288 chars is ~6.2k tokens

    def test_real_long_non_loop_outputs_never_fire(self):
        for name in NEGATIVES:
            with self.subTest(name):
                self.assertIsNone(detect(text(name)))

    def test_near_misses_are_close_but_below_threshold(self):
        # sensitivity control: a detector 20x more eager fires on the recorded recoveries,
        # so the negatives above really are the hard cases and the default margin is not vacuous
        for name in ("nearmiss_letters_recovered", "nearmiss_table_recovered"):
            with self.subTest(name):
                self.assertIsNotNone(detect(text(name), min_chars=256))
                self.assertIsNone(detect(text(name)))

    def test_deaf_detector_fails_the_positive_test(self):
        # control: the loop tests have teeth; an insensitive detector misses every recorded loop
        for name in LOOPS:
            self.assertIsNone(detect(text(name), min_chars=10 ** 9))

    def test_chunking_does_not_change_the_hit(self):
        s = text("loop_reasoning_letters_p12")
        hits = {step: detect(s, step=step) for step in (1, 7, 128, 4096, 10 ** 6)}
        self.assertEqual(len({(h["at"], h["period"], h["start"]) for h in hits.values()}), 1)

    def test_legit_repetitive_code_and_tables(self):
        row = "        [0; 4],\n"
        self.assertIsNone(detect(row * 255))               # a 256-entry zero table is legitimate
        self.assertIsNone(detect("        [0, 0, 0, 0],\n" * 300))  # 6.6k chars of rows
        self.assertIsNone(detect("        [0, 0, 0, 0],\n" * 400))  # 8.8k: the longest legit source tables reach this
        self.assertIsNotNone(detect("        [0, 0, 0, 0],\n" * 600))  # 13k chars is not
        md = "".join(f"| row{i} | {i * 7 % 13} | {i * i} | text-{i % 5} |\n" for i in range(4000))
        self.assertIsNone(detect(md))
        tests = "".join(f"    assert_eq!(f({i}), {i * 3});\n    assert!(g({i}).is_ok());\n" for i in range(3000))
        self.assertIsNone(detect(tests))
        self.assertIsNone(detect("-" * 200 + "\n" + "=" * 200))
        self.assertIsNone(detect("ab" * 2000))              # 4k chars < MIN_CHARS
        self.assertIsNone(detect("0,0,0,0\n" * 1024))       # 8192 chars of CSV zeros (Codex review)
        self.assertIsNone(detect("A" * 8192))               # base64 of 6144 zero bytes
        self.assertEqual(g.MIN_CHARS, 12288)

    def test_period_bounds(self):
        unit = "".join(chr(97 + (i * 7) % 26) + str(i % 10) for i in range(900))  # 1800-char unit, aperiodic inside
        self.assertIsNotNone(detect(unit * 8))  # 14.4k chars >= MIN_CHARS
        long_unit = unit + "".join(chr(65 + (i * 11) % 26) + str(i % 7) for i in range(300))  # 2400 > P_MAX
        self.assertIsNone(detect(long_unit * 8))

    def test_state_is_bounded_and_fast(self):
        d = g.LoopDetector(min_chars=10 ** 12)
        import random
        rng = random.Random(7)
        words = [f"w{rng.randrange(5000)}" for _ in range(200000)]
        t0 = time.monotonic()
        for i in range(0, len(words), 16):
            d.feed(" ".join(words[i:i + 16]) + " ")
        elapsed = time.monotonic() - t0
        self.assertLessEqual(len(d.buf), 4 * g.P_MAX + 4 * g.GRAM)
        self.assertLessEqual(len(d.cands), d.max_candidates)
        self.assertLessEqual(len(d.last), 4 * g.P_MAX)
        self.assertGreater(d.n, 1_000_000)
        self.assertLess(elapsed, 30)  # ~0.5 us/char measured; wide margin for loaded CI

    @staticmethod
    def indented_unit(seed, statements=4, lines=40):
        rng = random.Random(seed)
        pool = ["x += 1;", "}", "if ok {", "return Ok(());", "self.len -= 1;", "continue;"][:statements]
        return "".join(" " * rng.randrange(20, 25) + rng.choice(pool) + "\n" for _ in range(lines))

    def test_a_unit_made_of_long_uniform_runs_is_found(self):
        # every 16-gram of this unit recurs inside it, so the true period is never a gram's latest distance
        unit = "".join(c * 100 for c in "aabaabbabb")
        hits = {step: detect(unit * 20, step=step) for step in (1, 24, 4096, 20000)}
        for hit in hits.values():
            self.assertIsNotNone(hit)
            self.assertEqual(hit["period"] % 1000, 0)
        self.assertEqual(len({(h["at"], h["period"], h["start"]) for h in hits.values()}), 1)

    def test_deeply_indented_loops_are_found_whatever_the_read_size(self):
        units = [self.indented_unit(seed) for seed in range(6)]
        units.append("".join(" " * (20 + i % 5) + "x += 1;\n" for i in range(40)))  # minimal period 150
        for n, unit in enumerate(units):
            stream = unit * (4 * g.MIN_CHARS // len(unit) + 2)
            with self.subTest(n):
                hits = {step: detect(stream, step=step) for step in (1, 24, 5000)}
                for hit in hits.values():
                    self.assertIsNotNone(hit)
                    self.assertEqual(hit["period"] % (150 if n == 6 else len(unit)), 0)
                    self.assertGreaterEqual(hit["chars"], g.MIN_CHARS)
                self.assertEqual(len({(h["at"], h["period"], h["start"]) for h in hits.values()}), 1)

    def test_uniform_runs_that_are_not_loops_stay_clean(self):
        rng = random.Random(5)
        varied = "".join(f"line {i}: {rng.randrange(10 ** 6)} {'-' * 64}\n" if i % 4 == 0 else f"text {rng.randrange(10 ** 9)}\n"
                         for i in range(4000))
        boards = "".join("\n".join("".join(rng.choice(".#") for _ in range(64)) for _ in range(64)) + "\n\n"
                         for _ in range(12))
        dots = "".join(("." * rng.randrange(16, 64) + "#" * rng.randrange(1, 8) + "\n") for _ in range(2500))
        indented = "".join(" " * rng.randrange(20, 25) + rng.choice(["x += 1;", "}", "continue;"]) + f" // {rng.randrange(10 ** 6)}\n"
                           for _ in range(1200))
        for name, text in dict(varied=varied, boards=boards, dots=dots, indented=indented, spaces=" " * 12000,
                               separators=("-" * 100 + "\n") * 100, zeros="0" * 12000).items():
            with self.subTest(name):
                self.assertGreater(len(text), 10000)  # each below MIN_CHARS or aperiodic
                self.assertIsNone(detect(text, step=7))
                self.assertIsNone(detect(text, step=5000))

    def test_uniform_and_adversarially_repetitive_input_stays_bounded_and_cheap(self):
        rng = random.Random(1)
        noisy = "".join(" " * rng.randrange(20, 25) + rng.choice(["x += 1;", "}"]) + "\n" for _ in range(20000))
        for name, text, limit in (("all a", "a" * 300_000, 5), ("all space", " " * 300_000, 5), ("abab", "ab" * 150_000, 5),
                                  ("random indent, two statements", noisy, 30)):
            with self.subTest(name):
                d = g.LoopDetector(min_chars=10 ** 12)
                t0 = time.monotonic()
                for i in range(0, len(text), 64):
                    d.feed(text[i:i + 64])
                self.assertLess(time.monotonic() - t0, limit)  # measured 0.1-1.4 s; wide margin for a loaded box
                self.assertLessEqual(len(d.buf), 4 * g.P_MAX + 4 * g.GRAM)
                self.assertLessEqual(len(d.cands), d.max_candidates)
                self.assertLessEqual(len(d.last), 4 * g.P_MAX)
                self.assertLessEqual(sum(map(len, d.older.values())), 4 * g.P_MAX)
                if name != "random indent, two statements":
                    self.assertLessEqual(len(d.cands), 2)  # a uniform gram proposes only its latest occurrence

    def test_unicode_and_json_escaped_loops(self):
        self.assertIsNotNone(detect("żółć, " * 2500))
        self.assertIsNotNone(detect(json.dumps("        [0, 0, 0, 0],\n" * 600)[1:-1]))


# ------------------------------------------------------------------ fake engine

def responses_stream(items, tail=True):
    """items: list of (kind, iterable of delta strings), kind in reasoning|text|args. Yields SSE event bytes.

    Event shapes follow llama-server's Responses stream (tools/server/server-task.cpp to_json_oaicompat_resp).
    """
    def ev(name, data):
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
    yield ev("response.created", {"type": "response.created", "response": {"id": "r1", "object": "response", "status": "in_progress"}})
    yield ev("response.in_progress", {"type": "response.in_progress", "response": {"id": "r1", "object": "response", "status": "in_progress"}})
    done_items = []
    for n, (kind, deltas) in enumerate(items):
        iid = f"it{n}"
        acc = []
        if kind == "reasoning":
            etype = "response.reasoning_text.delta"
            yield ev("response.output_item.added", {"type": "response.output_item.added", "item": {
                "id": iid, "summary": [], "type": "reasoning", "content": [], "encrypted_content": "", "status": "in_progress"}})
        elif kind == "text":
            etype = "response.output_text.delta"
            yield ev("response.output_item.added", {"type": "response.output_item.added", "item": {
                "content": [], "id": iid, "role": "assistant", "status": "in_progress", "type": "message"}})
            yield ev("response.content_part.added", {"type": "response.content_part.added", "item_id": iid,
                                                     "part": {"type": "output_text", "text": ""}})
        else:
            etype = "response.function_call_arguments.delta"
            yield ev("response.output_item.added", {"type": "response.output_item.added", "item": {
                "id": iid, "arguments": "", "call_id": "call_" + iid, "name": "exec_command", "type": "function_call",
                "status": "in_progress"}})
        for d in deltas:
            acc.append(d)
            yield ev(etype, {"type": etype, "item_id": iid, "delta": d})
        full = "".join(acc)
        if kind == "reasoning":
            item = {"id": iid, "summary": [], "type": "reasoning", "content": [{"type": "reasoning_text", "text": full}],
                    "encrypted_content": "", "status": "completed"}
        elif kind == "text":
            yield ev("response.content_part.done", {"type": "response.content_part.done", "item_id": iid,
                                                    "part": {"type": "output_text", "text": full}})
            item = {"content": [{"type": "output_text", "text": full, "annotations": []}], "id": iid, "role": "assistant",
                    "status": "completed", "type": "message"}
        else:
            item = {"id": iid, "arguments": full, "call_id": "call_" + iid, "name": "exec_command", "type": "function_call",
                    "status": "completed"}
        done_items.append(item)
        yield ev("response.output_item.done", {"type": "response.output_item.done", "item": item})
    if tail:
        usage = {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12,
                 "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
        yield ev("response.completed", {"type": "response.completed", "response": {
            "id": "r1", "object": "response", "status": "completed", "output": done_items, "usage": usage}})


def chat_stream(parts):
    """parts: list of (field, iterable); field content|reasoning_content|tool|function_call."""
    def ev(delta, finish=None):
        return b"data: " + json.dumps({"object": "chat.completion.chunk", "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish}]}).encode() + b"\n\n"
    yield ev({"role": "assistant", "content": None})
    for field, deltas in parts:
        for d in deltas:
            if field == "tool":
                yield ev({"tool_calls": [{"index": 0, "function": {"arguments": d}}]})
            elif field == "function_call":
                yield ev({"function_call": {"arguments": d}})
            else:
                yield ev({field: d})
    yield ev({}, "stop")
    yield b"data: [DONE]\n\n"


def looping(prefix, unit):
    yield from chunks(prefix, 24)
    for i in itertools.count():
        yield unit[(i * 24) % len(unit):][:24] if len(unit) > 24 else unit


class FakeEngine(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """plan(index, path, body) -> iterable of SSE event bytes; the iterable may be endless."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, plan, handler=None):
        self.plan, self.requests, self.aborted, self.finished = plan, [], [], []
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), handler or FakeHandler)
        self.port = self.server_address[1]
        threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def stop(self):
        self.shutdown()
        self.server_close()


class FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"nonjson": True}
        if not isinstance(body, dict):
            body = {"nonobject": True}
        with self.server.lock:
            index = len(self.server.requests)
            self.server.requests.append(dict(path=self.path, body=body, raw=raw,
                                             headers={k.lower(): v for k, v in self.headers.items()}))
        if not body.get("stream"):
            out = json.dumps({"echo": body.get("model")}).encode()
            status = 400 if body.get("model") == "bad" else 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        events = self.server.plan(index, self.path, body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for e in events:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(e), e))
                self.wfile.flush()
                gate = getattr(self.server, "gate", None)
                if gate is not None:
                    gate(index)
            self.wfile.write(b"0\r\n\r\n")
            with self.server.lock:
                self.server.finished.append(index)
        except (BrokenPipeError, ConnectionResetError):
            with self.server.lock:
                self.server.aborted.append(index)
            self.close_connection = True


def post(port, path, body, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def events_of(data):
    out = []
    for raw in g.split_events(data)[0]:
        out.append(g.parse_event(raw))
    return out


def joined(data):
    """All delta text of an SSE body, concatenated."""
    return "".join(t for name, d in events_of(data) for _, t in g.sse_texts(name, d))


def wait_for(cond, timeout=10):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


class ProxyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.tmp.name, "guard.jsonl")
        self.cleanup = []
        self.escaped = []  # exceptions that reached socketserver's traceback printer

    def tearDown(self):
        for fn in reversed(self.cleanup):
            fn()
        self.tmp.cleanup()
        self.assertEqual(self.escaped, [], "a handler let an exception escape")

    def engine(self, plan):
        e = FakeEngine(plan)
        self.cleanup.append(e.stop)
        return e

    def proxy(self, engine, **kw):
        kw.setdefault("log", self.log)
        return self.started(g.GuardProxy(f"http://127.0.0.1:{engine.port}", **kw))

    def started(self, p):
        p.handle_error = lambda request, client_address: self.escaped.append(sys.exc_info()[1])
        p.start()
        self.cleanup.append(p.stop)
        return p

    def log_rows(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as stream:
            return [json.loads(l) for l in stream]


class ProxyResponses(ProxyCase):
    BODY = {"model": "m", "input": "hi", "stream": True, "temperature": 1.0}

    def clean_plan(self):
        return lambda i, path, body: responses_stream([("reasoning", chunks("Let me think. " * 50, 20)),
                                                       ("args", chunks('{"cmd": "ls"}', 5))])

    def test_clean_stream_is_relayed_byte_identically(self):
        expected = b"".join(responses_stream([("reasoning", chunks("Let me think. " * 50, 20)),
                                              ("args", chunks('{"cmd": "ls"}', 5))]))
        for mode in ("buffer", "stream"):
            with self.subTest(mode):
                e = self.engine(self.clean_plan())
                p = self.proxy(e, mode=mode)
                status, data = post(p.port, "/v1/responses", self.BODY)
                self.assertEqual(status, 200)
                self.assertEqual(data.replace(b": runaway-guard keepalive\n\n", b""), expected)
                self.assertEqual(e.requests[0]["raw"], json.dumps(self.BODY).encode())  # first attempt untouched

    def test_reasoning_loop_is_aborted_and_resampled_invisibly(self):
        loop = text("loop_reasoning_letters_p3")
        good = [("reasoning", chunks("Sensible reasoning. " * 30, 20)), ("text", chunks("Done.", 3))]

        def plan(i, path, body):
            if i == 0:
                return responses_stream([("reasoning", looping(loop, loop[-3:]))], tail=False)
            return responses_stream(good)

        e = self.engine(plan)
        p = self.proxy(e)
        t0 = time.monotonic()
        status, data = post(p.port, "/v1/responses", self.BODY)
        self.assertEqual(status, 200)
        self.assertEqual(data.replace(b": runaway-guard keepalive\n\n", b""), b"".join(responses_stream(good)))
        self.assertNotIn(b"S, S, S", data)  # the loop never reached the client
        self.assertEqual(len(e.requests), 2)
        self.assertTrue(wait_for(lambda: e.aborted == [0]), "upstream was not dropped after the loop")
        self.assertEqual(e.requests[1]["raw"], e.requests[0]["raw"])  # D3: the retry is the client's own bytes
        self.assertTrue(wait_for(lambda: "end" in [r["ev"] for r in self.log_rows()]))
        rows = self.log_rows()
        kinds = [r["ev"] for r in rows]
        self.assertEqual(kinds, ["request", "loop", "retry", "attempt_done", "end"])
        loop_row = rows[1]
        self.assertEqual(loop_row["period"], 3)
        self.assertGreaterEqual(loop_row["loop_chars"], g.MIN_CHARS)
        self.assertEqual(loop_row["channel"][0], "response.reasoning_text.delta")
        self.assertLess(time.monotonic() - t0, 30)

    def test_function_call_argument_loop(self):
        loop = text("loop_args_table_p23")
        e = self.engine(lambda i, path, body: responses_stream([("args", looping(loop, loop[-23:]))], tail=False)
                        if i == 0 else responses_stream([("args", chunks('{"cmd": "cargo test"}', 4))]))
        p = self.proxy(e)
        status, data = post(p.port, "/v1/responses", self.BODY)
        self.assertEqual(status, 200)
        self.assertIn("cargo test", joined(data))
        self.assertNotIn(b"[0, 0, 0, 0]", data)
        loop_row = [r for r in self.log_rows() if r["ev"] == "loop"][0]
        self.assertEqual(loop_row["channel"][0], "response.function_call_arguments.delta")

    def test_loop_in_output_text(self):
        unit = "the same sentence again. "
        e = self.engine(lambda i, path, body: responses_stream([("text", looping("", unit))], tail=False)
                        if i == 0 else responses_stream([("text", chunks("fine", 2))]))
        p = self.proxy(e)
        _, data = post(p.port, "/v1/responses", self.BODY)
        self.assertIn("fine", joined(data))
        self.assertEqual(len(e.requests), 2)

    def test_two_loops_in_a_row_end_in_a_clean_error(self):
        loop = text("loop_reasoning_letters_p12")
        e = self.engine(lambda i, path, body: responses_stream([("reasoning", looping(loop, loop[-12:]))], tail=False))
        p = self.proxy(e)
        status, data = post(p.port, "/v1/responses", self.BODY)
        self.assertEqual(status, 200)
        evs = events_of(data)
        self.assertEqual([n for n, _ in evs], ["response.failed"])
        err = evs[0][1]["response"]["error"]
        self.assertEqual(err["code"], "runaway_detected")
        self.assertEqual(len(e.requests), 2)
        self.assertTrue(wait_for(lambda: sorted(e.aborted) == [0, 1]))
        self.assertEqual(p.stats["failed"], 1)

    def test_client_seed_is_never_overridden_and_retry_is_refused(self):
        loop = text("loop_reasoning_letters_p3")
        e = self.engine(lambda i, path, body: responses_stream([("reasoning", looping(loop, loop[-3:]))], tail=False))
        p = self.proxy(e)
        _, data = post(p.port, "/v1/responses", dict(self.BODY, seed=1234))
        self.assertEqual(events_of(data)[0][0], "response.failed")
        self.assertEqual(len(e.requests), 1)
        self.assertIn("retry_refused", [r["ev"] for r in self.log_rows()])

    def test_client_seed_bump_policy(self):
        loop = text("loop_reasoning_letters_p3")
        e = self.engine(lambda i, path, body: responses_stream([("reasoning", looping(loop, loop[-3:]))], tail=False)
                        if i == 0 else responses_stream([("text", chunks("ok", 1))]))
        p = self.proxy(e, client_seed="bump")
        _, data = post(p.port, "/v1/responses", dict(self.BODY, seed=1234))
        self.assertIn("ok", joined(data))
        self.assertEqual(e.requests[0]["body"]["seed"], 1234)
        self.assertNotEqual(e.requests[1]["body"]["seed"], 1234)

    def test_stream_mode_forwards_live_then_fails_cleanly(self):
        loop = text("loop_reasoning_letters_p3")
        e = self.engine(lambda i, path, body: responses_stream([("reasoning", looping(loop, loop[-3:]))], tail=False))
        p = self.proxy(e, mode="stream")
        status, data = post(p.port, "/v1/responses", self.BODY)
        evs = events_of(data)
        self.assertEqual(evs[-1][0], "response.failed")
        self.assertGreater(len(evs), 100)  # the partial output was forwarded as it came
        self.assertEqual(len(e.requests), 1)  # no transparent retry: the client already saw output
        self.assertTrue(wait_for(lambda: e.aborted == [0]))

    def test_buffer_mode_holds_output_stream_mode_does_not(self):
        released = threading.Event()
        seen = []

        def gate(i):
            seen.append(1)
            if len(seen) == 8:  # the engine stalls after the first deltas
                released.wait(15)

        e = self.engine(lambda i, path, body: responses_stream([("reasoning", chunks("x" * 400, 20))]))
        e.gate = gate
        for mode, early in (("stream", True), ("buffer", False)):
            with self.subTest(mode):
                released.clear()
                seen.clear()
                p = self.proxy(e, mode=mode, keepalive=0.2)
                body = json.dumps(self.BODY).encode()
                with socket.create_connection(("127.0.0.1", p.port), timeout=15) as s:
                    s.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                              b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
                    time.sleep(0.8)
                    s.setblocking(False)
                    try:
                        before = s.recv(1 << 20)
                    except BlockingIOError:
                        before = b""
                    s.setblocking(True)
                    released.set()
                    full = before
                    while not full.endswith(b"0\r\n\r\n"):
                        chunk = s.recv(1 << 20)
                        if not chunk:
                            break
                        full += chunk
                self.assertEqual(b"reasoning_text.delta" in before, early, before[:300])
                self.assertIn(b"response.completed", full)

    def test_pass_through_of_non_stream_get_and_upstream_errors(self):
        e = self.engine(lambda *a: iter(()))
        p = self.proxy(e)
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("GET", "/health")
        r = c.getresponse()
        self.assertEqual((r.status, r.read()), (200, b'{"ok":true}'))
        status, data = post(p.port, "/v1/responses", {"model": "bad", "input": "x"})
        self.assertEqual((status, json.loads(data)), (400, {"echo": "bad"}))
        status, data = post(p.port, "/v1/responses", {"model": "m", "input": "x", "stream": False})
        self.assertEqual(status, 200)
        self.assertEqual(p.stats["guarded"], 0)

    def test_upstream_down_gives_gateway_error(self):
        p = self.started(g.GuardProxy("http://127.0.0.1:1", log=self.log))
        status, _ = post(p.port, "/v1/responses", self.BODY)
        self.assertEqual(status, 502)


class ProxyChat(ProxyCase):
    BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    def test_clean_chat_stream_identical(self):
        parts = [("reasoning_content", chunks("hm " * 30, 5)), ("content", chunks("hello", 2))]
        e = self.engine(lambda i, path, body: chat_stream(parts))
        p = self.proxy(e)
        status, data = post(p.port, "/v1/chat/completions", self.BODY)
        self.assertEqual(data.replace(b": runaway-guard keepalive\n\n", b""), b"".join(chat_stream(parts)))

    def test_each_chat_channel_is_guarded(self):
        loop = text("loop_args_table_p23")
        for field, unit in (("content", "the same sentence again. "), ("reasoning_content", "hmm, S, "),
                            ("tool", loop[-23:])):
            with self.subTest(field):
                e = self.engine(lambda i, path, body, field=field, unit=unit: chat_stream([(field, looping("", unit))])
                                if i == 0 else chat_stream([("content", chunks("recovered", 3))]))
                p = self.proxy(e)
                status, data = post(p.port, "/v1/chat/completions", self.BODY)
                self.assertIn("recovered", joined(data))
                self.assertEqual(len(e.requests), 2)
                self.assertTrue(wait_for(lambda: e.aborted == [0]))

    def test_chat_failure_is_an_error_object(self):
        e = self.engine(lambda i, path, body: chat_stream([("content", looping("", "again and again. "))]))
        p = self.proxy(e, retries=0)
        status, data = post(p.port, "/v1/chat/completions", self.BODY)
        evs = events_of(data)
        self.assertEqual(evs[-1][1]["error"]["code"], "runaway_detected")

    @staticmethod
    def interleaved(n=2000):
        """Two tool calls, 16 chars per event each: 16k chars per channel, 32k pooled."""
        def ev(delta):
            return b"data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}).encode() + b"\n\n"
        for i in range(n):
            yield ev({"tool_calls": [{"index": i % 2, "function": {"arguments": ("ab" if i % 2 else "cd") * 8}}]})
        yield b"data: [DONE]\n\n"

    def test_interleaved_tool_calls_do_not_share_a_channel(self):
        # Each channel is periodic for 16k chars, below min_chars=20000; pooled the periodic run is 32k chars, above it.
        # So a detector shared by both channels would fire here and the per-channel one must not.
        e = self.engine(lambda i, path, body: self.interleaved())
        p = self.proxy(e, detector=dict(min_chars=20000))
        _, data = post(p.port, "/v1/chat/completions", self.BODY)
        self.assertIn(b"[DONE]", data)
        self.assertEqual(p.stats["loops"], 0)
        self.assertEqual(len(e.requests), 1)

    def test_pooled_detector_mutant_is_caught_by_the_isolation_test(self):
        real = g.sse_texts
        with unittest.mock.patch.object(g, "sse_texts",
                                        lambda name, data: [(("pooled",), t) for _, t in real(name, data)]):
            e = self.engine(lambda i, path, body: self.interleaved())
            p = self.proxy(e, detector=dict(min_chars=20000), retries=0)
            post(p.port, "/v1/chat/completions", self.BODY)
        self.assertEqual(p.stats["loops"], 1)

    def test_a_loop_in_one_channel_is_found_among_another_channels_text(self):
        unit = "the same sentence again. "
        noise = [f"unrelated thought number {i}, " for i in range(3000)]
        d = g.Detectors()
        hit = None
        for i in range(2500):
            chunk = {"choices": [{"index": 0, "delta": {"content": unit}}]}
            other = {"choices": [{"index": 0, "delta": {"reasoning_content": noise[i]}}]}
            hit = d.feed_event(None, other) or d.feed_event(None, chunk)
            if hit:
                break
        self.assertIsNotNone(hit)
        self.assertEqual(hit["channel"], [0, "content"])


KEEPALIVE_COMMENT = b": runaway-guard keepalive\n\n"


def post_raw(port, path, raw, headers=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, raw, dict({"Content-Type": "application/json"}, **(headers or {})))
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def open_request(port, path, body, timeout=15):
    """A connected socket with the POST already sent; the caller reads or closes it."""
    raw = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.sendall(b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
              % (path.encode(), len(raw), raw))
    return s


def endless_clean(delay=0.005):
    """Plan: one never-ending, non-periodic reasoning stream."""
    def plan(i, path, body):
        def deltas():
            for n in itertools.count():
                time.sleep(delay)
                yield f"step {n}: thinking about item {n * 7 % 13}. "
        return responses_stream([("reasoning", deltas())], tail=False)
    return plan


def completions_stream(deltas):
    for d in deltas:
        yield b"data: " + json.dumps({"object": "text_completion", "choices": [{"index": 0, "text": d}]}).encode() + b"\n\n"
    yield b"data: [DONE]\n\n"


def sse(data, eol=b"\n", blank=b"\n"):
    return b"data: " + json.dumps(data).encode() + eol + blank


class CloseHandler(FakeHandler):
    """Engine that answers `Connection: close`: http.client then moves the socket from the connection to the response."""

    def send_header(self, k, v):
        super().send_header(k, v)
        if k == "Transfer-Encoding":
            super().send_header("Connection", "close")


class SseFraming(unittest.TestCase):
    # (line end, blank-line end) pairs that are valid SSE; ("\r", "\n") is excluded because it reads as one CRLF
    STYLES = [(b"\n", b"\n"), (b"\r\n", b"\r\n"), (b"\r", b"\r"), (b"\n", b"\r\n"), (b"\r\n", b"\n"),
              (b"\r", b"\r\n"), (b"\n", b"\r")]

    def stream(self, eol, blank, n=6):
        events = [{"choices": [{"index": 0, "delta": {"content": f"piece {i}"}}]} for i in range(n)]
        raw = b"".join(b"event: message" + eol + b"data: " + json.dumps(e).encode() + eol + blank for e in events)
        return raw, events

    @staticmethod
    def frame(pieces):
        f = g.SseFramer()
        out = []
        for piece in pieces:
            out += f.feed(piece)
        return out, f.finish()

    def test_every_terminator_style_yields_every_event_under_any_read_boundaries(self):
        rng = __import__("random").Random(11)
        for eol, blank in self.STYLES:
            raw, events = self.stream(eol, blank)
            cuts = [[raw[i:i + n] for i in range(0, len(raw), n)] for n in (1, 2, 3, 5, 64, len(raw))]
            for _ in range(20):
                points = sorted(rng.sample(range(1, len(raw)), 12))
                cuts.append([raw[a:b] for a, b in zip([0] + points, points + [len(raw)])])
            for pieces in cuts:
                with self.subTest(eol=eol, blank=blank, first=len(pieces[0])):
                    framed, tail = self.frame(pieces)
                    self.assertEqual(b"".join(framed) + tail, raw)  # bytes preserved exactly
                    parsed = [parse for parse in (g.parse_event(e)[1] for e in framed) if parse is not None]
                    self.assertEqual(parsed, events)

    def test_one_shot_split_events_handles_cr_terminators(self):
        # lone CR and mixed terminators are valid SSE; the old splitter saw no event boundary at all
        for eol, blank in self.STYLES:
            with self.subTest(eol=eol, blank=blank):
                raw, events = self.stream(eol, blank)
                framed, rest = g.split_events(raw)
                self.assertEqual(rest, b"")
                self.assertEqual([p for p in (g.parse_event(e)[1] for e in framed) if p is not None], events)

    def test_unicode_line_separators_inside_json_do_not_split_the_event(self):
        for ch in (" ", " ", "\u0085", "\x0b", "\x0c", "\x1c"):
            with self.subTest(ch=repr(ch)):
                data = {"choices": [{"index": 0, "delta": {"content": f"a{ch}b"}}]}
                raw = b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n"
                name, parsed = g.parse_event(raw)
                self.assertEqual(parsed, data)
                self.assertEqual(g.sse_texts(name, parsed), [((0, "content"), f"a{ch}b")])

    def test_multiline_data_comments_and_missing_space(self):
        raw = b": hi\r\nevent: message\r\ndata: {\"choices\":\r\ndata: [{\"delta\":{\"content\":\"a\"}}]}\r\n\r\n"
        framed, rest = g.split_events(raw)
        self.assertEqual(len(framed), 2)  # the leading comment is its own piece
        self.assertEqual(g.parse_event(framed[1]), ("message", {"choices": [{"delta": {"content": "a"}}]}))
        self.assertEqual(g.parse_event(b'data:{"a":1}\n\n'), (None, {"a": 1}))
        self.assertEqual(g.parse_event(b"data: [DONE]\n\n"), (None, None))

    def test_cr_and_lf_of_one_line_end_may_arrive_in_separate_reads(self):
        framed, tail = self.frame([b"data: a\r", b"\ndata: b\r\n\r", b"\n"])
        self.assertEqual(b"".join(framed) + tail, b"data: a\r\ndata: b\r\n\r\n")
        self.assertEqual(len([e for e in framed if e.startswith(b"data")]), 1)  # one event with two lines

    def test_comment_pings_do_not_accumulate(self):
        f = g.SseFramer()
        pieces = 0
        for _ in range(50000):
            pieces += len(f.feed(b": ping\n"))
            self.assertEqual(f.pending, 0)
        self.assertEqual(pieces, 50000)
        self.assertEqual(f.feed(b": ping\ndata: x\n\n"), [b": ping\n", b"data: x\n\n"])

    def test_a_huge_event_in_small_reads_is_scanned_linearly(self):
        f = g.SseFramer(max_event=10 ** 9)
        payload = b"data: " + b"x" * (4 << 20) + b"\n\n"
        t0 = time.monotonic()
        events = []
        for i in range(0, len(payload), 1024):
            events += f.feed(payload[i:i + 1024])
        self.assertEqual(events, [payload])
        self.assertLess(time.monotonic() - t0, 1.0)  # the old pending + chunk rescan took seconds

    def test_unterminated_event_flags_overflow_and_keeps_the_bytes(self):
        f = g.SseFramer(max_event=100)
        self.assertEqual(f.feed(b"data: " + b"x" * 500), [])
        self.assertTrue(f.overflowed)
        self.assertEqual(f.finish(), b"data: " + b"x" * 500)


class ParserRobustness(unittest.TestCase):
    def test_legacy_function_call_arguments_are_a_channel(self):
        data = {"choices": [{"index": 0, "delta": {"function_call": {"name": "f", "arguments": "abc"}}}]}
        self.assertEqual(g.sse_texts(None, data), [((0, "function_call"), "abc")])
        d = g.Detectors()
        hit = None
        for _ in range(2000):
            hit = d.feed_event(None, {"choices": [{"index": 0, "delta": {"function_call": {"arguments": "0, 0, 0, "}}}]})
            if hit:
                break
        self.assertIsNotNone(hit)
        self.assertEqual(hit["channel"], [0, "function_call"])

    def test_malformed_upstream_json_yields_nothing_and_never_raises(self):
        weird = [
            {"choices": [{"delta": "oops"}]}, {"choices": [{"delta": None, "index": [1]}]}, {"choices": 5},
            {"choices": {"a": 1}}, {"choices": [None, 3, "x"]},
            {"choices": [{"delta": {"content": 5, "tool_calls": [None, 3, {"function": 7}, {"function": {"arguments": 1}}]}}]},
            {"choices": [{"delta": {"tool_calls": {"index": 0}, "function_call": 4}, "text": 9}]},
            {"type": "response.output_text.delta", "item_id": ["unhashable"], "delta": 7},
            {"type": ["x"], "choices": []}, [], "str", 5, None,
        ]
        for data in weird:
            with self.subTest(data=data):
                d = g.Detectors()
                d.feed_event(None, data)
                g.sse_texts("response.x.delta", data)
        # unhashable ids still make a usable channel
        out = g.sse_texts(None, {"type": "response.output_text.delta", "item_id": ["a"], "delta": "t"})
        self.assertEqual([t for _, t in out], ["t"])
        hash(out[0][0])


class ProxyDefects(ProxyCase):
    RESP = {"model": "m", "input": "hi", "stream": True}
    CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    def kinds(self):
        return [r["ev"] for r in self.log_rows()]

    # D1: a client that went away must not leave a generation running
    def test_buffer_mode_aborts_upstream_when_the_client_disconnects(self):
        e = self.engine(endless_clean())
        p = self.proxy(e, mode="buffer")
        s = open_request(p.port, "/v1/responses", self.RESP)
        time.sleep(1.0)
        s.close()
        self.assertTrue(wait_for(lambda: e.aborted == [0], 6), "orphan generation: upstream never dropped")
        self.assertTrue(wait_for(lambda: "client_disconnected" in self.kinds()))
        self.assertEqual(len(e.requests), 1)  # no retry for a dead client
        self.assertNotIn("loop", self.kinds())

    def test_a_stalled_upstream_does_not_pin_the_handler_after_the_client_leaves(self):
        for mode in ("stream", "buffer"):
            with self.subTest(mode):
                self.log = os.path.join(self.tmp.name, f"{mode}.jsonl")
                released = threading.Event()
                self.cleanup.append(released.set)
                seen = []

                def gate(i, seen=seen, released=released):
                    seen.append(1)
                    if len(seen) == 3:
                        released.wait(30)

                e = self.engine(lambda i, path, body: responses_stream([("reasoning", chunks("x" * 400, 20))]))
                e.gate = gate
                p = self.proxy(e, mode=mode)
                s = open_request(p.port, "/v1/responses", self.RESP)
                self.assertTrue(wait_for(lambda: len(seen) >= 3))
                s.close()
                self.assertTrue(wait_for(lambda: "client_disconnected" in self.kinds(), 6),
                                "handler still waiting on the stalled upstream")
                released.set()
                self.assertTrue(wait_for(lambda: e.aborted == [0], 6))

    def test_buffer_mode_keepalive_flows_while_the_upstream_is_busy(self):
        def deltas():
            for n in range(300):
                time.sleep(0.01)  # the upstream is never silent for 0.1 s, let alone 0.3 s
                yield f"t{n} "

        e = self.engine(lambda i, path, body: responses_stream([("reasoning", deltas())]))
        p = self.proxy(e, mode="buffer", keepalive=0.3)
        with open_request(p.port, "/v1/responses", self.RESP) as s:
            got = b""
            while KEEPALIVE_COMMENT not in got and not got.endswith(b"0\r\n\r\n"):
                chunk = s.recv(1 << 16)
                if not chunk:
                    break
                got += chunk
        self.assertIn(KEEPALIVE_COMMENT, got)
        self.assertNotIn(b"response.completed", got)  # still held when the comment arrived

    # D2
    def test_abort_reaches_an_upstream_that_answers_connection_close(self):
        loop = text("loop_reasoning_letters_p3")

        def plan(i, path, body):
            if i == 0:
                def slow():
                    for d in looping(loop, loop[-3:]):
                        time.sleep(0.0005)
                        yield d
                return responses_stream([("reasoning", slow())], tail=False)
            return responses_stream([("text", chunks("fine", 2))])

        e = FakeEngine(plan, handler=CloseHandler)
        self.cleanup.append(e.stop)
        p = self.proxy(e)
        status, data = post(p.port, "/v1/responses", self.RESP)
        self.assertIn("fine", joined(data))
        self.assertTrue(wait_for(lambda: 0 in e.aborted, 8), "the looping generation was never dropped")

    # D3
    RAW = (b'{ "model":"m",\n "input" : "h\\u00e9llo \\u2028", "stream" : true,\n'
           b' "temperature":1.0 , "x": [1,  2] }')

    def looping_then_fine(self):
        loop = text("loop_reasoning_letters_p3")
        return lambda i, path, body: (responses_stream([("reasoning", looping(loop, loop[-3:]))], tail=False)
                                      if i == 0 else responses_stream([("text", chunks("ok", 1))]))

    def test_retry_resends_the_original_bytes(self):
        e = self.engine(self.looping_then_fine())
        p = self.proxy(e)
        status, data = post_raw(p.port, "/v1/responses", self.RAW)
        self.assertIn("ok", joined(data))
        self.assertEqual(e.requests[0]["raw"], self.RAW)
        self.assertEqual(e.requests[1]["raw"], self.RAW)
        self.assertIsNone([r for r in self.log_rows() if r["ev"] == "retry"][0]["seed"])

    def test_seed_injection_is_opt_in(self):
        e = self.engine(self.looping_then_fine())
        p = self.proxy(e, inject_seed=True)
        post_raw(p.port, "/v1/responses", self.RAW)
        first, second = e.requests[0], e.requests[1]
        self.assertEqual(first["raw"], self.RAW)
        self.assertIsInstance(second["body"]["seed"], int)
        self.assertEqual({k: v for k, v in second["body"].items() if k != "seed"}, first["body"])
        self.assertEqual([r for r in self.log_rows() if r["ev"] == "retry"][0]["seed"], second["body"]["seed"])

    # D5
    def test_cr_only_framing_is_still_guarded(self):
        loop = text("loop_reasoning_letters_p3")

        def plan(i, path, body):
            if i:
                return iter([b"data: [DONE]\r\r"])
            return itertools.islice((sse({"choices": [{"index": 0, "delta": {"content": d}}]}, b"\r", b"\r")
                                     for d in looping(loop, loop[-3:])), 5000)

        e = self.engine(plan)
        p = self.proxy(e)
        _, data = post(p.port, "/v1/chat/completions", self.CHAT)
        self.assertEqual(len(e.requests), 2, "the loop was not seen through CR-only framing")
        self.assertEqual(data.replace(KEEPALIVE_COMMENT, b""), b"data: [DONE]\r\r")

    def test_raw_unicode_line_separator_in_a_delta_does_not_blind_the_guard(self):
        unit = "loop line again; "

        def plan(i, path, body):
            if i:
                return responses_stream([("text", chunks("fine", 2))])
            return itertools.islice((("event: response.reasoning_text.delta\ndata: " + json.dumps(
                {"type": "response.reasoning_text.delta", "item_id": "a", "delta": d + " "},
                ensure_ascii=False) + "\n\n").encode() for d in looping("", unit)), 3000)

        e = self.engine(plan)
        p = self.proxy(e)
        post(p.port, "/v1/responses", self.RESP)
        self.assertEqual(len(e.requests), 2)

    # caps
    def test_buffer_mode_is_bounded_by_switching_to_live_forwarding(self):
        e = self.engine(endless_clean(0.0005))
        p = self.proxy(e, mode="buffer", max_held=50_000)
        with open_request(p.port, "/v1/responses", self.RESP, timeout=8) as s:
            got = b""
            while len(got) < 150_000:
                chunk = s.recv(1 << 16)  # times out (the old code held everything) instead of reaching 150 kB
                self.assertTrue(chunk)
                got += chunk
        self.assertIn(b"reasoning_text.delta", got)
        self.assertTrue(wait_for(lambda: e.aborted == [0], 6))
        self.assertIn("held_overflow", self.kinds())

    def test_a_loop_after_the_buffer_cap_ends_in_a_clean_error_not_a_retry(self):
        loop = text("loop_reasoning_letters_p3")
        prefix = "".join(f"step {n}: thinking about item {n * 7 % 13}. " for n in range(2000))

        def plan(i, path, body):
            return responses_stream([("reasoning", looping(prefix + loop, loop[-3:]))], tail=False)

        e = self.engine(plan)
        p = self.proxy(e, max_held=20_000)
        _, data = post(p.port, "/v1/responses", self.RESP)
        self.assertEqual(events_of(data)[-1][0], "response.failed")
        self.assertEqual(len(e.requests), 1)  # the client already saw output: no resample
        self.assertTrue(wait_for(lambda: e.aborted == [0]))
        self.assertEqual(p.stats["failed"], 1)
        self.assertIn("held_overflow", self.kinds())

    def test_stream_mode_stops_reading_a_client_that_stopped_reading(self):
        produced = []
        pad = "x" * 16000

        def plan(i, path, body):
            def events():
                for n in range(3000):  # 48 MB if nothing pushes back
                    produced.append(n)
                    yield (f"event: ping\ndata: {json.dumps({'pad': pad})}\n\n").encode()
            return events()

        e = self.engine(plan)
        p = self.proxy(e, mode="stream")
        s = socket.create_connection(("127.0.0.1", p.port))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        body = json.dumps(self.RESP).encode()
        s.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        time.sleep(2.0)
        n = len(produced)
        self.cleanup.append(s.close)
        self.assertLess(n * 16000, 40 << 20, "upstream fully drained into the proxy while the client read nothing")

    def test_pump_read_ahead_is_bounded_and_stops(self):
        class Endless:
            reads = 0

            def read(self, n=65536):
                self.reads += 1
                return b"x" * 10

        up = Endless()
        pump = g.Pump(up, 4)
        time.sleep(0.5)
        self.assertLessEqual(up.reads, 4 + 2)
        pump.stop.set()
        time.sleep(0.5)
        settled = up.reads
        time.sleep(0.3)
        self.assertEqual(up.reads, settled)

    def test_an_oversized_event_fails_open_to_a_raw_relay(self):
        big = b"data: " + b"y" * 300_000 + b"\n\n"
        tail = b"data: [DONE]\n\n"
        e = self.engine(lambda i, path, body: iter([sse({"choices": [{"delta": {"content": "hi"}}]}), big, tail]))
        p = self.proxy(e, max_event=100_000)
        _, data = post(p.port, "/v1/chat/completions", self.CHAT)
        self.assertEqual(data.replace(KEEPALIVE_COMMENT, b""), sse({"choices": [{"delta": {"content": "hi"}}]}) + big + tail)
        self.assertTrue(wait_for(lambda: "guard_disabled" in self.kinds()))

    # D6
    def dead_upstream(self):
        """A TCP server that reads the request and hangs up without answering."""
        class Hangup(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.recv(65536)

        srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Hangup)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.cleanup.append(lambda: (srv.shutdown(), srv.server_close()))
        return srv.server_address[1]

    def test_upstream_hangup_before_headers_is_a_502_and_an_upstream_error(self):
        port = self.dead_upstream()
        p = self.started(g.GuardProxy(f"http://127.0.0.1:{port}", log=self.log))
        self.assertEqual(post(p.port, "/v1/responses", self.RESP)[0], 502)
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("GET", "/health")
        self.assertEqual(c.getresponse().status, 502)
        kinds = self.kinds()
        self.assertEqual(kinds.count("upstream_error"), 2)
        self.assertNotIn("client_disconnected", kinds)

    def test_upstream_dying_mid_stream_ends_cleanly_with_an_upstream_error(self):
        def plan(i, path, body):
            def gen():
                yield from responses_stream([("reasoning", chunks("abc def. " * 20, 10))], tail=False)
                raise ConnectionResetError("engine died")
            return gen()

        for mode in ("buffer", "stream"):
            with self.subTest(mode):
                self.log = os.path.join(self.tmp.name, f"{mode}.jsonl")
                e = self.engine(plan)
                p = self.proxy(e, mode=mode)
                status, data = post(p.port, "/v1/responses", self.RESP)
                self.assertEqual(status, 200)
                last = events_of(data)[-1]
                self.assertEqual((last[0], last[1]["response"]["error"]["code"]), ("response.failed", "upstream_error"))
                self.assertTrue(wait_for(lambda: "upstream_error" in self.kinds()))
                self.assertNotIn("client_disconnected", self.kinds())

    def test_non_json_and_non_object_bodies_on_guarded_paths_are_logged(self):
        e = self.engine(lambda *a: iter(()))
        p = self.proxy(e)
        self.assertEqual(post_raw(p.port, "/v1/chat/completions", b"not json")[0], 200)
        self.assertEqual(post_raw(p.port, "/v1/chat/completions", b"[1, 2]")[0], 200)
        rows = [r for r in self.log_rows() if r["ev"] == "unguarded"]
        self.assertEqual([r["reason"] for r in rows], ["body is not JSON", "body is not a JSON object"])
        self.assertEqual(p.stats["guarded"], 0)

    def test_legacy_completions_endpoint_is_guarded(self):
        loop = text("loop_reasoning_letters_p3")
        e = self.engine(lambda i, path, body: completions_stream(itertools.islice(looping(loop, loop[-3:]), 4000))
                        if i == 0 else completions_stream(["recovered"]))
        p = self.proxy(e)
        _, data = post(p.port, "/v1/completions", {"model": "m", "prompt": "x", "stream": True})
        self.assertEqual(len(e.requests), 2)
        self.assertIn("recovered", joined(data))

    def test_legacy_function_call_loop_is_guarded(self):
        unit = '"a": 0, '
        e = self.engine(lambda i, path, body: chat_stream([("function_call", itertools.islice(looping("", unit), 3000))])
                        if i == 0 else chat_stream([("content", chunks("recovered", 3))]))
        p = self.proxy(e)
        _, data = post(p.port, "/v1/chat/completions", self.CHAT)
        self.assertEqual(len(e.requests), 2)
        self.assertEqual([r["channel"] for r in self.log_rows() if r["ev"] == "loop"], [[0, "function_call"]])

    def test_malformed_event_shapes_do_not_crash_the_handler(self):
        bad = [{"choices": [{"delta": "oops"}]}, {"choices": 5}, {"choices": [{"delta": {"tool_calls": [None, 3]}}]},
               {"type": "response.output_text.delta", "item_id": ["x"], "delta": "t"}]
        events = [sse(b) for b in bad] + [sse({"choices": [{"delta": {"content": "fine"}}]}), b"data: [DONE]\n\n"]
        for mode in ("buffer", "stream"):
            with self.subTest(mode):
                e = self.engine(lambda i, path, body: iter(events))
                p = self.proxy(e, mode=mode)
                _, data = post(p.port, "/v1/chat/completions", self.CHAT)
                self.assertEqual(data.replace(KEEPALIVE_COMMENT, b""), b"".join(events))
                self.assertEqual(p.stats["loops"], 0)

    # 9
    def test_conversation_id_is_not_forwarded(self):
        e = self.engine(lambda i, path, body: chat_stream([("content", chunks("hi", 1))]))
        p = self.proxy(e)
        headers = {"X-Conversation-Id": "c-123", "Authorization": "Bearer k"}
        post_raw(p.port, "/v1/chat/completions", json.dumps(self.CHAT).encode(), headers)
        post_raw(p.port, "/v1/chat/completions", json.dumps(dict(self.CHAT, stream=False)).encode(), headers)
        self.assertEqual(len(e.requests), 2)
        for r in e.requests:
            self.assertNotIn("x-conversation-id", r["headers"])
            self.assertEqual(r["headers"]["authorization"], "Bearer k")


KEEPALIVE_COMMENT = b": runaway-guard keepalive\n\n"


class OptInAndDefaults(unittest.TestCase):
    def test_guarded_port_is_a_no_op_without_the_variable(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN_RUNAWAY")}
        before = threading.active_count()
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            with g.guarded_port(8081, "/nonexistent") as port:
                self.assertEqual(port, 8081)
                self.assertEqual(threading.active_count(), before)

    ENV_OFF = ({}, {"QWEN_RUNAWAY_GUARD": ""}, {"QWEN_RUNAWAY_GUARD": "0"})

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN_RUNAWAY")}
        env.update(extra)
        return unittest.mock.patch.dict(os.environ, env, clear=True)

    def test_off_means_off_in_every_entry_point(self):
        for extra in self.ENV_OFF:
            with self.subTest(extra), self.env(**extra):
                before = threading.active_count()
                self.assertIsNone(g.guard_from_env(8081, "/nonexistent"))  # the old default was a buffer guard
                with g.guarded_port(8081, "/nonexistent") as port:
                    self.assertEqual(port, 8081)
                self.assertEqual(threading.active_count(), before)

    def test_only_buffer_and_stream_enable_it_and_anything_else_raises(self):
        for value, mode in (("buffer", "buffer"), ("stream", "stream")):
            with self.subTest(value), self.env(QWEN_RUNAWAY_GUARD=value):
                proxy = g.guard_from_env(8081)
                try:
                    self.assertEqual(proxy.mode, mode)
                finally:
                    proxy.server_close()
        for value in ("1", "true", "on", "buffr", "Buffer", "off", " stream"):
            with self.subTest(value), self.env(QWEN_RUNAWAY_GUARD=value):
                with self.assertRaisesRegex(ValueError, "QWEN_RUNAWAY_GUARD"):
                    with g.guarded_port(8081, "/nonexistent"):
                        self.fail("a typo must not silently start (or skip) the guard")
                with self.assertRaisesRegex(ValueError, "QWEN_RUNAWAY_GUARD"):
                    g.guard_from_env(8081)

    def test_bad_numeric_env_names_the_variable(self):
        for name in ("QWEN_RUNAWAY_MIN_CHARS", "QWEN_RUNAWAY_RETRIES"):
            with self.subTest(name), self.env(QWEN_RUNAWAY_GUARD="buffer", **{name: "lots"}):
                with self.assertRaisesRegex(ValueError, name):
                    g.guard_from_env(8081)
        with self.env(QWEN_RUNAWAY_GUARD="buffer", QWEN_RUNAWAY_INJECT_SEED="yes"):
            with self.assertRaisesRegex(ValueError, "QWEN_RUNAWAY_INJECT_SEED"):
                g.guard_from_env(8081)

    def test_inject_seed_is_env_opt_in(self):
        for value, expect in ((None, False), ("0", False), ("1", True)):
            with self.subTest(value), self.env(QWEN_RUNAWAY_GUARD="buffer",
                                               **({"QWEN_RUNAWAY_INJECT_SEED": value} if value else {})):
                proxy = g.guard_from_env(8081)
                try:
                    self.assertIs(proxy.inject_seed, expect)
                finally:
                    proxy.server_close()

    def test_default_threshold_is_12288(self):
        self.assertEqual(g.MIN_CHARS, 12288)
        self.assertEqual(g.LoopDetector().min_chars, 12288)

    def test_guarded_port_starts_a_proxy_when_enabled(self):
        e = FakeEngine(lambda i, path, body: responses_stream([("text", chunks("hi", 1))]))
        try:
            with tempfile.TemporaryDirectory() as out, \
                    unittest.mock.patch.dict(os.environ, {"QWEN_RUNAWAY_GUARD": "buffer"}):
                with g.guarded_port(e.port, out) as port:
                    self.assertNotEqual(port, e.port)
                    status, data = post(port, "/v1/responses", {"model": "m", "input": "x", "stream": True})
                    self.assertEqual(status, 200)
                self.assertTrue(Path(out, "runaway-guard.jsonl").exists())
        finally:
            e.stop()

    def test_importing_the_runners_does_not_import_the_guard(self):
        code = ("import sys; import scripts.instruments.agentic_probe, scripts.instruments.isolated_exec, "
                "scripts.instruments.hillclimb, scripts.instruments.swift_longtail, scripts.instruments.run_polyglot; "
                "import scripts.experiments.dual_5060ti.run_swift_longtail; "
                "assert 'scripts.instruments.runaway_guard' not in sys.modules; print('clean')")
        env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN_RUNAWAY")}
        env["CUDA_VISIBLE_DEVICES"] = ""
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "clean"), r.stderr[-2000:])

    def test_guard_module_has_no_import_side_effects(self):
        code = ("import threading, socket, sys; t = threading.active_count(); "
                "s = socket.socket.__init__; "
                "import scripts.instruments.runaway_guard as g; "
                "assert threading.active_count() == t; "
                "import os; assert not any(k.startswith('QWEN_RUNAWAY') for k in os.environ); print('ok')")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "ok"), r.stderr[-2000:])

    def test_runner_files_do_not_reference_the_guard_unless_lazily(self):
        # default path stays byte-for-byte what it was: any hook in a runner must be env-gated and lazy
        for rel in ("scripts/instruments/agentic_probe.py", "scripts/experiments/dual_5060ti/run_vllm_longtail.py",
                    "scripts/instruments/hillclimb.py", "scripts/experiments/dual_5060ti/run_swift_longtail.py"):
            src = (ROOT / rel).read_text()
            for i, line in enumerate(src.splitlines()):
                if "runaway_guard" in line:
                    self.assertTrue(line.startswith(" "), f"{rel}:{i + 1} imports the guard at module level")


class IntegrationPatch(unittest.TestCase):
    PATCH = FIX / "runner-integration.patch"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for rel in ("scripts/instruments/agentic_probe.py", "scripts/experiments/dual_5060ti/run_vllm_longtail.py"):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text((ROOT / rel).read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def test_patch_applies_to_the_current_runners(self):
        r = subprocess.run(["git", "apply", "--check", str(self.PATCH)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_patched_default_path_dials_the_original_port_without_importing_the_guard(self):
        subprocess.run(["git", "apply", str(self.PATCH)], cwd=self.root, check=True)
        code = f"""
import ast, sys
src = open('scripts/instruments/agentic_probe.py').read()
ns = {{}}
tree = ast.parse(src)
helper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_guard_port'][0]
exec('import os, contextlib\\n' + ast.get_source_segment(src, helper), ns)
with ns['_guard_port'](8081, '/nonexistent') as port:
    assert port == 8081
assert 'scripts.instruments.runaway_guard' not in sys.modules
print('default-identical')
"""
        env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN_RUNAWAY")}
        r = subprocess.run([sys.executable, "-c", code], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "default-identical"), r.stderr[-2000:])
        patched = (self.root / "scripts/instruments/agentic_probe.py").read_text()
        vllm = (self.root / "scripts/experiments/dual_5060ti/run_vllm_longtail.py").read_text()
        self.assertIn('os.environ.get("QWEN_RUNAWAY_GUARD", "") in ("", "0")', patched)  # "0" is off ...
        self.assertIn('os.environ.get("QWEN_RUNAWAY_GUARD", "") not in ("", "0")', vllm)  # ... on the vLLM path too
        self.assertIn("with _guard_port(port, out) as port, isolated_exec.inference_bridge(bridge, port):", patched)
        self.assertNotIn("\nfrom scripts.instruments import runaway_guard", patched)  # only the lazy, indented import

    def test_patched_gate_treats_zero_as_off_and_a_typo_as_an_error(self):
        subprocess.run(["git", "apply", str(self.PATCH)], cwd=self.root, check=True)
        code = """
import ast, sys, os
src = open('scripts/instruments/agentic_probe.py').read()
ns = {}
helper = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == '_guard_port'][0]
exec('import os, contextlib\\n' + ast.get_source_segment(src, helper), ns)
value = os.environ['QWEN_RUNAWAY_GUARD']
try:
    with ns['_guard_port'](8081, '/nonexistent') as port:
        print('port', port, 'imported', 'scripts.instruments.runaway_guard' in sys.modules)
except ValueError as e:
    print('ValueError', e)
"""
        base = {k: v for k, v in os.environ.items() if not k.startswith("QWEN_RUNAWAY")}
        base["PYTHONPATH"] = str(ROOT)
        for value, expect in (("0", "port 8081 imported False"), ("", "port 8081 imported False"),
                              ("buffr", "ValueError")):
            with self.subTest(value):
                r = subprocess.run([sys.executable, "-c", code], cwd=self.root, env=dict(base, QWEN_RUNAWAY_GUARD=value),
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertTrue(r.stdout.startswith(expect), r.stdout)


CODEX = Path.home() / ".codex/packages/standalone/releases/0.154.0-x86_64-unknown-linux-musl/bin/codex"


@unittest.skipUnless(CODEX.exists(), "pinned Codex binary not installed")
class RealCodexClient(ProxyCase):
    """The real Codex CLI (Responses API, tool-capable client) against the guard and a fake engine; no GPU, no network."""

    def run_codex(self, port):
        home, work = (Path(tempfile.mkdtemp(dir=self.tmp.name)) for _ in range(2))
        (home / "config.toml").write_text(f'''model = "local-benchmark"
model_provider = "benchmark"
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"
[model_providers.benchmark]
name = "t"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 2
stream_max_retries = 2
stream_idle_timeout_ms = 60000
''')
        env = {"CODEX_HOME": str(home), "HOME": str(work), "PATH": os.environ.get("PATH", "")}
        return subprocess.run([str(CODEX), "exec", "--skip-git-repo-check", "--json", "-"], input="say hi", cwd=work,
                              env=env, capture_output=True, text=True, timeout=120)

    def check(self, mode, where):
        loop = text("loop_reasoning_letters_p3") if where == "reasoning" else text("loop_args_table_p23")
        item = ("reasoning", looping(loop, loop[-3:])) if where == "reasoning" else ("args", looping(loop, loop[-23:]))
        e = self.engine(lambda i, path, body: responses_stream([item], tail=False) if i == 0 else
                        responses_stream([("text", chunks("All done, nothing to change.", 6))]))
        p = self.proxy(e, mode=mode)
        r = self.run_codex(p.port)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
        self.assertIn("All done, nothing to change.", r.stdout)
        self.assertEqual(len(e.requests), 2)
        self.assertTrue(wait_for(lambda: e.aborted == [0]))
        return r

    def test_buffer_mode_codex_never_sees_the_loop(self):
        for where in ("reasoning", "args"):
            with self.subTest(where):
                r = self.check("buffer", where)
                self.assertNotIn("Reconnecting", r.stdout)

    def test_stream_mode_codex_reconnects_on_the_clean_error(self):
        # measured with codex 0.154: response.failed counts as a retryable stream error (spends stream_max_retries)
        r = self.check("stream", "reasoning")
        self.assertIn("Reconnecting... 1/2", r.stdout)


if __name__ == "__main__":
    unittest.main()
