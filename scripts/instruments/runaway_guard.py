"""Client-side runaway (repetition-loop) guard for OpenAI-compatible streaming traffic.

Opt-in. Nothing imports this module unless a runner is told to; the default campaign path is untouched.
The only mitigation that keeps the sampling law intact is abort + resample, so that is all this does:
detect an exact suffix period in a generated stream, drop the upstream request, ask for a fresh sample.

  python -m scripts.instruments.runaway_guard proxy --upstream http://127.0.0.1:8080 --port 8090
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import queue
import random
import re
import socket
import socketserver
import sys
import threading
import time
import urllib.parse

# Calibrated on the recorded loops and the non-loop long outputs; see docs in the integration note.
P_MAX = 2000        # longest period (chars) that can be a loop unit
MIN_REPEATS = 4     # the unit must repeat at least this many times ...
MIN_CHARS = 12288   # ... and the periodic suffix must cover at least this many chars (legit source tables reach ~8.8k)
GRAM = 16           # seed length: a candidate period is only proposed when a GRAM-char window recurs
MAX_CANDIDATES = 64
MAX_OCC = 16        # a recurring gram proposes periods from this many of its most recent occurrences
MAX_HELD = 32 << 20     # buffer mode: bytes held per attempt (a 32k-token response is ~10 MB of SSE)
MAX_EVENT = 16 << 20    # bytes of one unterminated SSE event
QUEUE_CHUNKS = 64       # upstream read-ahead, in 64 KiB reads


class LoopDetector:
    """Streaming exact suffix-period detector, O(1) amortised per char.

    A candidate period p is proposed when the last GRAM chars occurred before at distance p <= P_MAX
    (hash of the most recent MAX_OCC occurrences) and survives while every new char equals the char p back.
    Older occurrences matter because a gram that recurs inside the unit (a loop of indented, near-identical
    lines) never has the true period as its latest distance. A gram of period <= 2 (a run of one char or "abab")
    only proposes its latest occurrence: its other recurrences are the run itself, not the unit.
    Fires when a candidate's periodic suffix reaches max(min_chars, min_repeats * p).
    Contract: feed() is the only mutator; after a hit the detector is spent (call reset() to reuse it).
    """

    __slots__ = ("p_max", "min_repeats", "min_chars", "gram", "max_candidates", "max_occ", "buf", "base", "last",
                 "older", "cands", "n", "hit", "record_min", "runs", "_prune_at")

    def __init__(self, p_max=P_MAX, min_repeats=MIN_REPEATS, min_chars=MIN_CHARS, gram=GRAM,
                 max_candidates=MAX_CANDIDATES, record_min=0, max_occ=MAX_OCC):
        self.p_max, self.min_repeats, self.min_chars = p_max, min_repeats, min_chars
        self.gram, self.max_candidates, self.max_occ = gram, max_candidates, max_occ
        self.record_min = record_min  # > 0: keep every maximal periodic run at least this long in .runs
        self.runs = []
        self.reset()

    def reset(self):
        self.buf = ""          # tail of the stream; buf[k] is absolute index base + k
        self.base = 0
        self.last = {}         # gram -> absolute index of its last char
        self.older = {}        # gram -> earlier such indices, newest first (at most max_occ - 1)
        self.cands = {}        # period -> absolute start of the periodic suffix
        self.n = 0             # chars seen
        self.hit = None
        self._prune_at = self.p_max

    def _record(self, p, start, end):
        if end - start >= self.record_min:
            self.runs.append((p, end - start, start))

    def feed(self, text):
        """Append text; return the hit dict on the first trigger, else None."""
        if self.hit is not None:
            return self.hit
        if not text:
            return None
        old = len(self.buf)
        buf = self.buf + text
        base, last, older, cands = self.base, self.last, self.older, self.cands
        gram, p_max, record = self.gram, self.p_max, self.record_min > 0
        max_occ = self.max_occ
        min_chars, min_rep = self.min_chars, self.min_repeats
        for k in range(old, len(buf)):
            c = buf[k]
            i = base + k
            if cands:
                dead = []
                for p, start in cands.items():
                    if buf[k - p] != c:
                        dead.append(p)
                    elif i + 1 - start >= min_chars and i + 1 - start >= min_rep * p:
                        self.hit = dict(period=p, chars=i + 1 - start, repeats=(i + 1 - start) / p,
                                        start=start, at=i + 1, unit=buf[k + 1 - p:k + 1])
                        if record:
                            self._record(p, start, i + 1)
                        self.buf, self.n = buf, i + 1
                        return self.hit
                if dead:
                    for p in dead:
                        if record:
                            self._record(p, cands[p], i)
                        del cands[p]
            if k + 1 >= gram:
                g = buf[k + 1 - gram:k + 1]
                j = last.get(g)
                last[g] = i
                if j is not None:
                    prev = older.get(g)
                    flat = g[2:] == g[:-2]
                    if not flat:
                        older[g] = ([j] + prev)[:max_occ - 1] if prev else [j]
                    for jj in (j,) if flat or not prev else (j, *prev):
                        p = i - jj
                        if p > p_max:
                            break
                        if p not in cands and k - p >= 0:
                            # extend backwards over chars that already matched; seeds only fire on the first recurrence
                            s = k + 1 - gram
                            floor = max(0, s - p - 2 * p_max)
                            while s - 1 >= 0 and s - 1 - p >= floor and buf[s - 1] == buf[s - 1 - p]:
                                s -= 1
                            if len(cands) >= self.max_candidates:
                                worst = min(cands, key=lambda q: cands[q] * -1)  # newest start = shortest run
                                if record:
                                    self._record(worst, cands[worst], i)
                                del cands[worst]
                            cands[p] = base + s - p
            if i >= self._prune_at:
                lim = i - p_max
                self.last = last = {g: j for g, j in last.items() if j >= lim}
                self.older = older = {g: v for g, v in ((g, [x for x in v if x >= lim]) for g, v in older.items()) if v}
                self._prune_at = i + p_max
        self.n = base + len(buf)
        if len(buf) > 4 * p_max + 4 * gram:
            cut = len(buf) - (2 * p_max + 2 * gram)
            buf = buf[cut:]
            self.base = base + cut
        self.buf = buf
        return None

    def finish(self):
        """End of stream: flush open runs into .runs (analysis only)."""
        if self.record_min > 0:
            for p, start in self.cands.items():
                self._record(p, start, self.n)
        self.cands = {}


def scan(text, **kw):
    """Whole-text convenience: the first hit or None."""
    d = LoopDetector(**kw)
    return d.feed(text)


# ---------------------------------------------------------------- SSE parsing

def _hashable(x):
    return x if x is None or isinstance(x, (str, int, float, bool)) else json.dumps(x, sort_keys=True, default=str)


def sse_texts(event_name, data):
    """(channel, text) pairs carried by one SSE event; channel identifies one generated text stream.

    Responses: any `response.*.delta` with a string `delta`. Chat Completions: delta.content,
    delta.reasoning_content / reasoning, delta.tool_calls[i].function.arguments and the legacy
    delta.function_call.arguments per choice. Contract: upstream JSON of any shape yields pairs or nothing, never raises.
    """
    if not isinstance(data, dict):
        return []
    out = []
    typ = data.get("type") or event_name or ""
    if isinstance(typ, str) and typ.startswith("response.") and typ.endswith(".delta"):
        if isinstance(data.get("delta"), str):
            out.append(((typ, _hashable(data.get("item_id")), _hashable(data.get("output_index")),
                         _hashable(data.get("content_index"))), data["delta"]))
        return out
    choices = data.get("choices")
    for choice in choices if isinstance(choices, list) else ():
        if not isinstance(choice, dict):
            continue
        idx = _hashable(choice.get("index", 0))
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        for field in ("content", "reasoning_content", "reasoning"):
            if isinstance(delta.get(field), str):
                out.append(((idx, field), delta[field]))
        calls = delta.get("tool_calls")
        for call in calls if isinstance(calls, list) else ():
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                out.append(((idx, "tool", _hashable(call.get("index", 0))), fn["arguments"]))
        legacy = delta.get("function_call")
        if isinstance(legacy, dict) and isinstance(legacy.get("arguments"), str):
            out.append(((idx, "function_call"), legacy["arguments"]))
        if isinstance(choice.get("text"), str):  # legacy /v1/completions
            out.append(((idx, "text"), choice["text"]))
    return out


_EOL = re.compile(rb"\r\n|\n|\r")


class FrameOverflow(ValueError):
    pass


class SseFramer:
    """Incremental SSE event splitter over raw bytes. Line ends are CRLF, LF or lone CR (WHATWG SSE), a blank line ends
    an event. Contract: everything feed() returns plus finish() concatenates back to exactly the bytes fed, whatever
    the read boundaries; only new bytes are scanned (linear in the stream, one huge event included).

    A comment line that opens an event is emitted alone, so keepalive pings never accumulate. A trailing CR is not
    a line end until the next byte shows whether an LF follows, except as the end of a blank line (the event is
    complete either way; a stray LF then surfaces as its own blank piece). `overflowed` is set when one unfinished
    event exceeds max_event; the caller must stop framing (finish() returns the raw tail).
    """

    def __init__(self, max_event=MAX_EVENT):
        self.max_event = max_event
        self.buf = bytearray()
        self.scan = 0
        self.cur = []
        self.cur_size = 0
        self.overflowed = False

    @property
    def pending(self):
        return self.cur_size + len(self.buf)

    def feed(self, data):
        buf = self.buf
        buf += data
        out, start, scan = [], 0, self.scan
        while True:
            m = _EOL.search(buf, scan)
            if m is None:
                scan = len(buf)
                break
            end = m.end()
            blank = m.start() == start
            if end == len(buf) and buf[end - 1] == 13 and not blank:
                scan = m.start()
                break
            line = bytes(buf[start:end])
            if blank:
                out.append(b"".join(self.cur) + line if self.cur else line)
                self.cur, self.cur_size = [], 0
            elif not self.cur and line[0] == 58:
                out.append(line)
            else:
                self.cur.append(line)
                self.cur_size += len(line)
            start = scan = end
        if start:
            del buf[:start]
        self.scan = scan - start
        if self.pending > self.max_event:
            self.overflowed = True
        return out

    def finish(self):
        tail = b"".join(self.cur) + bytes(self.buf)
        self.buf.clear()
        self.cur, self.cur_size, self.scan = [], 0, 0
        return tail


def split_events(buf):
    """One-shot framing of buf: (complete raw events, unterminated rest)."""
    framer = SseFramer(max_event=float("inf"))
    events = framer.feed(buf)
    return events, framer.finish()


_LINES = re.compile(rb"\r\n|\r|\n")


def parse_event(raw):
    # split the bytes, not the decoded text: str.splitlines() also breaks on U+2028 / U+0085 inside a JSON string
    name, data = None, []
    for line in _LINES.split(raw):
        if line.startswith(b"event:"):
            name = line[6:].decode("utf-8", "replace").strip()
        elif line.startswith(b"data:"):
            data.append(line[6:] if line[5:6] == b" " else line[5:])
    body = b"\n".join(data).decode("utf-8", "replace")
    if not body or body.strip() == "[DONE]":
        return name, None
    try:
        return name, json.loads(body)
    except (ValueError, RecursionError):
        return name, None


class Detectors:
    """One LoopDetector per generated text stream of a response."""

    def __init__(self, **params):
        self.params, self.by = params, {}
        self.chars = self.deltas = 0

    def feed_event(self, name, data):
        for chan, text in sse_texts(name, data):
            d = self.by.get(chan)
            if d is None:
                d = self.by[chan] = LoopDetector(**self.params)
            self.chars += len(text)
            self.deltas += 1
            hit = d.feed(text)
            if hit:
                return dict(hit, channel=list(chan), unit=hit["unit"][:200])
        return None


def failure_event(path, code, message, response_id="resp_runaway_guard"):
    if path.rstrip("/").endswith("/responses"):
        data = {"type": "response.failed", "response": {"id": response_id, "object": "response", "status": "failed",
                                                        "error": {"code": code, "message": message}}}
        return b"event: response.failed\ndata: " + json.dumps(data).encode() + b"\n\n"
    data = {"error": {"message": message, "type": code, "code": code}}
    return b"data: " + json.dumps(data).encode() + b"\n\n"


# ---------------------------------------------------------------- proxy

# x-conversation-id: llama-server only cancels such a stream through DELETE /v1/stream, not on socket close
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "content-length", "host", "x-conversation-id"}
GUARDED = ("/responses", "/chat/completions", "/completions")
MODES = ("buffer", "stream")
KEEPALIVE = b": runaway-guard keepalive\n\n"


class ClientGone(Exception):
    """The client socket failed or closed."""


class UpstreamError(Exception):
    """The upstream leg failed (connect, headers, mid-body)."""


class Upstream:
    """One upstream HTTP exchange. close() must stop the generation: it shuts down the socket captured before
    getresponse(), because on `Connection: close` http.client detaches it from the connection (conn.sock is None)
    and hands it to the response."""

    def __init__(self, host, port):
        self.conn = http.client.HTTPConnection(host, port, timeout=None)
        self.sock = None
        self.resp = None

    def open(self, method, target, body, headers):
        try:
            self.conn.connect()
            self.sock = self.conn.sock
            self.conn.request(method, target, body or None, headers)
            self.resp = self.conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            self.close()
            raise UpstreamError(f"{type(e).__name__}: {e}") from e
        return self.resp

    def read(self, n=65536):
        try:
            return self.resp.read1(n)
        except (OSError, http.client.HTTPException) as e:
            raise UpstreamError(f"{type(e).__name__}: {e}") from e

    def close(self):
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for closable in (self.resp, self.conn):
            try:
                if closable is not None:
                    closable.close()
            except OSError:
                pass


class Pump:
    """Reads the upstream body on a thread into a bounded queue (items: bytes, b"" at EOF, or an Exception).
    The bound is backpressure: a slow client stops the reads and TCP stalls the engine; nothing is dropped."""

    def __init__(self, upstream, maxsize):
        self.queue = queue.Queue(maxsize)
        self.stop = threading.Event()
        threading.Thread(target=self.run, args=(upstream,), daemon=True).start()

    def put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def run(self, upstream):
        try:
            while not self.stop.is_set():
                c = upstream.read()
                self.put(c)
                if not c:
                    return
        except Exception as e:
            self.put(e)


class GuardProxy(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, upstream, port=0, host="127.0.0.1", log=None, mode="buffer", retries=1, keepalive=10.0,
                 client_seed="error", detector=None, inject_seed=False, max_held=MAX_HELD, max_event=MAX_EVENT,
                 queue_chunks=QUEUE_CHUNKS, poll=1.0):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: expected one of {MODES}")
        u = urllib.parse.urlsplit(upstream if "//" in str(upstream) else f"http://127.0.0.1:{upstream}")
        self.up_host, self.up_port, self.up_prefix = u.hostname, u.port or 80, u.path.rstrip("/")
        self.log_path, self.mode, self.retries = log, mode, retries
        self.keepalive, self.client_seed, self.inject_seed = keepalive, client_seed, inject_seed
        self.max_held, self.max_event, self.queue_chunks, self.poll = max_held, max_event, queue_chunks, poll
        self.detector = dict(detector or {})
        self.lock = threading.Lock()
        self.counter = 0
        self.stats = dict(requests=0, guarded=0, loops=0, retries=0, failed=0, upstream_errors=0)
        super().__init__((host, port), GuardHandler)
        self.port = self.server_address[1]
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.shutdown()
        self.server_close()

    def next_id(self):
        with self.lock:
            self.counter += 1
            return self.counter

    def bump(self, key):
        with self.lock:
            self.stats[key] += 1

    def event(self, **row):
        row["t"] = time.time()
        if self.log_path:
            with self.lock, open(self.log_path, "a") as stream:
                stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")


class GuardHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            body = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0], 16)
                if not size:
                    while self.rfile.readline().strip():
                        pass
                    return body
                body += self.rfile.read(size)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def fwd_headers(self):
        return {k: v for k, v in self.headers.items() if k.lower() not in HOP}

    def open_upstream(self, body, headers):
        server = self.server
        up = Upstream(server.up_host, server.up_port)
        up.open(self.command, server.up_prefix + self.path, body, headers)
        return up

    def chunk(self, data):
        if not data:  # a zero-length chunk would end the stream
            return
        try:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        except OSError as e:
            raise ClientGone(str(e)) from e

    def end_chunks(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError as e:
            raise ClientGone(str(e)) from e

    def client_gone(self):
        """True once the client closed its end. A client that half-closes its write side counts as gone."""
        try:
            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    def start_stream(self, resp):
        self.started = True
        try:
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        except OSError as e:
            raise ClientGone(str(e)) from e

    def relay_body(self, up):
        while chunk := up.read():
            self.chunk(chunk)
        self.end_chunks()

    def relay(self):
        server = self.server
        self.started = False  # response headers already sent to the client
        self.committed = False  # response body bytes already sent to the client
        server.bump("requests")
        try:
            body = self.read_body()
        except (ValueError, OSError):
            self.close_connection = True
            return self.reply_error(400, "bad request body")
        path = self.path.split("?")[0]
        guarded = False
        parsed = None
        if self.command == "POST" and path.endswith(GUARDED):
            try:
                parsed = json.loads(body)
                guarded = isinstance(parsed, dict) and parsed.get("stream") is True
                if not isinstance(parsed, dict):
                    server.event(ev="unguarded", path=path, reason="body is not a JSON object")
            except (ValueError, RecursionError):
                server.event(ev="unguarded", path=path, reason="body is not JSON")
        if guarded:
            return self.guarded(body, parsed, path)
        up = None
        try:
            up = self.open_upstream(body, self.fwd_headers())
            self.start_stream(up.resp)
            self.relay_body(up)
        except ClientGone:
            self.close_connection = True
        except UpstreamError as e:
            server.bump("upstream_errors")
            server.event(ev="upstream_error", path=path, error=str(e))
            self.fail_upstream(path, e, guarded=False)
        finally:
            if up:
                up.close()

    def reply_error(self, status, detail):
        try:
            self.send_error(status, None, detail)
        except OSError:
            pass
        self.close_connection = True

    def fail_upstream(self, path, e, guarded=True, code="upstream_error"):
        """Contract: ends the response cleanly and never raises."""
        self.close_connection = True
        try:
            if not self.started:
                self.reply_error(502, str(e)[:200])
            elif guarded:
                self.finish_failed(path, f"{code}: {e}", code=code)
        except (ClientGone, OSError):
            pass

    # One guarded streaming request: up to 1 + retries upstream attempts (buffer mode only).
    def guarded(self, body, parsed, path):
        server = self.server
        server.bump("guarded")
        rid = server.next_id()
        t0 = time.monotonic()
        client_seed = parsed.get("seed") is not None
        server.event(ev="request", id=rid, path=path, mode=server.mode, client_seed=client_seed,
                     model=parsed.get("model"))
        attempts = 1 + (server.retries if server.mode == "buffer" else 0)
        headers = self.fwd_headers()
        try:
            for attempt in range(attempts):
                req_body = body  # the first attempt and, by default, every retry are the client's own bytes
                if attempt:
                    seed = None
                    if server.inject_seed or (client_seed and server.client_seed == "bump"):
                        seed = random.randrange(1, 2 ** 31)
                        req_body = json.dumps(dict(parsed, seed=seed)).encode()
                    server.bump("retries")
                    server.event(ev="retry", id=rid, attempt=attempt, seed=seed)
                if self.run_attempt(rid, attempt, req_body, headers) != "loop":
                    break
                if self.committed or attempt + 1 == attempts:
                    why = "" if server.mode == "stream" else (
                        ", output already delivered (buffer cap)" if self.committed else ", no retries left")
                    self.finish_failed(path, "runaway repetition detected; upstream aborted" + why)
                    server.bump("failed")
                    break
                if client_seed and server.client_seed == "error":
                    server.event(ev="retry_refused", id=rid, reason="client set seed")
                    self.finish_failed(path, "runaway repetition detected; client set a seed, identical retry refused")
                    server.bump("failed")
                    break
            server.event(ev="end", id=rid, elapsed_s=round(time.monotonic() - t0, 3))
        except ClientGone as e:
            server.event(ev="client_disconnected", id=rid, error=str(e))
            self.close_connection = True
        except UpstreamError as e:
            server.bump("upstream_errors")
            server.event(ev="upstream_error", id=rid, error=str(e))
            self.fail_upstream(path, e)
        except Exception as e:  # nothing may escape to socketserver's traceback printer
            server.event(ev="handler_error", id=rid, error=repr(e))
            self.fail_upstream(path, e, code="proxy_error")

    def finish_failed(self, path, message, code="runaway_detected"):
        if not self.started:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            except OSError as e:
                raise ClientGone(str(e)) from e
            self.started = True
        self.chunk(failure_event(path, code, message))
        self.end_chunks()

    def run_attempt(self, rid, attempt, body, headers):
        """Returns done | passthrough | loop.

        Buffer mode holds the events so a loop leaves no trace at the client. Memory is bounded everywhere:
        the upstream queue by backpressure; `held` by server.max_held, past which this request switches to live
        forwarding (detection stays on, but the client has seen output so a later loop ends in a clean error, not a
        retry); one unterminated event by server.max_event, past which framing and detection are dropped for the rest
        of the response (fail open: raw relay, logged as guard_disabled).
        """
        server = self.server
        up = self.open_upstream(body, headers)
        pump = None
        try:
            if up.resp.status != 200:
                # error pass-through: nothing generated, nothing to guard
                if self.started:
                    raise UpstreamError(f"upstream status {up.resp.status} on retry")
                self.start_stream(up.resp)
                self.relay_body(up)
                return "passthrough"
            if not self.started:
                self.start_stream(up.resp)
            detectors = Detectors(**server.detector)
            framer = SseFramer(server.max_event)
            live = server.mode == "stream"
            held, held_bytes, hit = [], 0, None
            pump = Pump(up, server.queue_chunks)
            a0 = last_out = last_poll = time.monotonic()

            def send(raw):
                self.committed = True
                self.chunk(raw)

            def flush_held():
                nonlocal held, held_bytes
                for raw in held:
                    send(raw)
                held, held_bytes = [], 0

            while True:
                try:
                    c = pump.queue.get(timeout=server.poll)
                except queue.Empty:
                    c = None
                now = time.monotonic()
                if c is None or now - last_poll >= server.poll:
                    last_poll = now
                    if self.client_gone():
                        raise ClientGone("client closed while upstream was streaming")
                if not live and now - last_out >= server.keepalive:
                    self.chunk(KEEPALIVE)  # while holding, whether or not the upstream is talking
                    last_out = now
                if c is None:
                    continue
                if isinstance(c, Exception):
                    raise UpstreamError(str(c)) from c
                if not c:
                    break
                for raw in framer.feed(c):
                    name, data = parse_event(raw)
                    hit = detectors.feed_event(name, data)
                    if hit:
                        break
                    if live:
                        send(raw)
                        last_out = now
                    else:
                        held.append(raw)
                        held_bytes += len(raw)
                        if held_bytes > server.max_held:
                            server.event(ev="held_overflow", id=rid, attempt=attempt, held_bytes=held_bytes,
                                         action="live")
                            flush_held()
                            live = True
                            last_out = now
                if hit:
                    break
                if framer.overflowed:
                    server.event(ev="guard_disabled", id=rid, attempt=attempt, pending_bytes=framer.pending,
                                 reason="one SSE event exceeded max_event; raw relay for the rest")
                    flush_held()
                    send(framer.finish())
                    self.relay_rest(pump)
                    return "done"
            if hit is None:
                tail = framer.finish()
                if tail:
                    if live:
                        send(tail)
                    else:
                        held.append(tail)
                flush_held()
                self.end_chunks()
                server.event(ev="attempt_done", id=rid, attempt=attempt, chars=detectors.chars,
                             deltas=detectors.deltas, seconds=round(time.monotonic() - a0, 3))
                return "done"
        finally:
            if pump:
                pump.stop.set()
            up.close()
        server.bump("loops")
        server.event(ev="loop", id=rid, attempt=attempt, channel=hit["channel"], period=hit["period"],
                     repeats=round(hit["repeats"], 1), loop_chars=hit["chars"], channel_chars=hit["at"],
                     total_chars=detectors.chars, deltas=detectors.deltas, unit=hit["unit"],
                     seconds=round(time.monotonic() - a0, 3), mode=server.mode)
        return "loop"

    def relay_rest(self, pump):
        last_poll = time.monotonic()
        while True:
            try:
                c = pump.queue.get(timeout=self.server.poll)
            except queue.Empty:
                c = None
            if c is None or time.monotonic() - last_poll >= self.server.poll:
                last_poll = time.monotonic()
                if self.client_gone():
                    raise ClientGone("client closed while upstream was streaming")
            if c is None:
                continue
            if isinstance(c, Exception):
                raise UpstreamError(str(c)) from c
            if not c:
                self.end_chunks()
                return
            self.chunk(c)

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = relay


def mode_from_env(environ=None):
    """None (guard off) for unset, empty or "0"; "buffer" or "stream"; anything else is a typo and raises."""
    value = (os.environ if environ is None else environ).get("QWEN_RUNAWAY_GUARD", "")
    if value in ("", "0"):
        return None
    if value not in MODES:
        raise ValueError(f"QWEN_RUNAWAY_GUARD={value!r}: expected 'buffer', 'stream', or unset/empty/'0' for off")
    return value


def _env_int(name, default):
    value = os.environ.get(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name}={value!r}: expected an integer") from None


def guard_from_env(port, out_dir=None, name="runaway-guard.jsonl", **kw):
    """Unstarted GuardProxy configured from QWEN_RUNAWAY_* variables (see guarded_port); None when the guard is off."""
    mode = mode_from_env()
    if mode is None:
        return None
    det = {}
    if os.environ.get("QWEN_RUNAWAY_MIN_CHARS"):
        det["min_chars"] = _env_int("QWEN_RUNAWAY_MIN_CHARS", MIN_CHARS)
    inject = os.environ.get("QWEN_RUNAWAY_INJECT_SEED", "")
    if inject not in ("", "0", "1"):
        raise ValueError(f"QWEN_RUNAWAY_INJECT_SEED={inject!r}: expected '1' or unset/'0'")
    log = str(os.path.join(out_dir, name)) if out_dir else None
    return GuardProxy(port, mode=mode, retries=_env_int("QWEN_RUNAWAY_RETRIES", 1), log=log, detector=det,
                      inject_seed=inject == "1", **kw)


def guarded_port(port, out_dir=None, **kw):
    """Context manager for runners: yields the port to dial. Yields `port` unchanged (and starts nothing) while
    QWEN_RUNAWAY_GUARD is unset, empty or "0"; "buffer" | "stream" enable it; anything else raises.
    Also read: QWEN_RUNAWAY_MIN_CHARS, QWEN_RUNAWAY_RETRIES, QWEN_RUNAWAY_INJECT_SEED (1 = retries carry a fresh seed)."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        proxy = guard_from_env(port, out_dir, **kw)  # raises on a bad QWEN_RUNAWAY_GUARD before anything starts
        if proxy is None:
            yield port
            return
        proxy.start()
        try:
            yield proxy.port
        finally:
            proxy.stop()

    return cm()


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scripts.instruments.runaway_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    det = argparse.ArgumentParser(add_help=False)
    det.add_argument("--p-max", type=int, default=P_MAX)
    det.add_argument("--min-repeats", type=int, default=MIN_REPEATS)
    det.add_argument("--min-chars", type=int, default=MIN_CHARS)
    px = sub.add_parser("proxy", parents=[det], help="OpenAI-compatible streaming guard proxy")
    px.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:8080")
    px.add_argument("--port", type=int, required=True)
    px.add_argument("--host", default="127.0.0.1")
    px.add_argument("--log", help="JSONL event log (append)")
    px.add_argument("--mode", choices=MODES, default="buffer",
                    help="buffer: hold each response, abort + transparent resample; "
                         "stream: forward live, on a loop abort and end with a clean error event")
    px.add_argument("--retries", type=int, default=1)
    px.add_argument("--keepalive", type=float, default=10.0, help="SSE comment interval while buffering (s)")
    px.add_argument("--client-seed", choices=("error", "bump"), default="error",
                    help="client-set seed: refuse the retry (error) or replace the seed (bump)")
    px.add_argument("--inject-seed", action="store_true",
                    help="retries carry a fresh random seed (re-serialised JSON); default: re-send the original bytes")
    px.add_argument("--max-held", type=int, default=MAX_HELD,
                    help="buffer mode: bytes held per attempt before this request switches to live forwarding")
    px.add_argument("--max-event", type=int, default=MAX_EVENT,
                    help="bytes of one unterminated SSE event before guarding is dropped for the response")
    sc = sub.add_parser("scan", parents=[det], help="scan text files offline; exit 1 if any loop is found")
    sc.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    params = dict(p_max=a.p_max, min_repeats=a.min_repeats, min_chars=a.min_chars)
    if a.cmd == "scan":
        rc = 0
        for f in a.files:
            import gzip
            opener = gzip.open if f.endswith(".gz") else open
            with opener(f, "rt", errors="replace") as stream:
                hit = scan(stream.read(), **params)
            print(f, json.dumps({k: v for k, v in hit.items() if k != "unit"}) if hit else "clean")
            rc |= bool(hit)
        return rc
    proxy = GuardProxy(a.upstream, a.port, a.host, a.log, a.mode, a.retries, a.keepalive, a.client_seed, params,
                       inject_seed=a.inject_seed, max_held=a.max_held, max_event=a.max_event)
    print(f"runaway-guard {a.mode} on {a.host}:{proxy.port} -> {a.upstream}", file=sys.stderr, flush=True)
    try:
        proxy.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
