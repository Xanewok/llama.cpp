"""Per-task usage relay, copied from two-agent/scripts/run_two_agent.py (UsageProxy and parsers).

Copied, not imported: that file is pinned by queued gpuq items. Differences from the original:
the bind address is a parameter (the relay must listen on docker0, never 0.0.0.0) and the
upstream host is a parameter. Forwarding stays byte-for-byte: no sampling field is inserted.
"""

from __future__ import annotations

import http.client
import http.server
import json
import math
from pathlib import Path
import socket
import socketserver
import threading
import time

TERMINAL = {"response.completed", "response.incomplete", "response.failed"}
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-connection", "content-length", "accept-encoding"}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def scan_response(event, call, stamp=None):
    if not isinstance(event, dict):
        return
    kind = event.get("type", "")
    kind = kind if isinstance(kind, str) else ""
    if kind in TERMINAL:
        call["event"] = kind
    response = event.get("response") or event
    if not isinstance(response, dict):
        return
    details = response.get("incomplete_details")
    if kind == "response.incomplete" and isinstance(details, dict) and isinstance(details.get("reason"), str):
        call["incomplete_reason"] = details["reason"]
    usage = response.get("usage") or {}
    if isinstance(usage, dict):
        for target, keys in {
            "input_tokens": ("input_tokens", "prompt_tokens"),
            "output_tokens": ("output_tokens", "completion_tokens"),
        }.items():
            for key in keys:
                if number(usage.get(key)):
                    call[target] = usage[key]
                    break
        details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            details = {}
        if number(details.get("cached_tokens")):
            call["cached_tokens"] = details["cached_tokens"]
    timing = event.get("timings") or response.get("timings")
    if isinstance(timing, dict):
        # Stream timings are cumulative; nulls and empty final rows cannot erase evidence.
        saved = call.setdefault("timings", {})
        saved.update({key: value for key, value in timing.items() if number(value)})
    choices = event.get("choices")
    choices = choices if isinstance(choices, list) else []
    bearing = kind.endswith(".delta") and bool(event.get("delta"))
    for choice in choices:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict) and any(value for key, value in delta.items() if key != "role"):
            bearing = True
    if bearing and stamp is not None:
        call.setdefault("first_output", stamp)
        call["last_output"] = stamp


def normalize_call(call):
    row = dict(call)
    t = row.get("timings") or {}
    row["prefill_seconds"] = t["prompt_ms"] / 1000 if number(t.get("prompt_ms")) else None
    row["decode_seconds"] = t["predicted_ms"] / 1000 if number(t.get("predicted_ms")) else None
    row["decode_tps"] = (t["predicted_n"] / row["decode_seconds"]
                         if number(t.get("predicted_n")) and row["decode_seconds"] else None)
    first, last = row.get("first_output"), row.get("last_output")
    row["prompt_wait_seconds"] = first - row["started"] if number(first) else None
    duration = last - first if number(first) and number(last) else None
    row["delivery_decode_seconds"] = duration
    row["delivery_decode_tps"] = ((row["output_tokens"] - 1) / duration
                                  if duration and number(row.get("output_tokens")) and row["output_tokens"] > 0
                                  else None)
    drafted, accepted = t.get("draft_n"), t.get("draft_n_accepted")
    row["draft_acceptance"] = accepted / drafted if drafted and number(accepted) and accepted <= drafted else None
    row["uncached_tokens"] = (row["input_tokens"] - row["cached_tokens"]
                              if number(row.get("input_tokens")) and number(row.get("cached_tokens")) else None)
    row["cache_hit_fraction"] = (row["cached_tokens"] / row["input_tokens"]
                                 if row.get("input_tokens") and number(row.get("cached_tokens")) else None)
    row["delivery_rate_scope"] = "client estimate: (output_tokens - 1)/(last_delta - first_delta); first chunk may contain several tokens"
    row["runaway"] = number(row.get("output_tokens")) and row["output_tokens"] >= 32768
    return row


def response_call(call):
    return call.get("method") == "POST" and call.get("path", "").split("?")[0].endswith("/responses")


class UsageProxy(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, upstream, out, agent, task, deadline, bind="127.0.0.1", upstream_host="127.0.0.1", port=0,
                 cap_s=None):
        self.upstream, self.out, self.upstream_host = upstream, out, upstream_host
        self.agent, self.task, self.deadline = agent, task, deadline
        self.cap_s, self.cap_epoch = cap_s, None
        self.calls, self.active = [], set()
        self.lock = threading.Lock()
        self.sequence = 0
        super().__init__((bind, port), ProxyHandler)
        self.port = self.server_address[1]
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1})

    def start(self):
        self.thread.start()

    def handle_error(self, request, client_address):
        # A client resetting mid-handshake (Codex's WebSocket probe) is routine, not a harness fault.
        pass

    def allocate(self):
        with self.lock:
            self.sequence += 1
            return self.sequence

    def past_cap(self, epoch):
        """True for a model call starting after the agent cap.

        Harbor's cap kills only the host-side `docker compose exec` client, so Codex keeps calling the model.
        The agent's start is unknown here; the first model call (about 1 s after it) anchors the cap, so a
        rejection can be at most that late. run_swe_fresh rescores against Harbor's exact `agent_execution.started_at`.
        """
        with self.lock:
            if self.cap_s is None:
                return False
            if self.cap_epoch is None:
                self.cap_epoch = epoch + self.cap_s
                return False
            return epoch > self.cap_epoch

    def record(self, call):
        row = normalize_call(call)
        with self.lock:
            self.calls.append(row)
            with (self.out / "model-calls.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()

    def stop(self):
        self.shutdown()
        self.server_close()
        self.thread.join()
        with self.lock:
            for connection in self.active:
                # close() alone can leave the response's makefile holding the socket open.
                if connection.sock:
                    try:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                connection.close()
        # Request handlers are bounded by the same deadline as the agent.
        until = time.monotonic() + 5
        while time.monotonic() < until:
            with self.lock:
                if not self.active:
                    break
            time.sleep(0.01)
        with self.lock:
            return list(self.calls)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def body(self):
        if self.headers.get("Transfer-Encoding", "").lower() != "chunked":
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))
        chunks = []
        while True:
            size = int(self.rfile.readline().split(b";")[0], 16)
            if not size:
                while self.rfile.readline().strip():
                    pass
                return b"".join(chunks)
            chunks.append(self.rfile.read(size))
            self.rfile.read(2)

    def relay(self):
        sequence = self.server.allocate()
        call = {"agent": self.server.agent, "task": self.server.task, "call": sequence,
                "started": time.monotonic(), "started_epoch": time.time(),
                "method": self.command, "path": self.path}
        prefix = self.server.out / f"call-{sequence:04d}"
        if response_call(call) and self.server.past_cap(call["started_epoch"]):
            return self.reject_post_cap(call, prefix)
        connection = None
        raw = prefix.with_suffix(".response").open("wb")
        try:
            body = self.body()
            prefix.with_suffix(".request").write_bytes(body)
            remaining = self.server.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("budget")
            connection = http.client.HTTPConnection(self.server.upstream_host, self.server.upstream, timeout=remaining)
            with self.server.lock:
                self.server.active.add(connection)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            connection.request(self.command, self.path, body or None, headers)
            response = connection.getresponse()
            call["status"] = response.status
            write_json(prefix.with_suffix(".headers.json"), dict(response.getheaders()))
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            pending = b""
            json_body = "json" in (response.getheader("Content-Type") or "")
            while True:
                remaining = self.server.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("budget")
                if connection.sock:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(65536)
                if not chunk:
                    break
                raw.write(chunk)
                raw.flush()
                if time.monotonic() >= self.server.deadline:
                    raise TimeoutError("budget")
                if not json_body:
                    *lines, pending = (pending + chunk).split(b"\n")
                    for line in lines:
                        self.scan_line(line, call)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            if json_body:
                raw.flush()
                try:
                    scan_response(json.loads(prefix.with_suffix(".response").read_bytes()), call)
                except ValueError:
                    call["error"] = "invalid_json_body"
            else:
                self.scan_line(pending, call)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception as error:
            call["error"] = f"{type(error).__name__}: {error}"
            self.close_connection = True
        finally:
            raw.close()
            if connection:
                connection.close()
            call["finished"] = time.monotonic()
            call["finished_epoch"] = time.time()
            self.server.record(call)
            with self.server.lock:
                self.server.active.discard(connection)

    def reject_post_cap(self, call, prefix):
        """403 without reaching the model: no GPU time, no tool call, for a call the cap already ended."""
        try:
            self.body()
            payload = json.dumps({"error": {"type": "post_cap", "message": "agent time cap reached"}}).encode()
            prefix.with_suffix(".response").write_bytes(payload)
            self.send_response(403, "Forbidden")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except Exception as error:
            call["error"] = f"{type(error).__name__}: {error}"
        finally:
            self.close_connection = True
            call.update(status=403, post_cap=True, finished=time.monotonic(), finished_epoch=time.time())
            call.setdefault("error", "PostCapRejected: call started after the agent cap")
            self.server.record(call)

    @staticmethod
    def scan_line(line, call):
        if line.startswith(b"data:"):
            try:
                scan_response(json.loads(line[5:]), call, time.monotonic())
            except ValueError:
                pass

    do_GET = do_POST = do_DELETE = relay
