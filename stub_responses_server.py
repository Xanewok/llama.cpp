"""Canned OpenAI Responses server for CPU-only harness proofs. Not an inference server.

Turn 1 (no function_call_output in `input`): one exec tool call. Later turns: a final message.
Every request body is saved so the proof can show which model/tools/headers Codex sent.
"""

from __future__ import annotations

import argparse
import http.server
import json
from pathlib import Path
import socketserver
import threading
import time

EXEC_TOOLS = ("exec_command", "shell", "local_shell", "shell_command")
COMMAND = "echo stub-tool-ran > /tmp/stub-marker; pwd; ls | head -5"


def tool_names(body):
    names = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        names.append(tool.get("name") or tool.get("type"))
        for inner in tool.get("tools") or []:
            if isinstance(inner, dict):
                names.append(inner.get("name"))
    return [n for n in names if n]


def pick_tool(body):
    names = tool_names(body)
    for wanted in EXEC_TOOLS:
        if wanted in names:
            return wanted
    return None


def arguments_for(name):
    if name == "exec_command":
        return {"cmd": COMMAND, "yield_time_ms": 2000}
    return {"command": ["bash", "-lc", COMMAND]}


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def usage(input_tokens, output_tokens):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}}


def response_events(body, turn, with_timings):
    rid = f"resp_stub_{turn}"
    base = {"id": rid, "object": "response", "status": "in_progress", "model": body.get("model", "stub"),
            "output": []}
    events = [("response.created", {"type": "response.created", "response": base}),
              ("response.in_progress", {"type": "response.in_progress", "response": base})]
    has_result = any(isinstance(i, dict) and i.get("type", "").endswith("_output")
                     for i in body.get("input") or [] if isinstance(i, dict))
    tool = None if has_result else pick_tool(body)
    if tool:
        args = json.dumps(arguments_for(tool))
        item = {"type": "function_call", "id": f"fc_{turn}", "call_id": f"call_stub_{turn}",
                "name": tool, "arguments": "", "status": "in_progress"}
        events.append(("response.output_item.added", {"type": "response.output_item.added",
                                                      "output_index": 0, "item": item}))
        events.append(("response.function_call_arguments.delta",
                       {"type": "response.function_call_arguments.delta", "item_id": item["id"],
                        "output_index": 0, "delta": args}))
        done = {**item, "arguments": args, "status": "completed"}
        events.append(("response.output_item.done", {"type": "response.output_item.done",
                                                     "output_index": 0, "item": done}))
        output, out_tokens = [done], 24
    else:
        text = "STUB DONE: no fix attempted."
        item = {"type": "message", "id": f"msg_{turn}", "role": "assistant", "status": "in_progress",
                "content": []}
        events.append(("response.output_item.added", {"type": "response.output_item.added",
                                                      "output_index": 0, "item": item}))
        part = {"type": "output_text", "text": "", "annotations": []}
        events.append(("response.content_part.added", {"type": "response.content_part.added",
                       "item_id": item["id"], "output_index": 0, "content_index": 0, "part": part}))
        for word in text.split(" "):
            events.append(("response.output_text.delta", {"type": "response.output_text.delta",
                           "item_id": item["id"], "output_index": 0, "content_index": 0,
                           "delta": word + " "}))
        done = {**item, "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        events.append(("response.output_item.done", {"type": "response.output_item.done",
                                                     "output_index": 0, "item": done}))
        output, out_tokens = [done], 12
    in_tokens = 6000 + 1500 * turn
    final = {**base, "status": "completed", "output": output, "usage": usage(in_tokens, out_tokens)}
    completed = {"type": "response.completed", "response": final}
    if with_timings:
        completed["timings"] = {"prompt_n": in_tokens, "prompt_ms": 2000.0, "predicted_n": out_tokens,
                                "predicted_ms": 400.0, "draft_n": 20, "draft_n_accepted": 13}
    events.append(("response.completed", completed))
    return events


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            t = self.server.totals
            body = "\n".join([
                "# HELP vllm counters mimic HQ naming",
                f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {t["draft"]}',
                f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {t["accepted"]}',
                f'vllm:generation_tokens_total{{engine="0"}} {t["tokens"]}',
                f'vllm:request_decode_time_seconds_sum{{engine="0"}} {t["decode_s"]}', ""]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        with self.server.lock, (self.server.out / "get-log.jsonl").open("a") as stream:
            stream.write(json.dumps({"path": self.path, "headers": dict(self.headers)}) + "\n")
        if self.path.rstrip("/").endswith("/models"):
            self.reply({"object": "list", "data": [{"id": "stub", "object": "model"}]})
        else:
            self.reply({"error": "not found"}, 404)

    def reply(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        server = self.server
        with server.lock:
            server.turn += 1
            turn = server.turn
        if not self.path.split("?")[0].endswith("/responses"):
            self.reply({"error": "stub serves /responses only"}, 404)
            return
        body = json.loads(raw or b"{}")
        (server.out / f"request-{turn:03d}.json").write_text(json.dumps(
            {"path": self.path, "headers": dict(self.headers), "tools": tool_names(body),
             "model": body.get("model"), "stream": body.get("stream"), "body": body}, indent=1))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        with server.lock:
            server.totals["draft"] += 20
            server.totals["accepted"] += 13
            server.totals["tokens"] += 24 if pick_tool(body) and turn == 1 else 12
            server.totals["decode_s"] += 0.4
        for event, data in response_events(body, turn, server.timings):
            chunk = sse(event, data)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timings", action="store_true", help="emit llama-style draft timings")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    server = Server(("127.0.0.1", args.port), Handler)
    server.out, server.timings, server.turn, server.lock = args.out, args.timings, 0, threading.Lock()
    server.totals = {"draft": 0, "accepted": 0, "tokens": 0, "decode_s": 0.0}
    server.serve_forever()


if __name__ == "__main__":
    main()
