"""Campaign with agents > 1: real Codex agents share one server over the campaign's task set from a dynamic queue.

The same campaign layout, orchestration, caps and exit codes as campaign.py (this module is its agents > 1 branch): one
fresh server per campaign, launched here from the arm's contract instead of by the hillclimb runner. Each agent runs one task
at a time behind its own usage proxy; a free agent takes the next queued task. Time to solution is the queue wall (boot
excluded), delivery tok/s is all agents' output tokens over that wall. Campaigns rank by solved count, then time to solution.
An unsolved task is data: it keeps the campaign eligible and its time to solution. A call that failed and that the agent then
retried successfully is counted (retried_calls, per agent) and does not invalidate; a failed call never retried, no completed
task, or no counter scrape is insufficient data (rc 5), never integrity. An agent that errors, or a cap, leaves a partial
campaign (rc 5), never a result.

experiment.toml, [campaign] (the keys this module reads, beside the campaign.py ones):
  agents          concurrent agents = server slots (default 1; > 1 selects this module).
  agent_ctx       per-agent context; the server -np must equal agents and -c agents x agent_ctx ({np} {ctx} expand in
                  server.args and are checked against the launch, as are the live /props slots and n_ctx).
  task_timeout_s  wall limit of one task (default 1800).
  min_overlap_s   decode overlap between the agents (seconds both were streaming) at or below this is insufficient data (default 0).
  runaway_tokens  a single response this long marks a runaway task (as campaign.py).
Not used here: runner, fit_task (no hillclimb runner), arms' runroot.
"""
import hashlib
import http.client
import http.server
import json
import math
import os
import queue
import signal
import socket
import socketserver
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import campaign, lib
from .lib import Fail

TERMINAL = {"response.completed", "response.incomplete", "response.failed"}
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection", "content-length", "accept-encoding"}
# What a failed gate says about the data: a counter that cannot be true is integrity, an absent one is insufficient.
GATE_CLASS = {"no_agent_overlap": "insufficient_data", "mtp_not_live_or_missing_counters": "integrity", "invalid_acceptance_counters": "integrity",
              "server_output_counter_mismatch": "integrity", "call_failed": "insufficient_data", "missing_call_usage": "insufficient_data",
              "missing_stream_timing": "insufficient_data", "invalid_stream_timing": "integrity", "invalid_cache_usage": "integrity",
              "missing_task_calls": "insufficient_data", "no_completed_task": "insufficient_data", "missing_counter_scrape": "insufficient_data"}


def n_agents(x):
    return x["campaign"].get("agents", 1)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


# ---------------------------------------------------------------- per-call usage and timing (one proxy per task)

def scan_response(event, call, stamp=None):
    """Fold one SSE event or JSON body into the call: terminal event, usage, cumulative stream timings, first/last output time."""
    if not isinstance(event, dict):
        return
    kind = event.get("type", "")
    kind = kind if isinstance(kind, str) else ""
    if kind in TERMINAL:
        call["event"] = kind
    response = event.get("response") or event
    if not isinstance(response, dict):
        return
    usage = response.get("usage") or {}
    if isinstance(usage, dict):
        for target, keys in {"input_tokens": ("input_tokens", "prompt_tokens"), "output_tokens": ("output_tokens", "completion_tokens")}.items():
            for key in keys:
                if number(usage.get(key)):
                    call[target] = usage[key]
                    break
        details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict) and number(details.get("cached_tokens")):
            call["cached_tokens"] = details["cached_tokens"]
    timing = event.get("timings") or response.get("timings")
    if isinstance(timing, dict):  # cumulative: nulls and empty final rows cannot erase evidence
        call.setdefault("timings", {}).update({k: v for k, v in timing.items() if number(v)})
    bearing = kind.endswith(".delta") and bool(event.get("delta"))
    for choice in event.get("choices") if isinstance(event.get("choices"), list) else []:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict) and any(v for k, v in delta.items() if k != "role"):
            bearing = True
    if bearing and stamp is not None:
        call.setdefault("first_output", stamp)
        call["last_output"] = stamp


def normalize_call(call, runaway_tokens=32768):
    row = dict(call)
    t = row.get("timings") or {}
    first, last = row.get("first_output"), row.get("last_output")
    duration = last - first if number(first) and number(last) else None
    row["prompt_wait_seconds"] = first - row["started"] if number(first) else None
    row["delivery_decode_seconds"] = duration
    row["delivery_decode_tps"] = (row["output_tokens"] - 1) / duration if duration and number(row.get("output_tokens")) and row["output_tokens"] > 0 else None
    drafted, accepted = t.get("draft_n"), t.get("draft_n_accepted")
    row["draft_acceptance"] = accepted / drafted if drafted and number(accepted) and accepted <= drafted else None
    row["uncached_tokens"] = row["input_tokens"] - row["cached_tokens"] if number(row.get("input_tokens")) and number(row.get("cached_tokens")) else None
    row["runaway"] = number(row.get("output_tokens")) and row["output_tokens"] >= runaway_tokens
    return row


def response_call(call):
    return call.get("method") == "POST" and call.get("path", "").split("?")[0].endswith("/responses")


def summarize_calls(calls, runaway_tokens):
    rows = [normalize_call(c, runaway_tokens) for c in calls if response_call(c)]
    drafted = sum((r.get("timings") or {}).get("draft_n", 0) for r in rows)
    accepted = sum((r.get("timings") or {}).get("draft_n_accepted", 0) for r in rows)
    complete = rows and all(number((r.get("timings") or {}).get("draft_n")) and number((r.get("timings") or {}).get("draft_n_accepted")) for r in rows)
    return {"calls": rows, "responses": len(rows), "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in rows), "cached_tokens": sum(r.get("cached_tokens") or 0 for r in rows),
            "uncached_tokens": sum(r.get("uncached_tokens") or 0 for r in rows),
            "peak_context_tokens": max(((r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) for r in rows), default=0),
            "draft_acceptance": accepted / drafted if drafted and complete else None,
            "max_response": max((r.get("output_tokens") or 0 for r in rows), default=0), "runaway_calls": [r["call"] for r in rows if r["runaway"]]}


class UsageProxy(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Relays one agent's model traffic to the server and records every call (usage, timings, raw bytes) in out/."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, upstream, out, agent, task, deadline):
        self.upstream, self.out, self.agent, self.task, self.deadline = upstream, out, agent, task, deadline
        self.calls, self.active, self.lock, self.sequence = [], set(), threading.Lock(), 0
        super().__init__(("127.0.0.1", 0), ProxyHandler)
        self.port = self.server_address[1]
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1})

    def start(self):
        self.thread.start()

    def allocate(self):
        with self.lock:
            self.sequence += 1
            return self.sequence

    def record(self, call):
        with self.lock:
            self.calls.append(call)
            with (self.out / "model-calls.jsonl").open("a") as stream:
                stream.write(json.dumps(call, allow_nan=False) + "\n")

    def stop(self):
        self.shutdown()
        self.server_close()
        self.thread.join()
        with self.lock:
            for connection in self.active:
                if connection.sock:  # close() alone can leave the response's makefile holding the socket
                    try:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                connection.close()
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
        server = self.server
        call = {"agent": server.agent, "task": server.task, "call": server.allocate(), "started": time.monotonic(),
                "started_epoch": time.time(), "method": self.command, "path": self.path}
        prefix = server.out / f"call-{call['call']:04d}"
        connection, raw = None, prefix.with_suffix(".response").open("wb")
        try:
            body = self.body()
            prefix.with_suffix(".request").write_bytes(body)
            if server.deadline - time.monotonic() <= 0:
                raise TimeoutError("budget")
            connection = http.client.HTTPConnection("127.0.0.1", server.upstream, timeout=server.deadline - time.monotonic())
            with server.lock:
                server.active.add(connection)
            connection.request(self.command, self.path, body or None, {k: v for k, v in self.headers.items() if k.lower() not in HOP})
            response = connection.getresponse()
            call["status"] = response.status
            lib.write_json(prefix.with_suffix(".headers.json"), dict(response.getheaders()))
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            pending, json_body = b"", "json" in (response.getheader("Content-Type") or "")
            while True:
                remaining = server.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("budget")
                if connection.sock:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(65536)
                if not chunk:
                    break
                raw.write(chunk)
                raw.flush()
                if not json_body:
                    *lines, pending = (pending + chunk).split(b"\n")
                    for line in lines:
                        self.scan_line(line, call)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            if json_body:
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
            call["finished"], call["finished_epoch"] = time.monotonic(), time.time()
            server.record(call)
            with server.lock:
                server.active.discard(connection)

    @staticmethod
    def scan_line(line, call):
        if line.startswith(b"data:"):
            try:
                scan_response(json.loads(line[5:]), call, time.monotonic())
            except ValueError:
                pass

    do_GET = do_POST = do_DELETE = relay


# ---------------------------------------------------------------- queue and jobs

def overlap_seconds(intervals, agents):
    """Seconds in which every agent had an interval open. intervals: (agent, start, end)."""
    events = []
    for agent, start, end in intervals:
        if number(start) and number(end) and end > start:
            events += [(start, agent, 1), (end, agent, -1)]
    active, total, previous = [0] * agents, 0.0, None
    for stamp, agent, delta in sorted(events):
        if previous is not None and all(active):
            total += stamp - previous
        active[agent] += delta
        previous = stamp
    return total


def run_queue(tasks, agents, execute, deadline, cpu_sets=None):
    """The first `agents` tasks start together; a finishing agent takes the next queued task until the queue or the deadline ends."""
    if len(tasks) < agents or len(set(tasks)) != len(tasks):
        raise ValueError("need at least as many unique tasks as agents")
    pending = queue.Queue()
    for task in tasks[agents:]:
        pending.put(task)
    barrier = threading.Barrier(agents)

    def worker(agent):
        if cpu_sets:
            os.sched_setaffinity(0, cpu_sets[agent])
        rows, task = [], tasks[agent]
        barrier.wait()
        while time.monotonic() < deadline:
            rows.append(execute(agent, task))
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
        return rows

    with ThreadPoolExecutor(max_workers=agents) as pool:
        return [row for future in [pool.submit(worker, a) for a in range(agents)] for row in future.result()]


def cpu_sets(agents):
    cpus = sorted(os.sched_getaffinity(0))[:8]
    size = max(1, len(cpus) // agents)
    return [cpus[i * size:(i + 1) * size] or cpus[:size] for i in range(agents)]


def job_id(task):
    return "solve-" + task.replace("/", "-")


def run_job(x, agent, task, root, port, deadline, work, attempt):
    """One task by one agent: proxy up, `work` does it, the proxy's calls are folded into the job record and its job file."""
    out = root / f"agent-{agent}-{task.replace('/', '-')}"
    out.mkdir(parents=True)
    started = time.monotonic()
    row = {"agent": agent, "task": task, "started": started, "started_epoch": time.time(), "solved": False, "error": None,
           "timed_out": False, "tool_calls": 0}
    proxy = UsageProxy(port, out, agent, task, deadline)
    proxy.start()
    try:
        row |= work(agent, task, out, proxy, deadline)
    except Exception as error:
        row["error"], row["timed_out"] = f"{type(error).__name__}: {error}", time.monotonic() >= deadline
    finally:
        row |= summarize_calls(proxy.stop(), x["campaign"]["runaway_tokens"])
        row |= {"finished": time.monotonic(), "finished_epoch": time.time()}
        row["wall_seconds"] = row["finished"] - started
    passed = row["solved"] and not row["error"]
    row |= {"job": job_id(task), "attempt": attempt, "status": "passed" if passed else "failed",
            "reason": None if passed else "probe_incomplete" if row["error"] or row["timed_out"] else "probe_failed"}
    lib.write_json(out / "result.json", row)
    lib.write_json(root.parent / f"job-{job_id(task)}.json", row)
    return row


GRADE_TAIL = 8192


def tail_output(text):
    """The retry prompt carries the end of the grade output (the failures), not all of it: it must fit the agent's context."""
    if len(text) <= GRADE_TAIL:
        return text
    return f"[grade output truncated: last {GRADE_TAIL} of {len(text)} characters]\n" + text[-GRADE_TAIL:]


def codex_work(x, modules, assets):
    """The real task: Codex in the sandbox against the proxy, one grading-driven retry, trusted grading."""
    probe, emit, isolated, poly = modules
    c = x["campaign"]
    tasks = json.loads((assets / "tasks.json").read_text())
    checks = json.loads((assets / "selftest.json").read_text())
    tools = json.loads((assets / "tools.json").read_text())

    def work(agent, task_id, out, proxy, deadline):
        task = next(t for t in tasks if t["task_id"] == task_id)
        expected = next(t["expected_passed_tests"] for t in checks["tasks"] if t["task_id"] == task_id)
        workdir, home = out / "workspace", out / "codex-state"
        probe.materialize(task, assets, workdir)
        home.mkdir()
        (home / "config.toml").write_text(probe.client_config(c["agent_ctx"]))
        prompt = (poly._prompt(task) + " Do not modify tests. Dependencies are preinstalled; external network access is disabled. "
                  "The supported solution files are: " + ", ".join(task["solution_path"]) + ".")
        row = {"tool_calls": 0, "grading_rounds": [], "solved": False}
        # sun_path holds 107 bytes and attempt dirs exceed it; bwrap rebinds the socket to /bridge.
        with tempfile.TemporaryDirectory(prefix="qwen-bridge-") as bridge_dir, isolated.inference_bridge(Path(bridge_dir) / "model.sock", proxy.port):
            for attempt in range(2):
                (out / f"prompt-{attempt}.txt").write_text(prompt)
                transcript = out / f"codex-{attempt}.jsonl"
                result = isolated.run(workdir, tools, ["/runner/codex", "exec", "--skip-git-repo-check", "--json", "-"], poly._remaining(deadline),
                                      bridge=Path(bridge_dir) / "model.sock", client_home=home, readonly=task["test_paths"], input_text=prompt,
                                      stdout_path=transcript, stderr_path=out / f"codex-{attempt}.stderr")
                row["tool_calls"] += emit.parse_codex_stats(transcript.read_text()).tool_calls or 0
                if result.returncode:
                    raise RuntimeError(f"Codex exited with status {result.returncode}")
                grade_dir = out / f"grade-{attempt}"
                probe.materialize(task, assets, grade_dir)
                try:
                    ignored = probe.transfer_solution(task, workdir, grade_dir)
                    grade = probe.test_suite(task, grade_dir, tools, poly._remaining(deadline), asset_dir=assets)
                    grade["ignored_cargo_metadata_changes"] = ignored
                except (ValueError, FileNotFoundError) as error:
                    grade = {"returncode": 1, "passed": 0, "failed": 0, "ignored": 0, "output": str(error)}
                lib.write_json(out / f"grade-{attempt}.json", grade)
                row["grading_rounds"].append(grade)
                row["solved"] = grade["returncode"] == 0 and grade["passed"] == expected and grade["failed"] == grade["ignored"] == 0
                if row["solved"]:
                    return row
                prompt = (poly._prompt(task) + "\nTrusted grading failed. Repair the solution without changing tests.\n"
                          f"Expected {expected} trusted tests; got {grade['passed']} passed, {grade['failed']} failed, {grade['ignored']} ignored; "
                          f"exit {grade['returncode']}.\n" + tail_output(grade["output"]))
        return row

    return work


def stub_work():
    """CPU stand-in for an agent: a few Responses-API turns through the proxy to the stub server, growing context."""
    def work(agent, task_id, out, proxy, deadline):
        if os.environ.get("STUB_FAIL_TASK") == task_id:
            raise RuntimeError("Codex exited with status 1")
        mean, calls = campaign.PROFILE.get(task_id, (8000, 8))
        turns, scale = min(calls, int(os.environ.get("STUB_AGENT_CALLS", "2"))), float(os.environ.get("STUB_AGENT_SCALE", "0.02"))
        for i in range(turns):
            spec = {"input": 2000 * (i + 1), "cached": 2000 * i, "output": int(mean * scale / turns) + 1, "chunks": 20, "seed": 7 * agent + i + 1}
            for aborted in (True, False) if i == 0 and os.environ.get("STUB_RETRY_TASK") == task_id else (False,):  # a failed call, then its retry
                req = urllib.request.Request(f"http://127.0.0.1:{proxy.port}/v1/responses", json.dumps({"stub": spec | {"abort": aborted}, "stream": True}).encode(),
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as reply:
                    reply.read()
        time.sleep(float(os.environ.get("STUB_TASK_SLEEP", "0")))
        return {"solved": os.environ.get("STUB_UNSOLVED_TASK") != task_id, "tool_calls": turns - 1, "grading_rounds": []}
    return work


# ---------------------------------------------------------------- summary of one campaign

def gate(problem):
    return GATE_CLASS[problem.split(":")[0]]


def failed_call(c):
    return c.get("status") != 200 or bool(c.get("error")) or c.get("event") != "response.completed"


def retried_calls(job):
    """Numbers of a job's failed calls that a later call of the same job completed: the agent went on, so they are not gated."""
    calls = sorted(job.get("calls", []), key=lambda c: c["call"])
    return {c["call"] for i, c in enumerate(calls) if failed_call(c) and any(not failed_call(d) for d in calls[i + 1:])}


def campaign_summary(jobs, wall, delta, tasks, agents, min_overlap_s):
    """The validity gates and the numbers. Jobs that errored are partial data (their calls are not gated); a gate that fails
    makes the campaign invalid with the class of the worst gate. Retried failed calls are counted, not gated, and make the
    server's output counter a lower bound on the billed tokens (the failed call generated some). A never-retried failed call
    leaves usage short of the server's counters, so the counter gates wait for a campaign with none."""
    ok_jobs = [j for j in jobs if not j.get("error")]
    retried = {(j["agent"], j["task"]): retried_calls(j) for j in jobs}
    calls = [c for j in ok_jobs for c in j.get("calls", []) if c["call"] not in retried[j["agent"], j["task"]]]
    problems = []
    for c in calls:
        if failed_call(c):
            problems.append(f"call_failed: {c['agent']}/{c['task']}/{c['call']}")
        if any(not number(c.get(key)) for key in ("input_tokens", "output_tokens", "cached_tokens")):
            problems.append("missing_call_usage")
        elif c["cached_tokens"] > c["input_tokens"]:
            problems.append("invalid_cache_usage")
        if c.get("first_output") is None or c.get("last_output") is None:
            problems.append("missing_stream_timing")
        elif not c["started"] <= c["first_output"] <= c["last_output"] <= c["finished"]:
            problems.append("invalid_stream_timing")
    request_overlap = overlap_seconds(((c["agent"], c["started"], c["finished"]) for c in calls), agents)
    decode_overlap = overlap_seconds(((c["agent"], c.get("first_output"), c.get("last_output")) for c in calls), agents)
    if decode_overlap <= min_overlap_s:
        problems.append("no_agent_overlap")
    billed, n_retried = sum(j.get("output_tokens") or 0 for j in jobs), sum(map(len, retried.values()))
    if not ok_jobs:
        problems.append("no_completed_task")
    elif "decode_tokens" not in delta:
        problems.append("missing_counter_scrape")
    elif not any(p.startswith("call_failed") for p in problems):
        if not delta.get("draft_tokens") or not delta.get("draft_accepted_tokens"):
            problems.append("mtp_not_live_or_missing_counters")
        if delta.get("draft_accepted_tokens", 0) > delta.get("draft_tokens", 0):
            problems.append("invalid_acceptance_counters")
        if not (delta["decode_tokens"] >= billed - 2 if n_retried else math.isclose(delta["decode_tokens"], billed, abs_tol=2)):
            problems.append("server_output_counter_mismatch")
    for task in tasks:
        if any(j["task"] == task for j in ok_jobs) and not any(c.get("task") == task for c in calls):
            problems.append("missing_task_calls: " + task)
    per_agent = []
    for agent in range(agents):
        owned = [j for j in jobs if j["agent"] == agent]
        total = lambda key: sum(j.get(key) or 0 for j in owned)
        streaming = sum(c.get("delivery_decode_seconds") or 0 for j in owned for c in j.get("calls", []))
        per_agent.append({"agent": agent, "tasks": [j["task"] for j in owned], "solved": [j["task"] for j in owned if j["status"] == "passed"],
                          "errors": [{"task": j["task"], "error": j["error"]} for j in owned if j["error"]],
                          "unsolved": [j["task"] for j in owned if j["status"] != "passed" and not j["error"]],
                          "wall_seconds": total("wall_seconds"), "responses": total("responses"), "input_tokens": total("input_tokens"),
                          "output_tokens": total("output_tokens"), "cached_tokens": total("cached_tokens"), "prefill_tokens": total("uncached_tokens"),
                          "delivery_tps": total("output_tokens") / streaming if streaming else None,
                          "runaway_tasks": [j["task"] for j in owned if j.get("runaway_calls")],
                          "retried_calls": sum(len(retried[j["agent"], j["task"]]) for j in owned)})
    finished = len(jobs) == len(tasks) and not any(j["error"] for j in jobs)
    output = sum(a["output_tokens"] for a in per_agent)
    return {"gates": problems, "gate_class": lib.worst_class(gate(p) for p in problems) if problems else None,
            "tasks": len(jobs), "solved": sum(j["status"] == "passed" for j in jobs), "wall_seconds": wall,
            "tasks_planned": len(tasks), "retried_calls": n_retried,
            "time_to_solution_seconds": wall if finished and not problems else None, "output_tokens": output,
            "input_tokens": sum(a["input_tokens"] for a in per_agent), "cached_tokens": sum(a["cached_tokens"] for a in per_agent),
            "prefill_tokens": sum(a["prefill_tokens"] for a in per_agent),
            "aggregate_delivery_tps": output / wall if wall and not problems else None,
            "agent_overlap_seconds": overlap_seconds(((j["agent"], j["started"], j["finished"]) for j in jobs), agents),
            "request_overlap_seconds": request_overlap, "decode_delivery_overlap_seconds": decode_overlap,
            "decode_overlap_fraction": decode_overlap / wall if wall else None, "metrics_scope": "exclusive_server_whole_campaign",
            "metrics_delta": delta, "draft_acceptance": delta["draft_accepted_tokens"] / delta["draft_tokens"] if delta.get("draft_tokens") else None,
            "agents": per_agent}


# ---------------------------------------------------------------- the child: boot, run, seal

def launch_record(x, arm, port, stub, live=False):
    """argv, executed argv and scrubbed env exactly as the child will launch them, verified against the contract. live: the
    caller is the launcher, so its nice is the server's (CPU checks run niced and prove the rest)."""
    command = [str(campaign.runner_binary(arm)), *campaign.contract(x, arm, port)]
    executed = [str(campaign.STUB) if stub else str(arm["exe"]), *command[1:]]
    if stub:
        env = campaign.stub_env(x, arm)
    else:
        env = lib.server_env(dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(x["devices"]), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8"), arm, campaign.knobs(x))
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    launch = {"command": command, "executed_command": executed, "executed_env": lib.env_record(env, campaign.knobs(x)), "server_nice": nice,
              **({"stub_exe": str(arm["exe"])} if stub else {})}
    campaign.verify_launch(x, arm, launch, stub=stub)
    lib.verify_env(launch["executed_env"], arm["env"], campaign.knobs(x))
    if live and not stub:
        lib.verify_nice(nice)
    return launch, env


def child(x, arm, out, deadline_epoch, tasks, port, stub):
    """Boot the arm, run the queue, seal the campaign directory (manifest, attempt, job files, SUMMARY, state)."""
    sys.path.insert(0, str(campaign.EVAL))
    from scripts.instruments import emit
    c, agents = x["campaign"], n_agents(x)
    k, p_min, split, ctx, slots, mode = campaign.template_config(x, arm)
    deadline = time.monotonic() + (deadline_epoch - time.time())
    attempt_name = hashlib.sha256(f"agents|{arm['name']}".encode()).hexdigest()[:16] + "-01"
    attempt, assets = out / "attempts" / attempt_name, Path(c["assets"])
    attempt.mkdir(parents=True)
    launch, env = launch_record(x, arm, port, stub, live=True)
    binary = campaign.runner_binary(arm)
    pins = {str(binary.resolve()): "stub" if stub else lib.file_sha(binary)}
    queue_ = [{"id": job_id(t), "kind": "profile", "task": t, "workload": "agentic", "targets": [0],
               "configs": [{"context": ctx, "slots": slots, "mode": mode, "split": split, "ubatch": 128, "maximum": k, "p_min": p_min}]} for t in tasks]
    lib.write_json(out / "manifest.json", {"engine_commit": arm["commit"], "binary": str(binary), "binary_pins": pins, "queue": queue_,
                                           "server_extra": campaign.extra(x, arm), "port": port, "agents": agents, "stub": stub})
    lib.write_json(attempt / "launch.json", launch)
    server = lib.Server(launch["executed_command"], env, attempt / "server.log")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    memlog = lib.MemLog(attempt / "memlog.jsonl", lambda: server.proc.pid if server.proc else None).start()
    try:
        server.start()
        base = f"http://127.0.0.1:{port}"
        server.wait_ready(base, x["server"].get("boot_cap_s", 420))
        if not stub:
            outside = lib.libs_outside(server.proc.pid, arm["root"])
            if outside:
                raise Fail("frozen bytes changed: loaded outside frozen root: " + ", ".join(outside), "integrity")
        props = lib.http_json(base + "/props")
        lib.write_json(attempt / "props.json", props)
        lib.verify_props(props, ctx // slots, slots)
        if stub:
            work = stub_work()
        else:
            from scripts.instruments import agentic_probe, isolated_exec, run_polyglot
            work = codex_work(x, (agentic_probe, emit, isolated_exec, run_polyglot), assets)
        before = emit.scrape_metrics(base)
        lib.write_json(out / "metrics-before.json", before)
        start = time.monotonic()
        timeout = c.get("task_timeout_s", 1800)
        jobs = run_queue(tasks, agents, lambda agent, task: run_job(x, agent, task, out / "tasks", port, min(deadline, time.monotonic() + timeout),
                                                                    work, f"attempts/{attempt_name}"), deadline, cpu_sets(agents))
        wall = time.monotonic() - start
        after = emit.scrape_metrics(base)
        lib.write_json(out / "metrics-after.json", after)
    finally:
        server.stop(grace=30)
        lib.write_json(attempt / "mem.json", memlog.stop())
    summary = campaign_summary(jobs, wall, emit.metrics_delta(before, after), tasks, agents, c.get("min_overlap_s", 0))
    capped = time.monotonic() >= deadline or len(jobs) < len(tasks) or any(j["timed_out"] for j in jobs)
    state = "stopped" if capped else "partial" if any(j["error"] for j in jobs) else "invalid" if summary["gates"] else "complete"
    summary |= {"state": state, "reason": "budget_expired" if capped else None, "valid": state == "complete"}
    lib.write_json(out / "SUMMARY.json", summary)
    lib.write_json(out / "state.json", {"status": state, "reason": summary["reason"]})
    return 0 if state == "complete" else 1


# ---------------------------------------------------------------- campaign row and rendering

METRICS = [("tts_s", "time to solution s", "{:.1f}"), ("agg_tok_s", "aggregate delivery tok/s", "{:.1f}"), ("wall_s", "queue wall s", "{:.1f}"),
           ("solved", "solved tasks", "{:.0f}"), ("output_tokens", "output tokens", "{:.0f}"), ("cached_tokens", "cached input tokens", "{:.0f}"),
           ("prefill_tokens", "prefill (uncached input) tokens", "{:.0f}"), ("overlap_s", "decode overlap s", "{:.1f}"),
           ("overlap_frac", "decode overlap / wall", "{:.3f}"), ("acceptance", "draft acceptance", "{:.3f}"), ("runaways", "runaway tasks", "{:.0f}"),
           ("retried_calls", "retried failed calls", "{:.0f}"),
           ("swap_out_mib", "host swap-out during boot MiB", "{:.0f}"), ("peak_swap_used_mib", "peak host swap in use MiB", "{:.0f}"),
           ("peak_vmswap_mib", "peak server VmSwap MiB", "{:.0f}"), ("peak_rss_mib", "peak server RSS MiB", "{:.0f}")]
MEM_KEYS = ("swap_out_mib", "swap_in_mib", "peak_swap_used_mib", "peak_zram_used_mib", "peak_vmswap_mib", "peak_rss_mib", "min_avail_mib")


def ranking(rows):
    """Eligible campaigns, best first: most tasks solved, then shortest time to solution."""
    return sorted((r for r in rows if r["eligible"]), key=lambda r: (-(r.get("solved") or 0), r["tts_s"] if r.get("tts_s") is not None else math.inf, r.get("idx") or 0))


def campaign_row(x, path, meta=None):
    """Never raises on a partial campaign or a failed job. Eligible = complete, every task ran, no agent error; unsolved tasks
    stay eligible (ranked by solved count, then time to solution)."""
    path, tasks = Path(path), x["campaign"]["tasks"]
    meta = meta or lib.read_json(path / "ab.json", {}) or {}
    summary, state = lib.read_json(path / "SUMMARY.json", {}) or {}, lib.read_json(path / "state.json", {}) or {}
    jobs, errors = [], []
    for file in sorted(path.glob("job-*.json")):
        job = lib.read_json(file)
        errors += [] if job is not None else [f"unreadable {file.name}"]
        jobs += [] if job is None else [job]
    passed = [j for j in jobs if j.get("status") == "passed"]
    runaways = [{"task": j["task"], "max_response": j.get("max_response")} for j in jobs if j.get("runaway_calls")]
    mem = next((lib.read_json(f, {}) or {} for f in sorted(path.glob("attempts/*/mem.json"))), {})
    row = {"path": str(path), **{k: meta.get(k) for k in ("idx", "arm", "k", "p_min", "elapsed_s", "capped", "rc", "note")},
           **{k: mem.get(k) for k in MEM_KEYS}, "state": summary.get("state") or state.get("status") or "missing", "jobs": len(jobs), "jobs_ok": len(passed),
           "failed": [{"task": j["task"], "reason": j.get("error") or j.get("reason"), "max_response": j.get("max_response")} for j in jobs if j.get("status") != "passed"],
           "runaway_tasks": runaways, "runaway_unknown": [], "runaways": len(runaways), "errors": errors,
           "tasks": [{k: j.get(k) for k in ("task", "agent", "status", "reason", "error", "wall_seconds", "output_tokens", "cached_tokens", "uncached_tokens", "max_response")} for j in jobs],
           "tts_s": summary.get("time_to_solution_seconds"), "wall_s": summary.get("wall_seconds"), "agg_tok_s": summary.get("aggregate_delivery_tps"),
           "solved": summary.get("solved"), "retried_calls": summary.get("retried_calls"), "output_tokens": summary.get("output_tokens"), "cached_tokens": summary.get("cached_tokens"),
           "prefill_tokens": summary.get("prefill_tokens"), "overlap_s": summary.get("decode_delivery_overlap_seconds"),
           "overlap_frac": summary.get("decode_overlap_fraction"), "acceptance": summary.get("draft_acceptance"), "agents": summary.get("agents"),
           "gates": summary.get("gates")}
    row["eligible"] = row["state"] == "complete" and len(jobs) == len(tasks) and not errors
    return row


def render(x, rows, plan=None, notes=()):
    c, arms = x["campaign"], sorted(x["arms"])
    f = campaign.fmt
    lines = [f"# {x['id']}: " + ", ".join(f"{a} = {x['arms'][a]['label']} (k{x['arms'][a]['k']} p_min {x['arms'][a]['p_min']})" for a in arms), ""]
    lines += list(notes) + [""] if notes else []
    lines += [f"Plan: {plan}", ""] if plan else []
    lines += [f"Sample unit: one {len(c['tasks'])}-task campaign, {c['agents']} Codex agents sharing one server (-np {c['agents']}), tasks from a dynamic queue. "
              "Time to solution = queue wall (boot excluded), defined when every task ran to its end, solved or not, and no gate failed; campaigns rank by solved count, then time to solution. Aggregate delivery tok/s = "
              "all agents' output tokens / wall. Decode overlap = seconds in which every agent was streaming a response. Prefill tokens = input minus "
              "cached input tokens, summed over calls. Values are per campaign, never pooled.", "",
              "| # | arm | state | solved | TTS s | agg tok/s | wall s | overlap s | overlap/wall | out tok | cached | prefill | accept | elapsed s |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        state = r["state"] + (" (capped)" if r.get("capped") else "") + ("" if r["eligible"] else " [excluded]")
        cells = [str(r.get("idx")), str(r.get("arm")), state, f"{f('{:.0f}', r.get('solved'))}/{len(c['tasks'])}"]
        cells += [f(spec, r.get(k)) for k, spec in (("tts_s", "{:.1f}"), ("agg_tok_s", "{:.1f}"), ("wall_s", "{:.1f}"), ("overlap_s", "{:.1f}"), ("overlap_frac", "{:.3f}"),
                                                    ("output_tokens", "{:.0f}"), ("cached_tokens", "{:.0f}"), ("prefill_tokens", "{:.0f}"), ("acceptance", "{:.3f}"),
                                                    ("elapsed_s", "{:.0f}"))]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Ranking (solved count, then time to solution; eligible campaigns): " +
              ("; ".join(f"#{r.get('idx')} {r.get('arm')} {f('{:.0f}', r.get('solved'))}/{len(c['tasks'])} solved, TTS {f('{:.1f}', r.get('tts_s'))} s" for r in ranking(rows)) or "none")]
    lines += ["", "## Per agent", "", "| # | agent | tasks | solved | errors | unsolved | retried calls | out tok | input tok | cached | prefill | delivery tok/s | wall s |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        for a in r.get("agents") or []:
            lines.append(f"| {r.get('idx')} | {a['agent']} | {len(a['tasks'])} | {len(a['solved'])} | {len(a['errors'])} | {len(a['unsolved'])} | {a.get('retried_calls', 0)} | {a['output_tokens']} | "
                         f"{a['input_tokens']} | {a['cached_tokens']} | {a['prefill_tokens']} | {f('{:.1f}', a['delivery_tps'])} | {a['wall_seconds']:.1f} |")
    lines += ["", "## Host memory per campaign (memlog every 2 s; swap/zram are host-wide, VmSwap/RSS the server's; swapping during a boot can contaminate timing)", "",
              "| # | arm | swap-out MiB | swap-in MiB | peak swap in use | peak zram | peak server VmSwap | peak server RSS | min MemAvailable |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r.get('idx')} | {r.get('arm')} | " + " | ".join(f("{:.0f}", r.get(k)) for k in MEM_KEYS) + " |")
    flagged = [r for r in rows if r["runaway_tasks"] or r["failed"] or r.get("gates")]
    if flagged:
        lines += ["", "Runaway, failed or gated:"]
        for r in flagged:
            lines += [f"- #{r.get('idx')} {r.get('arm')}: {t['task']} response of {t['max_response']} tokens" for t in r["runaway_tasks"]]
            lines += [f"- #{r.get('idx')} {r.get('arm')}: {t['task']} not passed ({t['reason']})" for t in r["failed"]]
            lines += [f"- #{r.get('idx')} {r.get('arm')}: gate {g}" for g in r.get("gates") or []]
    comparison = campaign.compare(x, rows, METRICS)
    if len(arms) > 1:
        lines += ["", f"## Arms side by side (campaign values in run order; delta = arm mean - {arms[0]} mean)", "",
                  "| metric | " + " | ".join(arms) + " | " + " | ".join(f"{a} - {arms[0]}" for a in arms[1:]) + " |", "|---|" + "---|" * (2 * len(arms) - 1)]
        for key, label, spec in METRICS:
            cells = [", ".join(f"#{i}: {f(spec, v)}" for i, v in comparison[key][a]["values"]) or "n/a" for a in arms]
            for a in arms[1:]:
                v = comparison[key][a].get("vs_base")
                cells.append("n/a" if not v else f"{v['delta']:+.4g}" + ("" if v["pct"] is None else f" ({v['pct']:+.1f}%)"))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CPU checks

def check_assets(assets, tasks):
    probe = sys.modules.get("scripts.instruments.agentic_probe")
    if probe is None:
        sys.path.insert(0, str(campaign.EVAL))
        from scripts.instruments import agentic_probe as probe
    checks = json.loads((assets / "selftest.json").read_text())
    if not checks.get("harness_verified") or checks.get("test_scope") != probe.TEST_SCOPE:
        raise Fail("inputs changed: trusted grader selftest missing", "integrity")
    for path, digest in json.loads((assets / "tools.json").read_text())["binary_hashes"].items():
        if lib.file_sha(path) != digest:
            raise Fail("inputs changed: tool binary changed: " + path, "integrity")
    missing = [t for t in tasks if not any(e["task_id"] == t for e in checks["tasks"])]
    if missing:
        raise Fail(f"inputs changed: selftest.json lacks {missing}", "integrity")


def cpu_checks(x, out, stub, capture_proof=True):
    c = x["campaign"]
    if not stub:
        for arm in x["arms"].values():
            lib.check_frozen(arm)
        lib.check_model(x["model"])
        if "draft_vocab" in x:
            lib.check_input(x["draft_vocab"], "draft vocab")
        lib.log(out, "frozen roots of every arm verified (sums, pinned hashes, commits)")
    assets = Path(c["assets"])
    known = {t["task_id"] for t in json.loads((assets / "tasks.json").read_text())}
    if not set(c["tasks"]) <= known:
        raise Fail(f"inputs changed: assets lack {set(c['tasks']) - known}", "integrity")
    if not stub:
        check_assets(assets, c["tasks"])
    if len(c["tasks"]) < n_agents(x):
        raise Fail(f"inputs changed: {len(c['tasks'])} tasks for {n_agents(x)} agents", "integrity")
    for arm in x["arms"].values():
        launch_record(x, arm, 0, stub)  # any port: the contract takes {port} as a wildcard
    lib.log(out, "launch contract (argv, env, -np/-c shape) ok for arms " + ",".join(x["arms"]))


def capture(x, arm, out):
    """CPU proof: seal the launch.json the child would use for this arm. No server, no GPU."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    launch, _ = launch_record(x, arm, 0, stub=False)
    lib.write_json(out / "launch.json", launch)
    print(out / "launch.json")
    return 0
