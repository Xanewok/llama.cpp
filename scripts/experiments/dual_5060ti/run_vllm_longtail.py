"""The eight-task Swift long tail against an already-running HyperQwen (patched vLLM) server.

Same assets, Codex client config, jail, grader and probe as run_swift_longtail.py; the probe runs
with a vLLM engine hook instead of the llama-server single-slot contract. One long-lived server
serves every task. Output layout (attempts/<key>-NN/, sealed job-*.json, state.json, SUMMARY.json,
REPORT.md) matches the llama campaigns so the same comparison scripts read it.

Deltas from the llama path, all recorded in manifest.json:
- engine: vLLM 0.30 (HyperQwen), W4A16 checkpoint, fp8 KV, MTP k=3, TP2 on both 5060 Ti.
- sampling: the server's --override-generation-config defaults, recorded per campaign (min_p needs
  patches/minp-spec-decode-v2-responses.patch; the llama path used min_p 0.05).
- prompt reuse: vLLM prefix caching (hybrid model, block-aligned) instead of llama-server's
  slot prompt cache. The cache is reset before every task so each task starts cold, like the
  llama path's fresh server.
- context peak: the largest input+output of any /v1/responses call, read by a proxy on the
  inference bridge; vLLM has no n_tokens_max.
"""

import argparse
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request

from scripts import ROOT
from scripts.instruments import agentic_probe
from scripts.instruments import hillclimb as h
from scripts.instruments import swift_longtail as s
from scripts.instruments.hillclimb_probe import api, digest_file

SERVED = "local-benchmark"
HYPERQWEN = Path.home() / "repos/HyperQwen"
MODEL_DIR = Path.home() / "models/hyperqwen/Swift-Qwen3.8-27B-W4A16-syv-fast"
TEMPLATE = ROOT / "corpora/templates/swift-leading-instructions-v1.jinja"
# Byte-identical to the chat-template.jinja the llama long-tail attempts launched with.
TEMPLATE_SHA256 = "4af3dcf0e98bf87adf684a4824683ab15fffd64e93ed1170061ef86a15827ba8"
REQUIRED_ARGS = [["--max-num-seqs", "1"], ["--served-model-name", SERVED], ["--enable-prefix-caching"],
                 ["--chat-template", str(TEMPLATE)], ["--speculative-config"]]
CAP_WATTS = 150
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
               "proxy-connection", "content-length", "accept-encoding"}
MODEL = {"id": "hyperqwen-swift-w4a16", "engine": "vllm", "local": str(MODEL_DIR)}
# Keyed by the served model directory, read from the live server's argv.
MODEL_IDS = {"Swift-Qwen3.8-27B-W4A16-syv-fast": "hyperqwen-swift-w4a16",
             "Swift-1.5-Qwen3.8-27B-W4A16-syv-fast": "hyperqwen-swift15-w4a16",
             "Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8": "hyperqwen-swift15-w4a16-g32i8",
             "Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh": "hyperqwen-swift15-w4a16-codexh"}


class UsageProxy(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP relay between the inference bridge and vLLM that records each response's usage.

    The ContextSampler interface: start(), stop() -> peak, plus the port the bridge dials.
    """

    daemon_threads = True

    def __init__(self, upstream, out):
        self.upstream, self.out, self.calls = upstream, out, []
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), ProxyHandler)
        self.port = self.server_address[1]
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)

    def start(self):
        self.thread.start()

    def record(self, call):
        with self.lock:
            self.calls.append(call)

    def stop(self):
        self.shutdown()
        self.server_close()
        if getattr(self, "guard", None):
            self.guard.stop()
        with self.lock:
            calls = list(self.calls)
        with (self.out / "model-calls.jsonl").open("w") as stream:
            for call in calls:
                stream.write(json.dumps(call) + "\n")
        totals = [c["input_tokens"] + c["output_tokens"] for c in calls if c.get("input_tokens") is not None]
        return max(totals, default=0)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def request_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            body = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0], 16)
                if not size:
                    self.rfile.readline()
                    return body
                body += self.rfile.read(size)
                self.rfile.readline()
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else None

    def relay(self):
        body = self.request_body()
        call = {"started": time.time(), "method": self.command, "path": self.path}
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        upstream = http.client.HTTPConnection("127.0.0.1", self.server.upstream, timeout=None)
        try:
            upstream.request(self.command, self.path, body, headers)
            response = upstream.getresponse()
            call["status"] = response.status
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            pending, whole = b"", []
            json_body = "json" in (response.getheader("Content-Type") or "")
            while chunk := response.read1(65536):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                if json_body:
                    whole.append(chunk)
                    continue
                *lines, pending = (pending + chunk).split(b"\n")
                for line in lines:
                    self.scan_event(line, call)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            if json_body and response.status < 400:
                self.scan_usage(json.loads(b"".join(whole) or b"{}"), call)
            elif json_body:
                call["error"] = b"".join(whole)[:2000].decode(errors="replace")
        except (BrokenPipeError, ConnectionResetError):
            # Closing upstream is what lets vLLM abort a generation the client abandoned.
            call["client_disconnected"] = True
            self.close_connection = True
        finally:
            upstream.close()
            call["finished"] = time.time()
            self.server.record(call)

    def scan_event(self, line, call):
        if not line.startswith(b"data:"):
            return
        try:
            event = json.loads(line[5:])
        except ValueError:
            return
        if isinstance(event, dict) and event.get("type") in ("response.completed", "response.incomplete",
                                                               "response.failed"):
            call["event"] = event["type"]
            self.scan_usage(event.get("response") or {}, call)

    @staticmethod
    def scan_usage(response, call):
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return
        call.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                    cached_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
                    reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"))

    do_GET = do_POST = do_DELETE = relay


class VllmEngine:
    peak_source = "bridge proxy: max input+output tokens of one /v1/responses call"

    def __init__(self, url):
        self.upstream = int(url.rsplit(":", 1)[-1])

    def check(self, url, context):
        models = api(url, "/v1/models")["data"]
        if [(m["id"], m.get("max_model_len")) for m in models] != [(SERVED, context)]:
            raise ValueError("agentic probe requires the one declared model at the declared context")

    def meter(self, out):
        guard = None
        if os.environ.get("QWEN_RUNAWAY_GUARD", "") not in ("", "0"):  # opt-in: bridge -> UsageProxy -> guard -> vLLM
            from scripts.instruments import runaway_guard
            guard = runaway_guard.guard_from_env(self.upstream, out).start()
        proxy = UsageProxy(guard.port if guard else self.upstream, out)
        proxy.guard = guard
        return proxy

    @staticmethod
    def context_peak(sampled, context):
        if not isinstance(sampled, int) or not 0 < sampled <= context:
            raise ValueError("invalid per-task context high-water mark")
        return sampled


def xid_count():
    log = subprocess.run(["journalctl", "-k", "-b", "--no-pager"], capture_output=True, text=True).stdout
    return sum(("Xid" in line or "fallen off" in line) for line in log.splitlines())


def gpu_state():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,power.limit,temperature.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    return [dict(zip(("index", "power_limit", "temperature", "memory_used_mib"), map(float, row.split(","))))
            for row in rows]


def server_process(port):
    """The `vllm serve` process on this port, from HyperQwen's serve.pid parent."""
    parent = int((HYPERQWEN / "serve.pid").read_text())
    for pid in [parent, *map(int, subprocess.check_output(["ps", "--ppid", str(parent), "-o", "pid="],
                                                          text=True).split())]:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
        argv = [arg.decode() for arg in argv]
        if "serve" in argv and argv[argv.index("--port") + 1] == str(port):
            environ = dict(item.decode().split("=", 1) for item in
                           Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in item)
            env = {k: v for k, v in environ.items() if k.startswith(("CUDA_", "NCCL_", "VLLM_", "PYTORCH_"))}
            return pid, argv, env
    raise SystemExit(f"no HyperQwen vllm serve process on port {port} under serve.pid {parent}")


def check_launch(argv):
    for required in REQUIRED_ARGS:
        if not any(argv[i:i + len(required)] == required for i in range(len(argv))):
            raise SystemExit("server launch lacks " + " ".join(required))
    if digest_file(TEMPLATE) != TEMPLATE_SHA256:
        raise SystemExit("chat template differs from the llama path's")


def wait_idle(url, limit=600):
    until = time.monotonic() + limit
    while time.monotonic() < until:
        with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
            text = response.read().decode()
        busy = [line for line in text.splitlines()
                if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting"))
                and float(line.split()[-1]) != 0]
        if not busy:
            return
        time.sleep(2)
    raise h.StopCampaign("server_not_idle")


def reset_prefix_cache(url):
    request = urllib.request.Request(url + "/reset_prefix_cache", b"", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise h.StopCampaign("prefix_cache_reset_failed")


def job_id(task):
    return "solve-" + MODEL["id"] + "-" + task.replace("/", "-")


def run_task(campaign, assets, task, url, context, timeout, launch, xid_baseline):
    config = {"engine": "vllm", "context": context, "slots": 1, "workload": "agentic"}
    spec = {"job": job_id(task), "model": MODEL["id"], "config": config, "target": 0, "task": task, "seed": None}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    result = None
    for number in range(1, 4):
        out = campaign / "attempts" / f"{key}-{number:02d}"
        if out.exists():
            continue
        for gpu in gpu_state():
            if abs(gpu["power_limit"] - CAP_WATTS) > 0.1 or gpu["temperature"] >= 80:
                raise h.StopCampaign(f"gpu_state: {gpu}")
        wait_idle(url)
        reset_prefix_cache(url)
        out.mkdir(parents=True)
        h.sealed_json(out / "spec.json", spec)
        h.sealed_json(out / "launch.json", launch)
        result = {**spec, "status": "failed", "attempt": str(out.relative_to(campaign)), "workload": "agentic"}
        began = time.monotonic()
        before = gpu_state()
        try:
            probe = agentic_probe.probe(assets, task, url, context, out, timeout, engine=VllmEngine(url))
            if not (probe["complete"] and probe["mtp_live"]):
                result["reason"] = "probe_incomplete"
            else:
                delta = probe["metrics_delta"]
                result.update(status="passed", probe=probe,
                              decode_tps=delta["decode_tokens"] / delta["decode_seconds"],
                              prefill_tps=delta["prefill_tokens"] / delta["prefill_seconds"])
        except Exception as error:
            result.update(status="error", reason=f"{type(error).__name__}: {error}")
        result.update(finished_at=time.time(), elapsed_seconds=time.monotonic() - began,
                      gpu_before=before, gpu_after=gpu_state())
        h.sealed_json(out / "result.json", result)
        if xid_count() != xid_baseline:
            raise h.StopCampaign("gpu_fault")
        if result["status"] == "passed":
            return result
    if result is None:
        raise h.StopCampaign("attempts_exhausted: " + task)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--context", type=int, default=102400)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--task", action="append", help="default: the eight long-tail tasks")
    args = parser.parse_args()
    tasks = args.task or s.LONG_TAIL_TASKS
    campaign, assets = args.campaign.resolve(), args.assets.resolve()
    pid, argv, env = server_process(int(args.url.rsplit(":", 1)[-1]))
    check_launch(argv)
    global MODEL_DIR
    MODEL_DIR = Path(argv[argv.index("serve") + 1])
    MODEL.update(id=MODEL_IDS[MODEL_DIR.name], local=str(MODEL_DIR))
    overrides = json.loads(argv[argv.index("--override-generation-config") + 1])
    launch = {"pid": pid, "command": argv, "env": env, "chat_template_sha256": TEMPLATE_SHA256,
              "vllm_version": api(args.url, "/version").get("version")}
    campaign.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign / "manifest.json"
    if not manifest_path.exists():
        git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        h.sealed_json(manifest_path, {
            "engine": "vllm", "model": MODEL, "tasks": tasks, "assets": str(assets), "url": args.url,
            "context": args.context, "timeout": args.timeout, "launch": launch, "repo_head": git,
            "sources": {str(p.relative_to(ROOT)): digest_file(p) for p in
                        (ROOT / "scripts/instruments/agentic_probe.py", Path(__file__).resolve())},
            "model_files": {p.name: p.stat().st_size for p in sorted(MODEL_DIR.iterdir()) if p.is_file()},
            "sampling": {"server_defaults": overrides,
                         "note": "min_p needs patches/minp-spec-decode-v2-responses.patch and the V2 runner; "
                                 "llama path used min_p 0.05"},
            "prompt_reuse": "vLLM prefix caching, reset before each task",
            "context_peak_source": VllmEngine.peak_source,
        })
    elif h.read_sealed(manifest_path)["tasks"] != tasks:
        raise SystemExit("campaign was prepared for a different task list")
    xid_baseline = xid_count()
    state = {"status": "running", "completed": {}}
    h.atomic_json(campaign / "state.json", state)
    try:
        for task in tasks:
            path = campaign / ("job-" + job_id(task) + ".json")
            if not path.exists():
                h.sealed_json(path, run_task(campaign, assets, task, args.url, args.context, args.timeout,
                                             launch, xid_baseline))
            state["completed"][job_id(task)] = path.name
            h.atomic_json(campaign / "state.json", state)
        state["status"] = "complete"
    except h.StopCampaign as error:
        state.update(status="stopped", reason=str(error))
    h.atomic_json(campaign / "state.json", state)
    s.MODEL = MODEL
    s.queue = lambda: [{"id": job_id(task), "task": task} for task in tasks]
    result = s.summary(campaign)
    with (campaign / "REPORT.md").open("a") as report:
        report.write("\nEngine: HyperQwen vLLM (" + str(launch["vllm_version"]) + "), one long-lived server; "
                     "prefix cache reset per task; sampling defaults " + json.dumps(overrides)
                     + "; context peak from " + VllmEngine.peak_source + ".\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
