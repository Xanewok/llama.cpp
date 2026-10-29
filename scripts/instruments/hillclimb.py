"""Bounded, restartable mixed-GPU screening with immutable campaign snapshots."""

from __future__ import annotations

import argparse
import csv
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time

from scripts import ROOT
from scripts.instruments import bench_mtp, chat_templates
from scripts.instruments.hillclimb_probe import api, digest_file


COMMIT = "03fa73cb27f5c251b9528489b18d303b1366aca4"
LLAMA = Path("/home/xanewok/repos/llama.cpp")
SERVER_EXTRA = []  # extra llama-server flags; frozen into a new campaign's manifest, which attempts read
# Models whose embedded template is the pinned Bartowski source (checked by hash in bartowski_template).
TEMPLATE_OVERRIDE_MODELS = {"bartowski-iq4-xs", "swift-iq4-xs", "swift-q4-m", "swift-iq3-xxs"}
GPU0 = "GPU-ee9a3956-297c-604b-b4c1-f07930d27e46"
GPU1 = "GPU-a4995e42-1ad9-148d-d7cd-9641bc1f5ece"
DEVICES = [GPU0, GPU1]
# These COSMIC components can hold display/Vulkan contexts. They are expected
# desktop load, not inference workloads; unknown compute clients still abort.
DISPLAY_COMPUTE_NAMES = {
    "cosmic-monitor", "cosmic-app-library", "cosmic-workspaces",
    "xdg-desktop-portal-cosmic", "cosmic-term",
}
UNSLOTH = "4ca720788d1e01f1bff70c033e0d0028fd02e502"
MODELS = [
    {"id": "iq3-control", "repo": None, "revision": None,
     "file": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
     "sha256": "c0b7c3038681ed2e3040456c1dd45f9858b6c2290bed172c70388a94874f3eee",
     "bytes": 10934860704,
     "local": "/home/xanewok/models/Qwen3.8-27B-UD-IQ3_XXS.gguf"},
    {"id": "unsloth-q4-xl", "repo": "unsloth/Qwen3.8-27B-GGUF", "revision": UNSLOTH,
     "file": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "bytes": 17559178144,
     "sha256": "3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e",
     "local": "/home/xanewok/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"},
    {"id": "unsloth-q4-m", "repo": "unsloth/Qwen3.8-27B-GGUF", "revision": UNSLOTH,
     "file": "Qwen3.8-27B-UD-Q4_K_M.gguf", "bytes": 16464440224,
     "sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"},
    {"id": "bartowski-iq4-xs", "repo": "bartowski/Qwen3.8-27B-GGUF",
     "revision": "e4cc3ff3b37f5aabf253d8583b0b0247e8b25b70",
     "file": "Qwen3.8-27B-IQ4_XS.gguf", "bytes": 15475951328,
     "sha256": "4b927360fa7c4aa41a734302f0795ca45df0e8b8a258bc76cc5fbf9d98a484db"},
    {"id": "unsloth-iq4-xs", "repo": "unsloth/Qwen3.8-27B-GGUF", "revision": UNSLOTH,
     "file": "Qwen3.8-27B-UD-IQ4_XS.gguf", "bytes": 14252845984,
     "sha256": "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199"},
    {"id": "jpetrina-iq4-pure", "repo": "jpetrina/Qwen3.8-27B-MTP-IQ4_XS-pure-GGUF",
     "revision": "28b472eeb494159f31a2ebe40223d2f5f6b9923c",
     "file": "qwen3.8-27b-mtp-IQ4_XS-pure.gguf", "bytes": 14534384640,
     "sha256": "ea5a3c45d407f9b9e5d2c0d647f0ea600f486f6b86b92b56d0823ba073dae675"},
]
TASKS = ["javascript/affine-cipher", "javascript/alphametics", "javascript/bowling",
         "javascript/connect", "rust/word-count", "rust/robot-name", "rust/forth",
         "rust/doubly-linked-list"]
# Q4's measured reserve is narrower than two percentage points; do not skip 70/30.
RATIOS = [69, 70, 68, 67, 71, 65, 73, 75]
TRANSIENT_FAILURES = {"server_failure", "transport_failure", "startup_timeout", "subprocess_timeout"}
RETRY_BACKOFF_SECONDS = (10, 30, 90, 270)


class StopCampaign(RuntimeError):
    pass


class RejectArm(RuntimeError):
    pass


class CampaignBusy(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sealed_json(path, value):
    atomic_json(path, value)
    atomic_json(str(path) + ".sha256", digest_file(path))


def read_sealed(path):
    expected = json.loads(Path(str(path) + ".sha256").read_text())
    if digest_file(path) != expected:
        raise StopCampaign(f"corrupt immutable record: {path}")
    return json.loads(Path(path).read_text())


def proc_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def desktop_inventory():
    return subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,args="], text=True, timeout=10).splitlines()


def _normal_process_name(value):
    tokens = value.strip().split()
    if tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    return Path(tokens[0] if tokens else "").name.lower()


def require_free_compute(allowed_processes=()):
    allowed = {_normal_process_name(name) for name in allowed_processes} | DISPLAY_COMPUTE_NAMES
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if row and row[0] in DEVICES and _normal_process_name(row[2] if len(row) > 2 else "") not in allowed:
            raise StopCampaign("unowned_gpu_workload: " + ",".join(row))


def telemetry():
    fields = "uuid,temperature.gpu,memory.free,power.limit,power.draw,utilization.gpu,pcie.link.gen.current,pcie.link.width.current"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True)
        rows = {}
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            values = [float(value) for value in row[1:]]
            if len(values) != 7 or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid GPU telemetry")
            rows[row[0]] = dict(zip(fields.split(",")[1:], values))
        memory = re.search(r"^MemAvailable:\s+(\d+) kB$", Path("/proc/meminfo").read_text(), re.M)
        if not memory or any(device not in rows for device in DEVICES):
            raise ValueError("missing device or available RAM")
        return {"time": time.time(), "gpus": rows, "ram_available_mib": int(memory[1]) / 1024}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise StopCampaign(f"telemetry_failed: {error}") from error


def safety_check(sample, reserves=True, vram_reserves=(512, 256)):
    if sample["ram_available_mib"] < 2048:
        raise StopCampaign("system_ram_below_2gib")
    if len(vram_reserves) != 2 or any(type(value) is not int or value < 0 for value in vram_reserves):
        raise StopCampaign("invalid_vram_reserves")
    for device, cap, reserve in zip(DEVICES, [150, 140], vram_reserves):
        gpu = sample["gpus"][device]
        if gpu["temperature.gpu"] >= 80:
            raise StopCampaign("thermal_limit: " + device)
        if abs(gpu["power.limit"] - cap) > 0.1:
            raise StopCampaign("power_limit_changed: " + device)
        if reserves and gpu["memory.free"] < reserve:
            raise RejectArm("vram_reserve: " + device)


def endpoint_free(port):
    with socket.socket() as sock:
        # TIME_WAIT is not a competing listener; match the server's reuse policy.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)


def classify_exit(log):
    lower = log.lower()
    # Routine startup warnings must never mask a later fatal device error.
    if any(word in lower for word in ("illegal memory access", "device-side assert", "xid",
                                     "launch timed out", "unspecified launch failure", "fallen off the bus")):
        return "gpu_fault"
    if any(word in lower for word in ("out of memory", "cudamalloc", "failed to allocate")):
        return "oom"
    if "llama_params_fit is not implemented for split_mode_tensor" in lower:
        return "vram_fit_unavailable"
    diagnostic = "\n".join(line for line in lower.splitlines() if not any(warning in line for warning in (
        "unsupported responses tool type", "backend sampling not supported", "swa_full is not supported")))
    if any(word in diagnostic for word in ("not supported", "unsupported", "invalid argument", "unknown argument")):
        return "unsupported"
    return "server_failure"


def queue():
    jobs = [{"id": "commission-depth", "kind": "smoke", "model": "iq3-control", "target": 4096},
            {"id": "commission-restart", "kind": "smoke", "model": "iq3-control", "target": 0},
            {"id": "control-single", "kind": "control", "model": "iq3-control", "single": True},
            {"id": "control-pair", "kind": "control", "model": "iq3-control", "single": False}]
    jobs += [{"id": "screen-" + m["id"], "kind": "screen", "model": m["id"]} for m in MODELS[1:]]
    jobs += [{"id": f"mtp-{m['id']}-{maximum}", "kind": "mtp", "model": m["id"], "maximum": maximum}
             for m in MODELS[1:4] for maximum in [2, 4, 6]]
    jobs += [{"id": f"slots-{rank}-{slots}", "kind": "concurrency", "rank": rank, "slots": slots}
             for rank in [0, 1] for slots in [2, 4]]
    jobs += [{"id": f"repeat-{rank}-{task.replace('/', '-')}-{seed}", "kind": "repeat",
              "rank": rank, "task": task, "seed": seed}
             for seed in [1730, 1731, 1732] for rank in [0, 1]
             for task in [TASKS[0], TASKS[2], TASKS[4], TASKS[6]]]
    return jobs


def validate_profiles(jobs, *, allowed_tasks=None):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("profile queue must be a nonempty array")
    seen = set()
    for job in jobs:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job.get("id", "")) or job["id"] in seen:
            raise ValueError("profile IDs must be unique safe filenames")
        if job.get("kind") != "profile" or job.get("model") not in {m["id"] for m in MODELS}:
            raise ValueError("profile queue must select a pinned model")
        if job.get("base_job") and job["base_job"] not in seen:
            raise ValueError("base_job must precede its consumer")
        seen.add(job["id"])
        if job.get("task", TASKS[0]) not in (TASKS if allowed_tasks is None else allowed_tasks):
            raise ValueError("profile task must belong to the frozen pilot")
        targets = job.get("targets", [98304])
        if not isinstance(targets, list) or not targets or any(type(n) is not int or n < 0 for n in targets):
            raise ValueError("targets must be nonnegative token counts")
        configs = job.get("configs", [{}])
        if not isinstance(configs, list) or not configs:
            raise ValueError("configs must be a nonempty array")
        for config in configs:
            if not isinstance(config, dict) or set(config) - {"context", "slots", "mode", "split", "ubatch", "maximum", "chat_template", "batch", "minimum", "p_min", "draft_k", "draft_v"}:
                raise ValueError("unknown profile setting")
            if "chat_template" in config and (job["model"] not in TEMPLATE_OVERRIDE_MODELS or
                    config["chat_template"] != chat_templates.BARTOWSKI_PREFIX):
                raise ValueError("chat template override must select the pinned leading-instructions adapter")
            for key in ("context", "slots", "ubatch", "maximum", "batch"):
                if key in config and (type(config[key]) is not int or config[key] < 1):
                    raise ValueError("profile sizes must be positive integers")
            minimum = config.get("minimum", 0)
            if type(minimum) is not int or not 0 <= minimum <= config.get("maximum", 3):
                raise ValueError("MTP minimum must be between zero and maximum")
            floor = config.get("p_min", 0)
            if type(floor) not in (int, float) or not math.isfinite(floor) or not 0 <= floor <= 1:
                raise ValueError("MTP probability floor must be between zero and one")
            if any(config.get(key, "f16") not in ("f16", "q8_0") for key in ("draft_k", "draft_v")):
                raise ValueError("draft cache must be f16 or experimental q8_0")
            if config.get("ubatch", 128) > config.get("batch", 2048):
                raise ValueError("microbatch exceeds logical batch")
            if config.get("slots", 1) > 4 or config.get("context", 102400) > 262144:
                raise ValueError("profile exceeds the declared model/slot bounds")
            if config.get("mode", "tensor") not in ("layer", "tensor"):
                raise ValueError("unsupported profile split mode")
            split = [float(n) for n in config.get("split", "69,31").split(",")]
            if len(split) != 2 or any(not math.isfinite(n) or n <= 0 for n in split):
                raise ValueError("profile split must give two positive finite weights")
    return jobs


def initialize(campaign, model_dir, smoke=False, profiles=None, max_attempts=None, agentic_assets=None,
               continuation=None, vram_reserves=(512, 256), measurement_contract=None):
    allowed_tasks = None
    if agentic_assets is not None:
        asset_tasks = json.loads((agentic_assets / "tasks.json").read_text())
        checks = json.loads((agentic_assets / "selftest.json").read_text())
        allowed_tasks = {task["task_id"] for task in asset_tasks}
        verified = {row["task_id"] for row in checks["tasks"]
                    if row["reference_passes"] and row["stub_fails"] and row["expected_passed_tests"] > 0}
        if (not checks["harness_verified"] or len(allowed_tasks) != len(asset_tasks)
                or allowed_tasks != verified):
            raise ValueError("every asset task requires passing reference/stub validation")
    if profiles is not None:
        validate_profiles(profiles, allowed_tasks=allowed_tasks)
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts not in range(1, 6)):
        raise ValueError("max_attempts must be between 1 and 5")
    if len(vram_reserves) != 2 or any(type(value) is not int or value < 0 for value in vram_reserves):
        raise ValueError("vram_reserves must contain two non-negative integers")
    if campaign.exists():
        manifest = read_sealed(campaign / "manifest.json")
        if profiles is not None and profiles != manifest["queue"]:
            raise ValueError("existing campaign has a different frozen queue")
        if smoke and manifest["queue"] != [{"id": "commission", "kind": "smoke", "model": "iq3-control"}]:
            raise ValueError("existing campaign is not a smoke campaign")
        if max_attempts is not None and max_attempts != manifest.get("max_attempts", 2):
            raise ValueError("existing campaign has a different frozen retry policy")
        if continuation is not None and continuation != manifest.get("continuation"):
            raise ValueError("existing campaign has a different continuation policy")
        if measurement_contract is not None and measurement_contract != manifest.get("measurement_contract"):
            raise ValueError("existing campaign has a different measurement contract")
        if SERVER_EXTRA != manifest.get("server_extra", []):
            raise ValueError("existing campaign has different extra server flags")
        if tuple(vram_reserves) != tuple(manifest.get("safety", {}).get("vram_reserves_mib", [512, 256])):
            raise ValueError("existing campaign has different VRAM reserves")
        for model in manifest["models"]:
            expected = model.get("local") or str(model_dir / model["id"] / model["file"])
            if model["path"] != expected:
                raise ValueError("existing campaign has different model paths")
        return
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "attempts").mkdir()
    (campaign / "artifacts").mkdir()
    snapshot = campaign / "snapshot"
    snapshot.mkdir()
    shutil.copytree(ROOT / "scripts", snapshot / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(LLAMA / "gguf-py" / "gguf", snapshot / "gguf-py" / "gguf",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "corpora").mkdir()
    for name in ["js-ts.txt", "rust.txt"]:
        shutil.copy2(ROOT / "corpora" / name, snapshot / "corpora" / name)
    tasks = json.loads((ROOT / "subsets/polyglot.json").read_text())
    selected = {t["task_id"]: {"task_id": t["task_id"], "language": t["language"],
                               "prompt": bench_mtp.bench_prompt(t, ROOT / "subsets")}
                for t in tasks if t["task_id"] in TASKS}
    if set(selected) != set(TASKS):
        raise ValueError("frozen pilot task is absent")
    atomic_json(snapshot / "tasks.json", selected)
    if agentic_assets is not None:
        shutil.copytree(agentic_assets, snapshot / "agentic-assets")
    models = json.loads(json.dumps(MODELS))
    for model in models:
        if "bytes" not in model:
            model["bytes"] = Path(model["local"]).stat().st_size
        model["path"] = model.get("local") or str(model_dir / model["id"] / model["file"])
    binary = LLAMA / "build/bin/llama-server"
    libraries = sorted((LLAMA / "build/bin").glob("*.so*"))
    pins = {str(p.resolve()): digest_file(p) for p in [binary, *libraries] if p.is_file()}
    frozen = {str(p.relative_to(snapshot)): digest_file(p)
              for p in snapshot.rglob("*") if p.is_file()}
    manifest = {"schema_version": 1, "created": time.time(), "source_root": str(ROOT),
                "engine_commit": COMMIT, "binary": str(binary), "binary_pins": pins,
                "devices": DEVICES, "port": 8081, "models": models,
                "queue": profiles if profiles is not None else (
                    [{"id": "commission", "kind": "smoke", "model": "iq3-control"}] if smoke else queue()),
                "snapshot_pins": frozen, "ratios": RATIOS,
                "safety": {"max_temp_c": 80, "display_free_mib": vram_reserves[0], "other_free_mib": vram_reserves[1],
                           "vram_reserves_mib": list(vram_reserves),
                           "ram_available_mib": 2048, "poll_seconds": 2},
                "max_attempts": max_attempts if max_attempts is not None else 3,
                "scope": "screening only; no graded quality claims or full-suite authorization"}
    if agentic_assets is not None:
        manifest["scope"] = "isolated verified coding tasks declared in queue; no automatic production promotion"
    if continuation is not None:
        manifest["continuation"] = continuation
    if measurement_contract is not None:
        manifest["measurement_contract"] = measurement_contract
        manifest["desktop_inventory_at_prepare"] = desktop_inventory()
    if SERVER_EXTRA:
        manifest["server_extra"] = list(SERVER_EXTRA)
    sealed_json(campaign / "manifest.json", manifest)
    atomic_json(campaign / "state.json", {"completed": {}, "status": "prepared", "owned": []})


class Runner:
    def __init__(self, campaign, budget_hours, deadline_epoch=None):
        self.path = campaign.resolve()
        self.manifest = read_sealed(self.path / "manifest.json")
        self.state = json.loads((self.path / "state.json").read_text())
        self.deadline = time.monotonic() + budget_hours * 3600
        if deadline_epoch is not None:
            self.deadline = min(self.deadline, time.monotonic() + max(0, deadline_epoch - time.time()))
        self.max_attempts = self.manifest.get("max_attempts", 3)
        self.vram_reserves = tuple(self.manifest.get("safety", {}).get("vram_reserves_mib", [512, 256]))
        if type(self.max_attempts) is not int or self.max_attempts not in range(1, 6):
            raise StopCampaign("invalid frozen retry policy")
        self.snapshot = self.path / "snapshot"
        self.url = f"http://127.0.0.1:{self.manifest['port']}"
        self.children = []
        self.lockfiles = []
        self.stopping = False
        self.current_dir = None
        self.last_sample = self.last_report = 0
        self.samples = []
        self.verified = {}
        self.env = dict(os.environ, PYTHONPATH=str(self.snapshot), PYTHONDONTWRITEBYTECODE="1",
                        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(DEVICES),
                        PATH="/usr/local/cuda-13.2/bin:" + os.environ.get("PATH", ""))
        # An inherited tuning variable must not silently change the control.
        for key in list(self.env):
            if key.startswith(("NCCL_", "GGML_CUDA_")):
                self.env.pop(key)

    def save(self):
        atomic_json(self.path / "state.json", self.state)

    def event(self, kind, **data):
        row = {"time": time.time(), "kind": kind, **data}
        with (self.path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row), flush=True)

    def report(self):
        status = {"status": self.state["status"], "active": self.state.get("active"),
                  "completed_jobs": len(self.state["completed"]), "total_jobs": len(self.manifest["queue"]),
                  "remaining_seconds": max(0, self.deadline - time.monotonic()),
                  "updated": time.time(), "pid": os.getpid(), "reason": self.state.get("reason"),
                  "resource_wait": self.state.get("resource_wait"),
                  "telemetry": self.samples[-1] if self.samples else None}
        atomic_json(self.path / "status.json", status)
        lines = ["# Autonomous screening status", "", f"Status: {status['status']}",
                 f"Completed: {status['completed_jobs']}/{status['total_jobs']}",
                 f"Active: {status['active']}", f"Reason: {status['reason']}", "",
                 "These are throughput/allocation probes, not graded coding results.", "",
                 "| Job | Outcome | PP tok/s | TG tok/s | MTP acceptance |",
                 "|---|---|---:|---:|---:|"]
        for job, value in self.state["completed"].items():
            result = read_sealed(self.path / value)
            acceptance = result.get("probe", {}).get("mtp", {}).get("draft_accept_rate")
            rate = "" if acceptance is None else f"{acceptance:.1%}"
            lines.append(f"| {job} | {result['status']} | {result.get('prefill_tps', '')} | {result.get('decode_tps', '')} | {rate} |")
        temp = self.path / "REPORT.md.tmp"
        temp.write_text("\n".join(lines) + "\n")
        os.replace(temp, self.path / "REPORT.md")
        self.last_report = time.monotonic()

    def poll(self, reserves=True):
        if self.stopping or (self.path / "STOP").exists():
            raise StopCampaign("requested_stop")
        if time.monotonic() >= self.deadline:
            raise StopCampaign("budget_expired")
        if time.monotonic() - self.last_sample >= 2:
            sample = telemetry()
            self.samples.append(sample)
            dest = self.current_dir or self.path
            with (dest / "telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            self.last_sample = time.monotonic()
            safety_check(sample, reserves, self.vram_reserves)
        if time.monotonic() - self.last_report >= 1800:
            self.report()

    def sleep(self, seconds, reserves=True):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            self.poll(reserves)
            time.sleep(min(0.5, max(0, until - time.monotonic())))

    def spawn(self, command, log, env=None):
        parent = os.getpid()

        def child_setup():
            os.setsid()
            # Parent death must not leave an unguarded inference process running.
            if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent:
                os._exit(126)

        with Path(log).open("ab") as stream:
            child = subprocess.Popen(command, cwd=self.snapshot, env=env or self.env,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, preexec_fn=child_setup)
        identity = {"pid": child.pid, "start": proc_identity(child.pid), "command": command}
        self.children.append(child)
        self.state["owned"].append(identity)
        self.save()
        self.event("spawn", **identity)
        return child

    def stop_child(self, child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        self.children = [p for p in self.children if p is not child]
        self.state["owned"] = [p for p in self.state["owned"] if p["pid"] != child.pid]
        self.save()

    def wait_child(self, child, timeout, reserves=True):
        deadline = time.monotonic() + timeout
        try:
            while child.poll() is None:
                self.poll(reserves)
                if time.monotonic() >= deadline:
                    raise RejectArm("subprocess_timeout")
                time.sleep(0.5)
            self.poll(reserves)
            return child.returncode
        finally:
            self.stop_child(child)

    def worker(self, arguments, log, timeout=900):
        child = self.spawn([sys.executable, "-m", "scripts.instruments.hillclimb_probe", *arguments], log)
        return self.wait_child(child, timeout)

    def run_probe(self, config, target, task, seed, out):
        return self.worker(["probe", "--url", self.url, "--tasks", str(self.snapshot / "tasks.json"),
                            "--task", task, "--target", str(target), "--max-tokens", "32" if target == 0 else "2048",
                            "--slots", str(config["slots"]), "--seed", str(seed), "--out", str(out)],
                           out / "probe.log", timeout=900)

    def acquire(self):
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        for lock in [runtime / "qwen-hillclimb.lock", runtime / "qwen-local-server.lock"]:
            handle = lock.open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise CampaignBusy("another benchmark or local server owns the GPU lock") from error
            self.lockfiles.append(handle)
        for item in self.state.get("owned", []):
            if proc_identity(item["pid"]) == item["start"] and item["start"] is not None:
                raise StopCampaign("owned_process_still_alive: inspect before resume")
        self.state["owned"] = []
        for proc in Path("/proc").iterdir():
            if proc.name.isdigit():
                try:
                    if (proc / "comm").read_text().strip() == "llama-server":
                        raise StopCampaign("existing_llama_server")
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
        endpoint_free(self.manifest["port"])
        if self.manifest.get("measurement_contract"):
            require_free_compute(self.manifest.get("desktop_inventory_at_prepare", []))
            self.event("desktop_inventory_at_resume", processes=desktop_inventory())
        for relative, expected in self.manifest["snapshot_pins"].items():
            if digest_file(self.snapshot / relative) != expected:
                raise StopCampaign("snapshot_changed: " + relative)
        for path, expected in self.manifest["binary_pins"].items():
            if digest_file(path) != expected:
                raise StopCampaign("engine_changed: " + path)
        commit = subprocess.check_output(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True).strip()
        if commit != COMMIT:
            raise StopCampaign("engine_source_changed")
        for value in self.state["completed"].values():
            read_sealed(self.path / value)
        self.poll()
        self.state["status"] = "running"
        self.state["reason"] = None
        self.state["coordinator"] = {"pid": os.getpid(), "start": proc_identity(os.getpid()),
                                     "pid_namespace": os.readlink("/proc/self/ns/pid")}
        self.save()
        self.report()

    def ensure_model(self, model_id):
        if model_id in self.verified:
            return self.verified[model_id]
        model = next(m for m in self.manifest["models"] if m["id"] == model_id)
        path = Path(model["path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            required = model["bytes"] - (partial.stat().st_size if partial.exists() else 0)
            if shutil.disk_usage(path.parent).free < required + 10 * 1024**3:
                raise RejectArm("insufficient_disk_space")
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['file']}"
            for attempt in range(self.max_attempts):
                self.event("download", model=model_id, attempt=attempt + 1)
                command = ["curl", "--fail", "--location", "--continue-at", "-",
                           "--connect-timeout", "15", "--speed-limit", "1024", "--speed-time", "60",
                           "--output", str(partial), url]
                child = self.spawn(command, self.path / "artifacts" / f"{model_id}.download.log")
                code = self.wait_child(child, min(3600, self.deadline - time.monotonic()))
                if code == 0:
                    break
                if attempt + 1 < self.max_attempts:
                    self.sleep(RETRY_BACKOFF_SECONDS[attempt])
            else:
                raise RejectArm("download_failed")
            inspect_path = partial
        else:
            inspect_path = path
        metadata = self.path / "artifacts" / f"{model_id}.json"
        self.event("verify_model", model=model_id)
        code = self.worker(["inspect", "--path", str(inspect_path), "--sha256", model["sha256"],
                            "--bytes", str(model["bytes"]), "--gguf-path", str(self.snapshot / "gguf-py"),
                            "--out", str(metadata)], metadata.with_suffix(".log"), timeout=600)
        if code != 0:
            raise RejectArm("artifact_verification_failed")
        info = json.loads(metadata.read_text())
        if info["native_context"] < 102400 or info["block_count"] != 65 or info["architecture"] != "qwen35":
            raise RejectArm("unexpected_model_architecture")
        if inspect_path != path:
            os.replace(inspect_path, path)
        self.verified[model_id] = model
        return model

    def template_override(self, model, config, out):
        if "chat_template" not in config:
            return [], None
        if model["id"] not in TEMPLATE_OVERRIDE_MODELS or config["chat_template"] != chat_templates.BARTOWSKI_PREFIX:
            raise RejectArm("unsupported_chat_template_override")
        metadata = json.loads((self.path / "artifacts" / f"{model['id']}.json").read_text())
        effective = chat_templates.bartowski_template(metadata["chat_template"])
        path = out / "chat-template.jinja"
        path.write_text(effective)
        return ["--jinja", "--chat-template-file", str(path)], {
            "adapter": config["chat_template"],
            "source_sha256": chat_templates.BARTOWSKI_TEMPLATE_SHA256,
            "effective_sha256": digest_file(path)}

    def retryable_failure(self, reason):
        return reason in TRANSIENT_FAILURES

    def retryable_gpu_fault(self, log):
        return False

    def retry_limit(self, reason):
        return self.max_attempts

    def retry_count(self, failures, started_attempts):
        return len(failures)

    def retry_backoff(self, failures, count):
        return RETRY_BACKOFF_SECONDS[len(failures) - 1]

    def wait_before_retry(self, job_id, result, seconds):
        self.sleep(seconds)

    def attempt(self, job_id, model, config, target, task=TASKS[0], seed=1729):
        spec = {"job": job_id, "model": model["id"], "config": config,
                "target": target, "task": task, "seed": seed}
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        # Resume preserves successes and deterministic rejections; partial attempts never pass.
        transient_failures = []
        for existing in sorted((self.path / "attempts").glob(key + "-*/result.json")):
            cached = read_sealed(existing)
            if cached["status"] == "passed":
                return cached
            if cached.get("reason") == "gpu_fault":
                raise StopCampaign("gpu_fault")
            if cached["status"] == "failed" and not self.retryable_failure(cached.get("reason")):
                return cached
            if cached["status"] == "failed":
                transient_failures.append(cached)
        started_attempts = len(list((self.path / "attempts").glob(key + "-*")))
        count = self.retry_count(transient_failures, started_attempts)
        if transient_failures and count >= self.retry_limit(transient_failures[-1].get("reason")):
            return transient_failures[-1]
        if transient_failures:
            delay = self.retry_backoff(transient_failures, count)
            retry_at = transient_failures[-1].get("finished_at", time.time()) + delay
            remaining = max(0, retry_at - time.time())
            self.event("transient_retry", job=job_id, attempt=count + 1,
                       max_attempts=self.retry_limit(transient_failures[-1].get("reason")), backoff_seconds=remaining,
                       reason=transient_failures[-1]["reason"])
            self.wait_before_retry(job_id, transient_failures[-1], remaining)
        number = started_attempts + 1
        out = self.path / "attempts" / f"{key}-{number:02d}"
        out.mkdir()
        self.current_dir = out
        self.samples = []
        self.last_sample = 0
        sealed_json(out / "spec.json", spec)
        result = {**spec, "status": "failed", "attempt": str(out.relative_to(self.path))}
        server = None
        began = time.monotonic()
        try:
            endpoint_free(self.manifest["port"])
            if self.manifest.get("measurement_contract"):
                require_free_compute(self.manifest.get("desktop_inventory_at_prepare", []))
                sealed_json(out / "desktop.json", desktop_inventory())
            self.poll()
            env = dict(self.env, CUDA_VISIBLE_DEVICES=",".join(config["devices"]))
            command = [self.manifest["binary"], "-m", model["path"], "--alias", "local-benchmark",
                       "-c", str(config["context"] * config["slots"]), "-np", str(config["slots"]),
                       "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-fa", "on",
                       "--spec-type", "draft-mtp", "--spec-draft-n-max", str(config["maximum"]),
                       "--spec-draft-n-min", str(config.get("minimum", 0)), "--spec-draft-p-min", str(config.get("p_min", 0)),
                       "--spec-draft-type-k", config.get("draft_k", "f16"), "--spec-draft-type-v", config.get("draft_v", "f16"),
                       "-sm", config["mode"], "--tensor-split", config["split"],
                       "-b", str(config.get("batch", 2048)), "-ub", str(config["ubatch"]),
                       "--host", "127.0.0.1", "--port", str(self.manifest["port"]), "--metrics"]
            if config.get("workload") == "agentic":
                command += ["--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05"]
            template_args, template_info = self.template_override(model, config, out)
            command += template_args + self.manifest.get("server_extra", [])
            sealed_json(out / "launch.json", {"command": command, "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                                               "target_kv": "q8_0/q8_0", "draft_kv": config.get("draft_k", "f16") + "/" + config.get("draft_v", "f16"),
                                               "model_sha256": model["sha256"], "chat_template_override": template_info})
            self.event("attempt_start", **spec, directory=str(out))
            server = self.spawn(command, out / "server.log", env)
            until = time.monotonic() + 300
            while True:
                self.poll()
                if server.poll() is not None:
                    raise RejectArm(classify_exit((out / "server.log").read_text(errors="replace")))
                try:
                    if api(self.url, "/health", timeout=1).get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= until:
                    raise RejectArm("startup_timeout")
                self.sleep(1)
            props = api(self.url, "/props", timeout=3)
            if (props["total_slots"] != config["slots"] or
                    props["default_generation_settings"]["n_ctx"] != config["context"] or
                    Path(props["model_path"]) != Path(model["path"])):
                raise RejectArm("server_configuration_mismatch")
            atomic_json(out / "props.json", props)
            self.sleep(2)
            if self.manifest.get("measurement_contract"):
                atomic_json(out / "memory-loaded.json", telemetry())
            code = self.run_probe(config, target, task, seed, out)
            if code != 0:
                if server.poll() is not None:
                    raise RejectArm(classify_exit((out / "server.log").read_text(errors="replace")))
                if (out / "probe.json").exists():
                    failed_probe = json.loads((out / "probe.json").read_text())
                    if failed_probe.get("failure_class") == "transport_failure":
                        raise RejectArm("transport_failure")
                raise RejectArm("probe_failed")
            probe = json.loads((out / "probe.json").read_text())
            if self.manifest.get("measurement_contract"):
                atomic_json(out / "memory-populated.json", telemetry())
            if not all(probe[k] for k in ["complete", "mtp_live", "depth_verified", "overlap_verified"]):
                raise RejectArm("probe_incomplete")
            result.update(status="passed", probe=probe, decode_tps=probe["metrics_delta"].get("decode_tps"),
                          prefill_tps=probe["metrics_delta"].get("prefill_tps"))
        except RejectArm as error:
            result["reason"] = str(error)
            log = out / "server.log"
            diagnostic = log.read_text(errors="replace") if log.exists() else ""
            if str(error) == "gpu_fault" or classify_exit(diagnostic) == "gpu_fault":
                result["reason"] = "gpu_fault"
                if self.retryable_gpu_fault(diagnostic):
                    result["reason"] = "cuda_watchdog"
                else:
                    raise StopCampaign("gpu_fault") from error
        except StopCampaign as error:
            result.update(status="interrupted", reason=str(error))
            raise
        except Exception as error:
            result.update(status="error", reason=f"{type(error).__name__}: {error}")
            raise StopCampaign("instrument_fault: " + result["reason"]) from error
        finally:
            for child in list(reversed(self.children)):
                self.stop_child(child)
            result["finished_at"] = time.time()
            result["elapsed_seconds"] = time.monotonic() - began
            if self.samples:
                result["minimum_free_mib"] = {gpu: min(s["gpus"][gpu]["memory.free"] for s in self.samples) for gpu in DEVICES}
                result["peak_temperature_c"] = {gpu: max(s["gpus"][gpu]["temperature.gpu"] for s in self.samples) for gpu in DEVICES}
                result["sampled_board_energy_wh"] = sum(
                    (b["time"] - a["time"]) * sum((a["gpus"][gpu]["power.draw"] + b["gpus"][gpu]["power.draw"]) / 2
                                                    for gpu in DEVICES) / 3600
                    for a, b in zip(self.samples, self.samples[1:]))
                result["energy_scope"] = "whole_attempt_including_load_and_probe; trapezoidal_board_power"
            sealed_json(out / "result.json", result)
            self.event("attempt_end", job=job_id, status=result["status"], reason=result.get("reason"))
            self.current_dir = None
        self.sleep(4, reserves=False)
        # Memory recovery is checked before another model is admitted.
        self.last_sample = 0
        try:
            self.poll()
        except RejectArm as error:
            raise StopCampaign("memory_not_recovered") from error
        if self.retryable_failure(result.get("reason")):
            return self.attempt(job_id, model, config, target, task, seed)
        return result

    def best(self):
        choices = []
        for job_id, path in self.state["completed"].items():
            if job_id.startswith(("screen-", "mtp-")):
                result = read_sealed(self.path / path)
                if result["status"] == "passed" and result.get("decode_tps"):
                    choices.append(result)
        choices.sort(key=lambda r: (-r["decode_tps"], -min(r["minimum_free_mib"].values()),
                                    next(i for i, m in enumerate(MODELS) if m["id"] == r["model"])))
        unique = []
        for choice in choices:
            if not any(item["model"] == choice["model"] for item in unique):
                unique.append(choice)
        return unique[:2]

    def execute_job(self, job):
        config = {"devices": DEVICES, "context": 102400, "slots": 1, "mode": "tensor",
                  "split": "69,31", "ubatch": 128, "maximum": 3}
        if job["kind"] == "profile":
            if job.get("base_job"):
                path = self.state["completed"].get(job["base_job"])
                base = read_sealed(self.path / path) if path else {}
                if base.get("status") != "passed":
                    return {"status": "skipped", "reason": "no_safe_base_profile", "model": job["model"]}
                config.update(base["config"])
            model = self.ensure_model(job["model"])
            last = None
            for changes in job.get("configs", [{}]):
                candidate = {**config, **changes}
                targets = job.get("targets", [98304])
                if any(target and target + 2048 > candidate["context"] for target in targets):
                    raise RejectArm("profile_output_exceeds_context")
                last = self.attempt(job["id"] + "-fit", model, candidate, 0)
                if last["status"] != "passed":
                    continue
                rows = []
                for target in targets:
                    last = self.attempt(job["id"], model, candidate, target,
                                        task=job.get("task", TASKS[0]), seed=job.get("seed", 1729))
                    rows.append(last)
                    if last["status"] != "passed":
                        break
                if all(row["status"] == "passed" for row in rows):
                    return {**last, "probes": rows}
            return last
        if job["kind"] in ("concurrency", "repeat"):
            best = self.best()
            if len(best) <= job["rank"]:
                return {"status": "skipped", "reason": "no_safe_candidate"}
            base = best[job["rank"]]
            model_id = base["model"]
            config.update(base["config"])
            config["slots"] = job.get("slots", 1)
        else:
            model_id = job["model"]
        model = self.ensure_model(model_id)
        if job["kind"] == "smoke":
            config.update(context=8192, devices=[GPU0], mode="layer", split="1")
            return self.attempt(job["id"], model, config, job.get("target", 4096))
        if job["kind"] == "control":
            config.update(context=65536, ubatch=512, split="1" if job["single"] else "70,30",
                          devices=[GPU0] if job["single"] else DEVICES,
                          mode="layer" if job["single"] else "tensor")
            rows = [self.attempt(job["id"], model, config, target) for target in [32768, 49152]]
            if not all(row["status"] == "passed" for row in rows):
                return {"status": "failed", "reason": "control_failed", "attempts": rows}
            return {"status": "passed", "attempts": rows,
                    "decode_tps": statistics.median(row["decode_tps"] for row in rows)}
        if job["kind"] == "screen":
            for ratio in self.manifest["ratios"]:
                config["split"] = f"{ratio},{100 - ratio}"
                fit = self.attempt(job["id"] + "-fit", model, dict(config), 0)
                if fit["status"] != "passed":
                    if fit.get("reason") not in ("oom", "unsupported", "vram_reserve: " + GPU0, "vram_reserve: " + GPU1):
                        return fit
                    continue
                depth = self.attempt(job["id"], model, dict(config), 98304)
                if depth["status"] == "passed":
                    return depth
                if depth.get("reason") not in ("oom", "vram_reserve: " + GPU0, "vram_reserve: " + GPU1):
                    return depth
            return {"status": "failed", "reason": "no_safe_ratio", "model": model_id}
        if job["kind"] == "mtp":
            baseline_path = self.state["completed"].get("screen-" + model_id)
            if not baseline_path:
                return {"status": "skipped", "reason": "screen_unavailable"}
            baseline = read_sealed(self.path / baseline_path)
            if baseline["status"] != "passed":
                return {"status": "skipped", "reason": "no_safe_screen"}
            config.update(baseline["config"], maximum=job["maximum"])
        if job["kind"] == "concurrency":
            fit = self.attempt(job["id"] + "-fit", model, config, 0)
            if fit["status"] != "passed":
                return fit
        return self.attempt(job["id"], model, config, 98304,
                            task=job.get("task", TASKS[0]), seed=job.get("seed", 1729))

    def run(self):
        busy = False
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, lambda *_: setattr(self, "stopping", True))
        try:
            self.acquire()
            for job in self.manifest["queue"]:
                if job["id"] in self.state["completed"]:
                    continue
                self.poll()
                dest = self.path / f"job-{job['id']}.json"
                # A crash between sealing the result and saving state must not repeat inference.
                if dest.exists():
                    recovered = read_sealed(dest)
                    if recovered.get("status") not in ("passed", "failed", "skipped"):
                        raise StopCampaign("invalid terminal job record: " + job["id"])
                    self.state["completed"][job["id"]] = dest.name
                    self.save()
                    self.event("job_recovered", job=job["id"], status=recovered["status"])
                    if job["kind"] == "smoke" and recovered["status"] != "passed":
                        raise StopCampaign("commissioning_failed")
                    continue
                self.state["active"] = job
                self.save()
                self.report()
                try:
                    result = self.execute_job(job)
                except RejectArm as error:
                    result = {"status": "failed", "reason": str(error)}
                sealed_json(dest, result)
                self.state["completed"][job["id"]] = str(dest.relative_to(self.path))
                self.save()
                self.event("job_end", job=job["id"], status=result["status"])
                if job["kind"] == "smoke" and result["status"] != "passed":
                    raise StopCampaign("commissioning_failed")
            self.state["status"] = "complete"
            self.state["active"] = None
        except CampaignBusy as error:
            busy = True
            print(str(error), flush=True)
        except StopCampaign as error:
            self.state.update(status="stopped", reason=str(error))
            self.event("campaign_stop", reason=str(error))
        except BaseException as error:
            self.state.update(status="error", reason=f"{type(error).__name__}: {error}")
            self.event("campaign_error", reason=self.state["reason"])
            raise
        finally:
            for child in list(reversed(self.children)):
                self.stop_child(child)
            if not busy:
                self.save()
                self.report()
            for handle in self.lockfiles:
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "resume", "status", "stop", "prepare"])
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--budget-hours", default=4, type=float)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".cache/hillclimb-models")
    parser.add_argument("--smoke", action="store_true", help="one small real-GPU commissioning probe only")
    parser.add_argument("--queue-json", type=Path, help="explicit pinned-model follow-up profiles; new campaigns only")
    parser.add_argument("--swift-tuning", action="store_true", help="prepare the pinned Swift IQ4 throughput contract")
    parser.add_argument("--max-attempts", type=int, choices=range(1, 6),
                        help="total transient attempts per configuration, frozen at creation (default: 3)")
    parser.add_argument("--deadline-epoch", type=float,
                        help="absolute wall-clock deadline, preserved across supervised restarts")
    args = parser.parse_args()
    if not math.isfinite(args.budget_hours) or args.budget_hours <= 0:
        parser.error("budget must be positive and finite")
    if args.deadline_epoch is not None and (not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= 0):
        parser.error("deadline must be a positive finite epoch")
    if args.max_attempts is not None and args.command not in ("run", "prepare"):
        parser.error("--max-attempts is set only when creating a campaign")
    campaign = args.campaign.resolve()
    if args.command in ("run", "prepare"):
        if args.smoke and args.queue_json:
            parser.error("--smoke and --queue-json are mutually exclusive")
        profiles = json.loads(args.queue_json.read_text()) if args.queue_json else None
        contract = None
        if args.swift_tuning:
            if not profiles or args.smoke:
                parser.error("--swift-tuning requires an explicit --queue-json")
            from scripts.instruments.swift_longtail import MODEL, TUNING_TASKS, tuning_contract
            MODELS[:] = [MODEL]
            TASKS[:] = TUNING_TASKS
            contract = tuning_contract()
        initialize(campaign, args.model_dir.resolve(), smoke=args.smoke, profiles=profiles,
                   max_attempts=args.max_attempts, measurement_contract=contract)
        if args.command == "prepare":
            print(campaign)
            return
    elif args.command == "status":
        state = json.loads((campaign / "state.json").read_text())
        owner = state.get("coordinator", {})
        same_namespace = owner.get("pid_namespace") == os.readlink("/proc/self/ns/pid")
        if state["status"] == "running" and same_namespace and proc_identity(owner.get("pid", -1)) != owner.get("start"):
            print("Coordinator is no longer alive; the report below is its last saved state.")
        print((campaign / "REPORT.md").read_text() if (campaign / "REPORT.md").exists()
              else (campaign / "state.json").read_text())
        return
    elif args.command == "stop":
        (campaign / "STOP").touch()
        return
    if (campaign / "STOP").exists():
        parser.error("STOP marker exists; explicitly archive it before resuming")
    snapshot = campaign / "snapshot"
    if ROOT.resolve() != snapshot.resolve():
        env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONDONTWRITEBYTECODE="1")
        os.chdir(snapshot)
        command = [sys.executable, "-m", "scripts.instruments.hillclimb", "resume",
                   "--campaign", str(campaign), "--budget-hours", str(args.budget_hours)]
        if args.deadline_epoch is not None:
            command += ["--deadline-epoch", str(args.deadline_epoch)]
        os.execve(sys.executable, command, env)
    Runner(campaign, args.budget_hours, args.deadline_epoch).run()


if __name__ == "__main__":
    main()
