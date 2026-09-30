"""Frozen campaign contract shared by planning, execution, and summary checks."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path.home() / "repos/qwen38-eval"
SP = Path("/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad")
RUNNER = EVAL / "results/session2/pair-5060ti-x2/run_swift_longtail.py"
ASSETS = EVAL / "results/session2/pair-5060ti-2060s/campaign-confirmed-fullsuite-20260914-r8/snapshot/agentic-assets"
FROZEN = ROOT / "frozen/stack2"
COMMIT = "0dba27b9002b8347c6f14a9f4a1651b3478065ef"
IDS = Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
MODEL_A = Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf"
SHA_A = "e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df"
ARMS = {"S0": 3, "S3": 3, "S4": 4, "S5": 5}
ORDER = (("S0", "S3", "S4", "S5"), ("S5", "S4", "S3", "S0"), ("S0", "S3", "S4", "S5"))
_BASE_QUEUE = None


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_b():
    directory = Path("/mnt/scratch/work/models/j5")
    preferred = directory / "Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf"
    fallback = directory / "Swift-1.5-Qwen3.8-27B-fdown6mid-bf16.gguf"
    path = Path(os.environ["MODEL_B"]) if os.environ.get("MODEL_B") else (
        preferred if preferred.is_file() and Path(str(preferred) + ".sha256").is_file() else fallback)
    sidecar = Path(str(path) + ".sha256")
    fields = sidecar.read_text().split()
    digest = fields[0]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"invalid digest in {sidecar}")
    if len(fields) > 1 and fields[1].isdigit() and path.stat().st_size != int(fields[1]):
        raise ValueError(f"model size differs from {sidecar}")
    return path.resolve(), digest


def extra(arm):
    flags = ["--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling", "-t", "8", "-tb", "8"]
    if arm != "S0":
        flags += ["--spec-draft-vocab", str(IDS)]
    return shlex.join(flags)


def configure(arm, model, digest):
    global _BASE_QUEUE
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    os.environ.update(SWIFT_ENGINE="frozen", SWIFT_LLAMA=str(FROZEN), SWIFT_COMMIT=COMMIT,
                      SWIFT_DRAFT_MAX=str(ARMS[arm]), SWIFT_EXTRA=extra(arm), SWIFT_MODEL="swift15-iq4-xs")
    os.environ.pop("SWIFT_TASKS", None)
    sys.path.insert(0, str(EVAL))
    from scripts.instruments import swift_longtail
    if _BASE_QUEUE is None:
        _BASE_QUEUE = swift_longtail.queue
    # runpy installs a wrapper each time; do not retain wrappers from older arms.
    swift_longtail.queue = _BASE_QUEUE
    g = runpy.run_path(str(RUNNER), run_name="weekend_check")
    h, s = g["h"], g["s"]
    if model != MODEL_A:
        model_id = "swift15-weekend-" + digest[:12]
        g["SWIFT15"][model_id] = (model.name, model.stat().st_size, digest)
        g["LOCAL_BUILDS"][model_id] = {"repo": "local:gguf-j weekend " + model.stem, "revision": digest}
        g["MODEL_DIRS"][model_id] = model.parent
        s.MODEL_PATH = model
        s.MODEL = {"id": model_id, "file": model.name, "bytes": model.stat().st_size,
                   "sha256": digest, "local": str(model), **g["LOCAL_BUILDS"][model_id]}
        h.TEMPLATE_OVERRIDE_MODELS.add(model_id)
        os.environ["SWIFT_MODEL"] = model_id
    jobs = s.queue()
    s.configure_globals()
    assert len(jobs) == 8 and len({j["task"] for j in jobs}) == 8
    assert all(j["configs"][0]["maximum"] == ARMS[arm] and j["configs"][0]["split"] == "50,50" for j in jobs)
    assert "--spec-draft-n-max" not in h.SERVER_EXTRA
    return h, s, jobs


def check_frozen():
    assert subprocess.check_output(["git", "-C", str(FROZEN), "rev-parse", "HEAD"], text=True).strip() == COMMIT
    assert (FROZEN / "SOURCE").read_text().splitlines()[0] == COMMIT
    assert (FROZEN / "gguf-py/gguf/__init__.py").is_file()
    directory = FROZEN / "build/bin"
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        assert sha(directory / name) == digest, f"changed frozen file {name}"
    assert sha(directory / "llama-server") == sha(directory / "weekend-srv")
    assert b"GGML_CUDA_FA_MMA_Q8_0" in (directory / "libggml-cuda.so").read_bytes(), "q8_0 FA kernel knob missing"
    output = subprocess.check_output(["ldd", str(directory / "llama-server")], env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""}, text=True)
    libs = [line for line in output.splitlines() if any(name in line for name in ("libggml", "libllama", "libmtmd"))]
    assert libs and all(f"=> {directory}/" in line for line in libs) and "not found" not in output


def check_campaign(path, arm, model, digest):
    summary = json.loads((path / "SUMMARY.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    _, _, queue = configure(arm, model, digest)
    assert summary["state"] == "complete" and summary["tasks"] == 8
    assert manifest["engine_commit"] == COMMIT and manifest["binary"] == str(FROZEN / "build/bin/llama-server")
    assert manifest["queue"] == queue and manifest.get("server_extra") == shlex.split(extra(arm))
    assert len(manifest["models"]) == 1 and manifest["models"][0]["sha256"] == digest
    assert manifest["models"][0]["path"] == str(model)
    jobs = [json.loads(p.read_text()) for p in path.glob("job-*.json")]
    assert len(jobs) == 8 and all(j.get("status") == "passed" for j in jobs)
    pins = {str(p.resolve()): sha(p) for p in (FROZEN / "build/bin").iterdir()
            if p.name == "llama-server" or ".so" in p.name}
    assert manifest["binary_pins"] == pins
    attempts = sorted(path.glob("attempts/*/launch.json"))
    assert attempts
    for attempt in attempts:
        launch = json.loads(attempt.read_text())
        command = launch["command"]
        for flag, value in (("-ctk", "q8_0"), ("-ctv", "q8_0"), ("--spec-draft-n-max", str(ARMS[arm]))):
            assert command.count(flag) == 1 and command[command.index(flag) + 1] == value, attempt
        assert "--cache-type-k" not in command and "--cache-type-v" not in command
        assert launch["target_kv"] == "q8_0/q8_0"
        assert launch.get("executed_command", [""])[0] == str(FROZEN / "build/bin/weekend-srv")
        assert "--backend-sampling" in command and "--spec-draft-backend-sampling" in command
        assert "--no-spec-draft-backend-sampling" not in command
        assert launch["model_sha256"] == digest
        import re
        log = attempt.with_name("server.log").read_text(errors="replace")
        assert all(m == "q8_0" for m in re.findall(r"type_[kv]\s*=\s*(\w+)", log)), attempt
        assert all(m == "q8_0" for m in re.findall(r"\b[KV] \((\w+)\): +[\d.]+ MiB", log)), attempt
    return summary
