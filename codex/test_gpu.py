#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import gpu_ab as h

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.nice(19)
root = Path(__file__).resolve().parent.parent
os.chdir(root)
fixtures = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
h.fixture_test(fixtures)
with tempfile.TemporaryDirectory(dir=root / "build-tmp") as tmp:
    temp = Path(tmp)
    frozen = temp / "frozen"; (frozen / "bin").mkdir(parents=True)
    shutil.copyfile(root / "build-ref/host-depth-base", frozen / "bin/host-depth-server")
    (frozen / "SOURCE").write_text("c23324d5f9f0fc00ad3f2e6441e39ba061f41544\n")
    (frozen / "bin/SHA256SUMS").write_text(h.sha(frozen / "bin/host-depth-server") + "  host-depth-server\n")
    (frozen / "SHA256SUMS").write_text("".join(h.sha(frozen / f) + "  " + f + "\n" for f in ("SOURCE", "bin/SHA256SUMS")))
    binary, pins = h.frozen(frozen)
    assert binary.name == "host-depth-server" and len(pins) == 3
    model = temp / "model"; model.write_bytes(b"CPU plan fixture, never execute")
    cmd = ["python3", "codex/gpu_ab.py", "--up", str(frozen), "--patch", str(frozen), "--model", str(model), "--model-sha", h.sha(model)]
    for flags, order in (([], ["up", "patch", "patch", "up"]), (["--canary"], ["patch"])):
        plan = json.loads(subprocess.check_output(cmd + flags, text=True))
        assert plan["order"] == order and plan["stage_estimate_s_G"] <= .7*plan["budget_s"]
    (frozen / "bin/host-depth-server").write_bytes(b"changed bytes")
    try:
        h.frozen(frozen)
    except ValueError as e:
        assert "frozen bytes changed" in str(e)
    else:
        raise AssertionError("changed binary escaped pin gate")
print("PASS: real fixture parsers, frozen positive/negative controls, full and canary plans; no GPU commands executed")
