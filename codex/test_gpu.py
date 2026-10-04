#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile
import gpu_ab as h

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
root = Path(__file__).resolve().parent.parent
os.chdir(root)
fixtures = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
h.fixture_test(fixtures)
base = subprocess.check_output(["git", "rev-parse", "ff5611656"], text=True).strip()
with tempfile.TemporaryDirectory(dir=root / "build-tmp") as tmp:
    temp = Path(tmp)
    roots = []
    for arm, source in (("up", base), ("patch", "1"*40)):
        frozen = temp / arm; (frozen / "bin").mkdir(parents=True)
        (frozen / "bin/host-depth-server").write_bytes((arm + " CPU plan fixture, never execute").encode())
        (frozen / "SOURCE").write_text(source + "\n")
        (frozen / "bin/SHA256SUMS").write_text(h.sha(frozen / "bin/host-depth-server") + "  host-depth-server\n")
        (frozen / "SHA256SUMS").write_text("".join(h.sha(frozen / f) + "  " + f + "\n" for f in ("SOURCE", "bin/SHA256SUMS")))
        binary, pins = h.frozen(frozen)
        assert binary.name == "host-depth-server" and len(pins) == 3
        roots.append(frozen)
    model = temp / "model"; model.write_bytes(b"CPU plan fixture, never execute")
    vocab = temp / "vocab"; vocab.write_text("[]")
    cmd = ["python3", "codex/gpu_ab.py", "--up", str(roots[0]), "--patch", str(roots[1]),
           "--model", str(model), "--model-sha", h.sha(model), "--draft-vocab", str(vocab), "--draft-vocab-sha", h.sha(vocab)]
    for flags, order in (([], ["up", "patch", "patch", "up"]), (["--canary"], ["patch"]), (["--canary", "--canary-arm", "up"], ["up"])):
        plan = json.loads(subprocess.check_output(cmd + flags, text=True))
        assert plan["order"] == order and plan["stage_estimate_s_G"] <= .7*plan["budget_s"]
    result = subprocess.run(cmd + ["--execute-parent"], text=True, capture_output=True)
    assert result.returncode != 0 and "parent execution requires" in result.stdout
    result = subprocess.run(cmd + ["--canary", "--budget", "100"], text=True, capture_output=True)
    assert result.returncode != 0 and "70 percent" in result.stdout
    result = subprocess.run(["python3", "codex/gpu_ab.py", "--up", str(roots[1]), "--patch", str(roots[0]),
                             "--model", str(model), "--model-sha", h.sha(model)], text=True, capture_output=True)
    assert result.returncode == 6 and "wrong upstream" in result.stdout
    (roots[0] / "bin/host-depth-server").write_bytes(b"changed bytes")
    try:
        h.frozen(roots[0])
    except ValueError as e:
        assert "frozen bytes changed" in str(e)
    else:
        raise AssertionError("changed binary escaped pin gate")
print("PASS: recorded parsers, frozen controls, base SOURCE, parent guard, budget, full and canary plans")
