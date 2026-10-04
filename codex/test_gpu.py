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
patch = subprocess.check_output(["git", "rev-parse", "c211ca41d"], text=True).strip()
with tempfile.TemporaryDirectory(dir=root / "build-tmp") as tmp:
    temp = Path(tmp)
    roots = []
    for arm, source in (("up", base), ("patch", patch), ("other", "1"*40)):
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
    def synthetic(depth, rounds=50, tokens=True):
        chunks, n = [], 0
        for r in range(1, rounds + 1):
            for t in range(3):
                n += 1
                chunks.append({"tokens": [n] if tokens else [], "timings": {"predicted_ms": 40.0*r + t, "predicted_n": n,
                    "draft_n": 5*r, "draft_n_accepted": 2*r, "prompt_n": depth, "cache_n": 0}})
        chunks.append({"stop": True, "tokens": [], "timings": dict(chunks[-1]["timings"])})
        return {"request": {}, "chunks": chunks}
    ref = temp / "ref"; ref.mkdir()
    for depth in (8192, 90112):
        (ref / f"depth-{depth}.json").write_text(json.dumps(synthetic(depth)))
    plan = json.loads(subprocess.check_output(cmd + ["--canary", "--identity-reference", str(ref)], text=True))
    assert plan["order"] == ["patch"] and all(str((ref / f"depth-{d}.json").resolve()) in plan["pins"] for d in (8192, 90112))
    (ref / "depth-90112.json").write_text(json.dumps(synthetic(90112, tokens=False)))
    result = subprocess.run(cmd + ["--canary", "--identity-reference", str(ref)], text=True, capture_output=True)
    assert result.returncode == 6 and "missing final token identity" in result.stdout
    result = subprocess.run(cmd + ["--execute-parent"], text=True, capture_output=True)
    assert result.returncode != 0 and "parent execution requires" in result.stdout
    result = subprocess.run(cmd + ["--canary", "--budget", "100"], text=True, capture_output=True)
    assert result.returncode != 0 and "70 percent" in result.stdout
    result = subprocess.run(["python3", "codex/gpu_ab.py", "--up", str(roots[1]), "--patch", str(roots[0]),
                             "--model", str(model), "--model-sha", h.sha(model)], text=True, capture_output=True)
    assert result.returncode == 6 and "wrong upstream" in result.stdout
    result = subprocess.run(["python3", "codex/gpu_ab.py", "--up", str(roots[0]), "--patch", str(roots[2]),
                             "--model", str(model), "--model-sha", h.sha(model)], text=True, capture_output=True)
    assert result.returncode == 6 and "wrong patch" in result.stdout
    assert h.exit_code("INVALID: depth 90112: token or acceptance identity failed") == 6
    assert h.exit_code("INVALID: depth 8192: insufficient full-K interior intervals: 3 < 40") == 8
    assert h.exit_code("port busy: [Errno 98]") == 4
    (roots[0] / "bin/host-depth-server").write_bytes(b"changed bytes")
    try:
        h.frozen(roots[0])
    except ValueError as e:
        assert "frozen bytes changed" in str(e)
    else:
        raise AssertionError("changed binary escaped pin gate")
print("PASS: recorded parsers, frozen controls, base and patch SOURCE, exit codes, parent guard, budget, full and canary plans")
