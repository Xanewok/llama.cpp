#!/usr/bin/env python3
# CPU-only checks for judge/gpu_ab.py. Never starts a server.
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.nice(19)
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import gpu_ab as h  # noqa: E402

h.fixture_test(Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures"))

# Native /completion stream shape (server-task.cpp to_json_non_oaicompat + send_final_response):
# partial chunks carry tokens/content; the streamed final chunk has tokens [] and content "".
t = lambda n, d: {"predicted_n": n, "predicted_ms": 10.0*n, "draft_n": d, "draft_n_accepted": d - 1, "prompt_n": 8192, "cache_n": 0}
stream = {"chunks": [{"content": "a", "tokens": [11], "stop": False, "timings": t(1, 5)},
                     {"content": "b", "tokens": [12, 13], "stop": False, "timings": t(3, 10)},
                     {"content": "", "tokens": [], "stop": True, "timings": t(3, 10)}]}
whole = {"chunks": [{"content": "ab", "tokens": [11, 12, 13], "stop": True, "timings": t(3, 10)}]}
assert h.identity(stream) == h.identity(whole), "streamed identity must equal the non-streamed reply"
try:
    h.identity({"chunks": [{"content": "", "tokens": [], "stop": True, "timings": t(0, 0)}]})
except ValueError:
    pass
else:
    raise AssertionError("empty token identity accepted")

assert h.loaded_outside(os.getpid(), here) == []
for reason, code in (("hazard: kernel fault since cutoff", 1), ("budget expired", 1), ("port busy: x", 4),
                     ("frozen bytes changed: x", 6), ("INVALID: depth 8192: token or acceptance identity failed", 6),
                     ("INVALID: depth 90112: insufficient full-K interior intervals: 3 < 40", 8)):
    assert h.exit_code(reason) == code and h.exit_code(reason) not in (3, 7), reason

with tempfile.TemporaryDirectory(dir=here) as tmp:
    temp = Path(tmp)
    roots = {}
    for arm, src, payload in (("up", "c23324d5f9f0fc00ad3f2e6441e39ba061f41544", b"up"), ("patch", "59c721c985f48bd2736deba7ddfec322ff0fa86e", b"patch")):
        r = temp / arm; (r / "bin").mkdir(parents=True)
        (r / "bin/host-depth-server").write_bytes(b"never executed " + payload)
        (r / "SOURCE").write_text(src + "\n")
        (r / "bin/SHA256SUMS").write_text(h.sha(r / "bin/host-depth-server") + "  host-depth-server\n")
        (r / "SHA256SUMS").write_text("".join(h.sha(r / f) + "  " + f + "\n" for f in ("SOURCE", "bin/SHA256SUMS")))
        roots[arm] = r
    model = temp / "model"; model.write_bytes(b"plan fixture")
    vocab = temp / "vocab.json"; vocab.write_text("[]")
    base = [sys.executable, str(here / "gpu_ab.py"), "--model", str(model), "--model-sha", h.sha(model),
            "--draft-vocab", str(vocab), "--draft-vocab-sha", h.sha(vocab)]
    plan = json.loads(subprocess.check_output(base + ["--up", str(roots["up"]), "--patch", str(roots["patch"])], text=True))
    assert plan["order"] == ["up", "patch", "patch", "up"] and str(vocab.resolve()) in plan["pins"]
    same = subprocess.run(base + ["--up", str(roots["up"]), "--patch", str(roots["up"])], capture_output=True, text=True)
    assert same.returncode == 6 and "not distinct" in same.stdout, (same.returncode, same.stdout)
    vocab.write_text("[1]")
    changed = subprocess.run(base + ["--up", str(roots["up"]), "--patch", str(roots["patch"])], capture_output=True, text=True)
    assert changed.returncode == 6 and "draft vocab changed" in changed.stdout, (changed.returncode, changed.stdout)
print("PASS: streamed identity, A/A rejection, vocab pin, loaded-library check, exit codes avoid 3/7; no GPU commands")
