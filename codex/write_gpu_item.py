"""Write only the worktree draft; the parent owns the overnight item directory."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
files = json.loads(subprocess.check_output(["bash", str(ROOT / "codex/gpu_ab.sh"), "--pins"], text=True))
script = str(ROOT / "codex/gpu_ab.sh")
output = str(ROOT / "codex/runs/stack3-identity")
q = json.dumps
text = """id = "stack3-identity"
title = "Swift S5 np2: stack2 vs stack3 defaults vs staged inputs, exact output gate"
kind = "measure"
ab = true
retry = "fresh"
budget_s = 1200
est_s = 780
startup_grace_s = 90
signatures = [
  { class = "hazard", regex = "\\\\bXid\\\\b|fallen off the bus" },
  { class = "timeout", regex = "queue wall budget" },
  { class = "integrity", regex = "identity failed|frozen.*changed|inputs changed|model SHA changed" },
  { class = "insufficient_data", regex = "insufficient steady|generated length|did not stream|missing authoritative rounds|did not overlap" },
  { class = "deterministic", regex = "boot validation failed|aggregate counters disagree|prompt size/cache gate failed" },
]
"""
for section, execute, canary, budget in (("run", True, False, 960), ("preflight", False, False, 960), ("canary", True, True, 240)):
    cmd = ["bash", script]
    if execute: cmd.append("--execute")
    if canary: cmd.append("--canary")
    cmd += ["--budget", str(budget), "--port", "8403"]
    if execute: cmd += ["--out", output + ("/canary-a{attempt}" if canary else "/full-a{attempt}")]
    text += "\n[" + section + "]\ncmd = " + q(cmd) + "\ncwd = " + q(str(ROOT)) + "\n"
    text += "env = " + ("{ RUN_GPU = \"1\" }" if execute else "{}") + "\n"
    if section == "run": text += "ports = [8403]\n"
    else: text += "timeout_s = " + str(240 if canary else 120) + "\n"
text += "\n[outputs]\npaths = [" + q(output) + "]\n"
text += "\n[fix]\neditable = []\nprotected = " + q(["identity", "minimum_steady_rounds", "768", "SHA256SUMS", "32768"]) + "\n"
text += "\n[pins]\nfiles = [\n" + "".join("  " + q(path) + ",\n" for path in files) + "]\n"
(ROOT / "codex/stack3-identity.toml").write_text(text)
print("M: draft written in worktree, pinned files", len(files))
