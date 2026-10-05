#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "codex/q8-probe4"


def quoted(value):
    return json.dumps(str(value))


pins = [ROOT / "tests/test-backend-ops.cpp", ROOT / "ggml/include/ggml.h"]
pins += [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "item.toml"]
for arm in ("q8-general-base", "q8-general-fix"):
    freeze = ROOT / "codex/frozen" / arm
    pins += [freeze / "SOURCE"] + list((freeze / "build/bin").iterdir())

text = f'''id = "q8-general-fa-ab4"
title = "Grouped q8_0 decode: all block-valid small heads and GQA ratios, both GPUs, frozen ABBA"
kind = "measure"
budget_s = 1800
est_s = 1000
priority = 40
ab = true
retry = "fresh"
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = [
    {{ class = "hazard", regex = "\\\\bXid\\\\b|fallen off the bus" }},
    {{ class = "integrity", regex = "frozen bytes changed|ggml libs resolve outside frozen dir|FA source shape mismatch" }},
    {{ class = "timeout", regex = "budget expired" }},
]

[run]
cmd = ["bash", {quoted(HERE / "run.sh")}, "--run"]
cwd = {quoted(ROOT)}
env = {{ CUDA_VISIBLE_DEVICES = "0,1", BUDGET_S = "1500", OUT = {quoted(ROOT / "codex/runs/q8-probe4/full-a{attempt}")} }}
ports = []

[preflight]
cmd = ["bash", {quoted(HERE / "run.sh")}, "--preflight"]
env = {{ CUDA_VISIBLE_DEVICES = "", OUT = {quoted(ROOT / "codex/runs/q8-probe4/preflight")} }}
timeout_s = 60

[canary]
cmd = ["bash", {quoted(HERE / "run.sh")}, "--canary"]
env = {{ CUDA_VISIBLE_DEVICES = "0", BUDGET_S = "280", OUT = {quoted(ROOT / "codex/runs/q8-probe4/canary-a{attempt}")} }}
timeout_s = 300

[outputs]
paths = [{quoted(ROOT / "codex/runs/q8-probe4/full-a{attempt}")}]

[fix]
editable = []
protected = ["frozen", "budget"]

[pins]
files = [
'''
text += "".join(f"  {quoted(p)},\n" for p in sorted(set(pins))) + "]\n"
(HERE / "item.toml").write_text(text)
