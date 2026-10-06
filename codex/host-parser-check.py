"""Replay parsers on recorded GPU artifacts without starting a server or a GPU tool."""
import copy
import json
import os
from pathlib import Path
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parent.parent
EVAL = Path.home() / "repos/qwen38-eval"
sys.path.insert(0, str(EVAL))
from harness import depth, lib

fixtures = EVAL / "orchestration/gpuq/fixtures"
depth.fixture_test(fixtures)
recorded = EVAL / "results/2026-10-04-duo-np2-depth/full-a3/0-up"
x = lib.read_experiment(EVAL / "harness/experiments/duo-np2-depth.toml")
x["depth"]["depths"] = [8192]
results = {}
failures = depth.evaluate_streams(x, recorded, "up", results)
assert not failures, failures
record = json.loads((recorded / "depth-8192.json").read_text())
window = depth.overlap(record["streams"], x["depth"]["min_overlap"])
ident = lib.token_identity(record["streams"][0])
negative = 0


def rejects(call):
    global negative
    try:
        call()
    except lib.Fail:
        negative += 1
    else:
        raise AssertionError("negative control escaped")


altered = copy.deepcopy(record["streams"])
for chunk in altered[1]["chunks"]:
    if "t" in chunk:
        chunk["t"] += 10**12
rejects(lambda: depth.overlap(altered, 0.8))
rejects(lambda: lib.steady_intervals(record["streams"][0], 100000, 5))
changed_ident = copy.deepcopy(ident)
changed_ident[0][0] ^= 1
settled = {8192: {"identity": ident}}
rejects(lambda: depth.settle(settled, "cross-arm", "patch", 8192, changed_ident))
rejects(lambda: lib.token_identity({"chunks": [{}]}))
bad = copy.deepcopy(record["streams"][0])
bad["error"] = "HTTP 500 recorded control"
rejects(lambda: depth.gate(x["depth"], bad, lambda _: None))
assert lib.min_free_vram(recorded / "smi.csv", x["devices"])
assert not re.search(lib.XID_RE, (recorded / "journal.txt").read_text())
for pattern in (r"host_trims: kv_seq_rm_indexed engaged\b", r"host_trims: mtp_verify_bulk rows=[1-9]\d*\b"):
    log = (recorded / "server.log").read_text()
    rejects(lambda: lib.verify_log_patterns(log, [pattern], "missing new engagement proof"))
    # Real CPU executions below must produce the new signatures; recorded GPU logs cannot prove new code engagement.
    cpu_logs = list((ROOT / "codex/host-results/identity").glob("patch-*.log"))
    assert cpu_logs, "CPU identity logs missing"
    assert any(re.search(pattern, path.read_text()) for path in cpu_logs), pattern
(ROOT / "codex/host-results/parser-check.json").write_text(json.dumps({
    "artifact": str(recorded), "depth_M": 8192, "overlap": window,
    "negative_controls_M": negative, "fixtures": str(fixtures), "status": "PASS"
}, indent=2) + "\n")
print(f"PASS recorded GPU parser checks; negative controls M={negative}")
