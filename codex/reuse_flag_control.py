"""Disable the retained-graph flag: byte identity must remain, the reuse contract must fail."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
out = ROOT / "codex/reuse-results/flag-control"
out.mkdir(parents=True, exist_ok=True)
records = []
reference = None
for enabled in (1, 0):
    trace, data = out / f"{enabled}.jsonl", out / f"{enabled}.bin"
    if trace.exists(): trace.unlink()
    env = dict(os.environ, LLAMA_MTP_GRAPH_REUSE=str(enabled), LLAMA_ROUND_TRACE=str(trace), OMP_NUM_THREADS="8")
    with (out / f"{enabled}.log").open("w") as log:
        subprocess.run(["nice", "-n", "19", str(ROOT / "build-cpu/bin/reuse-cpu"), "-", "1", "3", "1729", "12", str(data), "1"],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    value = data.read_bytes()
    if reference is None: reference = value
    assert value == reference, "flag changed token/id/q/logit bytes"
    rows = [json.loads(line) for line in trace.read_text().splitlines()][3:]
    hits = sum(c["count"] for row in rows for phase in row["phases"] if phase["phase"] in ("C", "D1")
               for c in phase["counters"] if c["kind"] == "graph_hit")
    assert hits == 2 * len(rows) if enabled else hits == 0, (enabled, hits)
    records.append({"enabled": enabled, "C_D1_hits_M": hits, "rounds_M": len(rows)})
(ROOT / "codex/reuse-flag-results.json").write_text(json.dumps(records, indent=2) + "\n")
print("M PASS: disable flag fails retained C/D1 hit contract; all output bytes identical")
