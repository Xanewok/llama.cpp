"""Count scheduler transfers with synchronous owned CPU callbacks; this is not GPU timing."""
import collections
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "codex/host-results"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def rows(path, kind):
    result = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        record = json.loads(line)
        for phase in record.get("phases", []):
            if not 0 < phase["n_tokens"] <= 8:
                continue
            count = sum(c["count"] for c in phase.get("counters", []) if c["kind"] == kind)
            if count:
                result[(phase["phase"], phase["n_tokens"])].append(count)
    return result


recorded = Path.home() / "repos/qwen38-eval/results/2026-10-04-duo-np2-depth/canary-a3/1-patch/round-trace.jsonl"
assert rows(recorded, "input_owned"), "recorded GPU trace parser did not find staged inputs"
trace = OUT / "input-count.jsonl"
assert not trace.exists(), "use a fresh counter trace"
env = dict(os.environ, HOST_INPUT_COUNT="1", LLAMA_ROUND_TRACE=str(trace), GGML_SCHED_INPUT_STAGING="1",
           LD_LIBRARY_PATH=str(ROOT / "build-cpu/bin"))
output = OUT / "input-count-identity.json"
with (OUT / "input-count.log").open("w") as log:
    subprocess.run(["bash", str(ROOT / "codex/host-gate.sh"), str(ROOT / "build-cpu/bin/host-identity"),
        "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf", str(output), "1", "2", "1729", "48"],
        env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
base = json.loads((OUT / "identity/patch-m1-np2-s1729.json").read_text())
actual = json.loads(output.read_text())
for item in (base, actual):
    item["stats"].sort(key=lambda row: row["slot"])
assert base == actual, "owned CPU staging proxy changed tokens or draft counters"
counts = rows(trace, "input_cpu_proxy")
assert counts, "owned CPU transfers did not engage"
result = [dict(phase=p, tokens_M=n, total_rank_transfers_M=sorted(set(values)), observations_M=len(values))
          for (p, n), values in sorted(counts.items())]
(OUT / "input-count.json").write_text(json.dumps({"unit": "CPU Meta(CPU,CPU) scheduler transfers, both ranks summed",
    "model": "Qwen3.5-4B", "status": "PASS", "results": result, "recorded_parser_artifact": str(recorded)}, indent=2) + "\n")
print(json.dumps(result, indent=2))
