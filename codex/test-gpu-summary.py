#!/usr/bin/env python3
import json
import pathlib
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="summary-check-", dir=root.parent / "build-cpu") as temp:
    out = pathlib.Path(temp)
    for k in (3, 4, 5):
        for arm in ("plain", "chain"):
            for boot in (1, 2):
                for kind in ("tool", "think", "long"):
                    sample = {"sha256": "identical", "timings": {"predicted_ms": 1000 if arm == "plain" else 900,
                              "draft_n": 30, "draft_n_accepted": 20}, "counters": {"spec_decode_num_drafts_total": 30/k,
                              "spec_decode_num_draft_tokens_total": 30, "spec_decode_num_accepted_tokens_total": 20}}
                    (out / f"{arm}{k}-{boot}-{kind}.json").write_text(json.dumps(sample))
                (out / f"{arm}{k}-{boot}.memory.csv").write_text(f"GPU-a, {1000 if arm == 'plain' else 1067}\nGPU-b, {1000 if arm == 'plain' else 1067}\n")
    args = ["python3", str(root / "gpu_summary.py"), str(out)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "k=3 tool chain - plain: -10.000000 ms/round" in result.stdout
    assert "chain - plain memory: 67.0 MiB" in result.stdout
    broken = out / "chain3-2-tool.json"
    row = json.loads(broken.read_text())
    row["sha256"] = "changed"
    broken.write_text(json.dumps(row))
    assert subprocess.run(args, capture_output=True).returncode != 0
    print("M: synthetic summary oracle and changed-completion negative control PASS")
