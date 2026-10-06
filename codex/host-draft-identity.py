"""Use existing batch debug output to compare every decoded token ID, including MTP proposals."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "codex/host-results/draft-identity"
OUT.mkdir(parents=True, exist_ok=True)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
results = []
for meta in (0, 1):
    for np in (1, 2):
        expected = None
        for arm in ("base", "patch"):
            name = f"{arm}-m{meta}-np{np}"
            bindir = ROOT / ("codex/base-cpu" if arm == "base" else "build-cpu/bin")
            output = OUT / (name + ".json")
            env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), LLAMA_BATCH_DEBUG="1", GGML_SCHED_INPUT_STAGING="1", HOST_KV_Q8="1")
            with (OUT / (name + ".log")).open("w") as log:
                subprocess.run(["bash", str(ROOT / "codex/host-gate.sh"), str(ROOT / "build-cpu/bin/host-identity"),
                    "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf", str(output), str(meta), str(np), "1729", "48"],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            decoded = re.findall(r"ubatch_print:\s+\d+:\s+id\s*=\s*(-?\d+)", (OUT / (name + ".log")).read_text(errors="replace"))
            assert len(decoded) > 48*np, "batch debug did not include prompt and draft decodes"
            data = json.loads(output.read_text())
            data["stats"].sort(key=lambda row: row["slot"])
            record = dict(decoded_ids=decoded, completion=data)
            if expected is None:
                expected = record
            else:
                assert record == expected, (meta, np, "decoded target/draft token identity mismatch")
                bad = list(decoded)
                bad[-1] = str(int(bad[-1]) ^ 1)
                assert bad != expected["decoded_ids"], "decoded-ID comparison control escaped"
            results.append(dict(name=name, decoded_ids_M=len(decoded)))
        print(f"CPU PASS every decoded token ID: meta={meta}, np={np}", flush=True)
(ROOT / "codex/host-results/draft-identity.json").write_text(json.dumps({"status": "PASS", "target_kv": "q8_0", "cases_M": len(results)//2,
    "runs_M": len(results), "negative_controls_M": len(results)//2, "results": results}, indent=2) + "\n")
