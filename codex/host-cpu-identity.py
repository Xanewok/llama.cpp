"""Compare deterministic CPU target tokens, draft counters and draft candidates against stack6."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf"
OUT = ROOT / "codex/host-results/identity"
OUT.mkdir(parents=True, exist_ok=True)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
GATE = ROOT / "codex/host-gate.sh"
results = []


def draft_candidates(log):
    return re.findall(r"draft candidate\s+\d+, pos\s+\d+:\s+(\d+)\s+\(\s*([\d.]+)\)", log)


def run(arm, meta, np, seed):
    name = f"{arm}-m{meta}-np{np}-s{seed}"
    bindir = ROOT / ("codex/base-cpu" if arm == "base" else "build-cpu/bin")
    path = OUT / (name + ".json")
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), LLAMA_HOST_TRIMS_PROOF="1", GGML_SCHED_INPUT_STAGING="1")
    with (OUT / (name + ".log")).open("w") as log:
        subprocess.run(["bash", str(GATE), str(ROOT / "build-cpu/bin/host-identity"), MODEL, str(path),
                        str(meta), str(np), str(seed), "48"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(path.read_text())
    data["stats"].sort(key=lambda item: item["slot"])
    drafts = draft_candidates((OUT / (name + ".log")).read_text(errors="replace"))
    assert drafts, "no draft candidates captured"
    data["draft_candidates"] = drafts
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    results.append(dict(name=name, sha256=digest, tokens_M=sum(map(len, data["tokens"])), draft_candidates_M=len(drafts)))
    return data


for meta in (0, 1):
    for np in (1, 2):
        for seed in (1729, 2718):
            base = run("base", meta, np, seed)
            patch = run("patch", meta, np, seed)
            assert base == patch, (meta, np, seed, "identity mismatch")
            changed = json.loads(json.dumps(patch))
            changed["tokens"][0][0] ^= 1
            assert changed != base, "comparison control escaped"
            print(f"CPU PASS meta={meta} np={np} seed={seed}: target tokens, MTP counters and draft candidates", flush=True)
(ROOT / "codex/host-results/cpu-identity.json").write_text(json.dumps({"status": "PASS", "runs_M": len(results),
    "cases_M": len(results)//2, "comparison_negative_controls_M": len(results)//2, "results": results}, indent=2) + "\n")
