"""Exercise the owned Meta path with test-only synchronous CPU transfer hooks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
out = ROOT / "codex/input-results/owned"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CPU_TARGET_MODEL=str(ROOT / "build-tmp/reuse-untied-target.gguf"))
results = []
for fixture in ("synthetic", "real"):
    model = "-" if fixture == "synthetic" else os.environ.get("MODEL", str(Path.home() / "models/Qwen3.5-4B-UD-Q4_K_XL.gguf"))
    for k in (3, 5):
        for seed in range(1, 5):
            label = f"{fixture}-k{k}-s{seed}"
            path = out / (label + ".bin")
            trace = out / (label + ".jsonl")
            if trace.exists():
                trace.unlink()
            child_env = dict(env, LLAMA_ROUND_TRACE=str(trace))
            with (out / (label + ".log")).open("w") as log:
                subprocess.run(["nice", "-n", "19", str(ROOT / "build-cpu/bin/input-cpu"), model, "1", str(k), str(seed), "12", str(path), "1"],
                               env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            reference = ROOT / f"codex/input-results/{fixture}/m1-k{k}-s{seed}-1-up.bin"
            data = path.read_bytes()
            assert data == reference.read_bytes(), (label, "owned token/id/q/logit bytes differ")
            rows = [json.loads(line) for line in trace.read_text().splitlines()][3:]
            fences = sum(c["count"] for row in rows for phase in row["phases"] if phase["phase"] in ("C", "D1")
                         for c in phase["counters"] if c["kind"] == "input_fence_sync")
            staged = sum(c["count"] for row in rows for phase in row["phases"] if phase["phase"] in ("C", "D1")
                         for c in phase["counters"] if c["kind"] == "input_fence_staged")
            assert staged == 2 * len(rows), (label, "owned C/D1 fence bypass missed", staged)
            assert fences == 0, (label, "owned C/D1 retained a host fence", fences)
            evidence = (out / (label + ".log")).read_text().splitlines()[-1]
            results.append({"fixture": fixture, "k": k, "seed": seed, "bytes_M": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dispatch": evidence, "C_D1_host_fences_M": fences, "C_D1_staged_decisions_M": staged})
            print(f"M: {label} owned token/id/q/logit identity", flush=True)
(ROOT / "codex/owned-input-results.json").write_text(json.dumps(results, indent=2) + "\n")
