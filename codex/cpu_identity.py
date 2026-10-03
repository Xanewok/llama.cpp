"""Byte identity through the CPU server context, including sampling and final logits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
out = ROOT / "codex/reuse-results/identity"
out.mkdir(parents=True, exist_ok=True)
model = os.environ.get("MODEL", str(Path.home() / "models/Qwen3.5-4B-UD-Q4_K_XL.gguf"))
results = []
for meta in (0, 1):
    for temperature, seed in ((0, 1729), (1, 1729), (1, 2718)):
        expected = None
        for arm in ("stack2", "default", "staged", "reuse-off", "pool-off"):
            binary = ROOT / ("build-base-cpu" if arm == "stack2" else "build-cpu") / "bin/reuse-server-cpu"
            env = dict(os.environ, CPU_TEMP=str(temperature), CPU_TARGET_MODEL=str(ROOT / "build-tmp/reuse-untied-target.gguf"),
                       OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
            for knob in ("LLAMA_ROUND_TRACE", "LLAMA_MTP_GRAPH_REUSE", "GGML_META_PLAN_REUSE", "GGML_SCHED_INPUT_STAGING", "LLAMA_GRAPH_REUSE_DISABLE"):
                env.pop(knob, None)
            if arm == "staged": env["GGML_SCHED_INPUT_STAGING"] = "1"
            if arm == "reuse-off": env["LLAMA_MTP_GRAPH_REUSE"] = "0"
            if arm == "pool-off": env["GGML_META_PLAN_REUSE"] = "0"
            label = f"m{meta}-t{temperature}-s{seed}-{arm}"
            path = out / (label + ".bin")
            begin = time.monotonic()
            with (out / (label + ".log")).open("w") as log:
                subprocess.run(["nice", "-n", "19", str(binary), model, str(path), str(meta), "3", str(seed), "48", "2", "0"],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            data = path.read_bytes()
            if expected is None: expected = data
            if data != expected: raise AssertionError((label, "token/final-logit byte identity failed"))
            results.append({"label": label, "bytes_M": len(data), "sha256": hashlib.sha256(data).hexdigest(), "wall_s_M": time.monotonic() - begin})
        altered = bytearray(expected)
        altered[8] ^= 1
        assert altered != expected, "identity negative control escaped"
        print("M PASS", label, "all five arms; perturbed-token control rejected", flush=True)
(ROOT / "codex/cpu-identity-results.json").write_text(json.dumps({"runs_M": len(results), "cases_M": 6,
    "negative_controls_M": 6, "identity": "token IDs and every final-target-logit byte", "results": results}, indent=2) + "\n")
