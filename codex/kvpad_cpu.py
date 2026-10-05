"""CPU n_kv_pad A/B on the in-process server fixture (MTP draft, Meta(CPU,CPU) optional).

Usage: kvpad_cpu.py OUTDIR SCENARIO META [BIN_DIR] ; PADS="0,4096" selects arms; first arm is the reference.
Scenarios: prefill (8.3k-token prompt, ub512), decode (2k prompt, 512 new tokens), long (33k prompt: Meta decode takes the n_kv >= 32768 mask-prefix path).
Identity = generated token IDs + final target logit bytes. Counts = graph_hit/graph_miss per phase from LLAMA_ROUND_TRACE.
Each process is nice 19, <= 8 threads, CUDA hidden.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
out = Path(sys.argv[1]); scenario = sys.argv[2]; meta = sys.argv[3]
bindir = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "build-cpu/bin"
pads = os.environ.get("PADS", "0,4096").split(",")
model = os.environ.get("CPU_TARGET_MODEL", "/mnt/scratch/work/kv-pad/untied-target.gguf")
SC = {"prefill": dict(words="8300", ub="512", batch="512", n="16"),
      "decode": dict(words="1990", ub="512", batch="512", n=os.environ.get("N_DECODE", "512")),
      "assert": dict(words="1000", ub="512", batch="512", n="4"),
      "long": dict(words="33000", ub="512", batch="512", n="16")}[scenario]
out.mkdir(parents=True, exist_ok=True)
rows = []
ref = None
for pad in pads:
    label = f"{scenario}-m{meta}-pad{pad}"
    trace = out / (label + ".jsonl")
    trace.unlink(missing_ok=True)
    path = out / (label + ".bin")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=os.environ.get("CPU_THREADS", "8"), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TMPDIR=str(ROOT / "build-tmp"), CPU_PREFILL_WORDS=SC["words"], CPU_UBATCH=SC["ub"], CPU_BATCH=SC["batch"],
               CPU_KV_PAD=pad, LLAMA_ROUND_TRACE=str(trace), LD_LIBRARY_PATH=str(bindir), CPU_TEMP=os.environ.get("CPU_TEMP", "1"), CPU_THREADS=os.environ.get("CPU_THREADS", "8"))
    for k, v in json.loads(os.environ.get("ENV_EXTRA", "{}")).items():
        env[k] = v
    with (out / (label + ".log")).open("w") as log:
        rc = subprocess.run(["nice", "-n", "19", str(bindir / "reuse-server-cpu"), model, str(path), meta, "3", "1729", SC["n"], "1", "0"],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200).returncode
    if rc != 0:
        rows.append({"label": label, "rc": rc}); print("M: FAIL", label, "rc", rc, flush=True); continue
    data = path.read_bytes()
    if ref is None:
        ref = data
    groups = {}
    for line in trace.read_text().splitlines():
        for ph in json.loads(line)["phases"]:
            big = ph["n_tokens"] > 6
            g = f'{ph["phase"][0]}_{"prefill" if big else "decode"}'
            d = groups.setdefault(g, {"phases": 0, "graph_hit": 0, "graph_miss": 0})
            d["phases"] += 1
            for c in ph["counters"]:
                if c["kind"] in ("graph_hit", "graph_miss"):
                    d[c["kind"]] += c["count"]
    ctx = [l.split("=")[1].strip() for l in (out / (label + ".log")).read_text().splitlines() if "llama_context: n_ctx  " in l or "llama_context: n_kv_pad" in l]
    rows.append({"label": label, "pad": pad, "sha256": hashlib.sha256(data).hexdigest(), "identical_to_ref": data == ref,
                 "bytes": len(data), "groups": groups, "ctx_lines": ctx})
    print("M:", json.dumps(rows[-1]), flush=True)
(out / f"{scenario}-m{meta}-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
