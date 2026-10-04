"""CPU server-context identity against unmodified 98c695111. Never load CUDA."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("LLAMACPP_TEST_MODELFILE", "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf"))
OUT = ROOT / "codex/host-results/identity"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert MODEL.is_file()
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
for build in ("build-baseline", "build-cpu"):
    subprocess.run(["g++", "-std=c++17", "-O2", "-pthread", "-DGGML_SHARED", "-DLLAMA_SHARED",
                    "-Itools/server", "-Icommon", "-Iinclude", "-Iggml/include", "-Ivendor", "-Itools/mtmd", "-I.",
                    "codex/reuse-server-cpu.cpp", f"{build}/tools/server/libserver-context.a", f"{build}/common/libllama-common-base.a",
                    f"-L{build}/bin", f"-Wl,-rpath,{ROOT}/{build}/bin", "-lllama-common", "-lmtmd", "-lllama", "-lggml", "-lggml-cpu", "-lggml-base",
                    "-o", f"{build}/bin/host-server-cpu"], check=True)

def counters(path):
    records = json.loads(path.read_text())
    return [(r["chunks"][-1]["timings"]["draft_n"], r["chunks"][-1]["timings"]["draft_n_accepted"]) for r in records]

def identical(a, b):
    return a[0] == b[0] and a[1] == b[1]

results = []
for meta in (0, 1):
    for temperature, seed in ((0, 1729), (1, 1729), (1, 2718)):
        expected = None
        for arm, build in (("up", "build-baseline"), ("patch", "build-cpu")):
            label = f"m{meta}-t{temperature}-s{seed}-{arm}"
            path = OUT / (label + ".bin")
            env = dict(os.environ, CPU_TEMP=str(temperature), GGML_SCHED_INPUT_STAGING="1",
                       OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
            start = time.monotonic()
            with (OUT / (label + ".log")).open("w") as log:
                run = subprocess.run([str(ROOT / build / "bin/host-server-cpu"), str(MODEL), str(path), str(meta), "5", str(seed), "32", "2", "0"],
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                raise RuntimeError(f"{label}: exit {run.returncode}; see {OUT / (label + '.log')}")
            value = (path.read_bytes(), counters(Path(str(path) + ".stats.json")))
            if expected is None:
                expected = value
            assert identical(expected, value), f"{label}: tokens, final logits or draft counts differ"
            assert all(n > 0 for n, accepted in value[1]), f"{label}: MTP did not draft"
            results.append({"label": label, "bytes_M": len(value[0]), "sha256": hashlib.sha256(value[0]).hexdigest(),
                            "draft_counts_M": value[1], "wall_s_M": time.monotonic() - start})
        corrupted = bytearray(expected[0])
        corrupted[8] ^= 1
        assert not identical(expected, (bytes(corrupted), expected[1]))
        changed = copy.deepcopy(expected[1])
        changed[0] = (changed[0][0] + 1, changed[0][1])
        assert not identical(expected, (expected[0], changed))
        (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"PASS M: meta={meta} temp={temperature} seed={seed}; token/logit bytes and draft counts; negative controls", flush=True)
