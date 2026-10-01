#!/usr/bin/env python3
"""Cold CUDA library build comparison. Does not execute a CUDA program."""
import json
import os
from pathlib import Path
import subprocess
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(__file__).resolve().parent.parent
jobs = int(os.environ.get("BUILD_JOBS", "4"))
assert 1 <= jobs <= 8
results = []
for profile in ("single", "default", "all-quants"):
    for arm in ("master", "scope"):
        source = root / "build-master-src" if arm == "master" else root
        build = root / f"build-cost-{profile}-{arm}"
        assert not build.exists(), f"Cold-build path already exists: {build}"
        cfg = ["cmake", "-S", str(source), "-B", str(build), "-DGGML_CUDA=ON",
               "-DCUDAToolkit_ROOT=/usr/local/cuda-13.2", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc",
               "-DGGML_CUDA_NCCL=OFF", "-DGGML_CCACHE=OFF", "-DGGML_NATIVE=OFF",
               "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
               "-DCMAKE_CUDA_FLAGS=--threads=1", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"]
        if profile == "single":
            cfg.append("-DCMAKE_CUDA_ARCHITECTURES=120")
        if profile == "all-quants":
            cfg.append("-DGGML_CUDA_FA_ALL_QUANTS=ON")
        log = root / "codex" / f"cost-{profile}-{arm}.log"
        with log.open("w") as stream:
            subprocess.run(cfg, stdout=stream, stderr=subprocess.STDOUT, check=True)
            start = time.monotonic()
            subprocess.run(["cmake", "--build", str(build), "--target", "ggml-cuda", f"-j{jobs}"], stdout=stream, stderr=subprocess.STDOUT, check=True)
        lib = build / "bin/libggml-cuda.so"
        result = {"profile": profile, "arm": arm, "seconds_M": time.monotonic()-start, "bytes_M": lib.stat().st_size, "jobs": jobs, "configure": cfg}
        results.append(result)
        (root / "codex/build-costs.json").write_text(json.dumps(results, indent=2)+"\n")
        print(json.dumps(result), flush=True)
