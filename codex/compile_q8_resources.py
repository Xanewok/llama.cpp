#!/usr/bin/env python3
"""Compile two existing CUDA instances with resource logging; do not execute them."""
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build-cuda-q8-1row/ggml/src/ggml-cuda"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    make = (BUILD / "CMakeFiles/ggml-cuda.dir/flags.make").read_text()
    flags = []
    for name in ("CUDA_DEFINES", "CUDA_INCLUDES", "CUDA_FLAGS"):
        value = re.search(r"^" + name + r" = (.*)$", make, re.M).group(1)
        flags.extend(shlex.split(value))
    files = ("fattn-vec-instance-q8_0-q8_0", "fattn-mma-f16-instance-q8_0-ncols1_1-ncols2_8", "fattn-mma-f16-instance-q8_0-ncols1_4-ncols2_8")
    for name in files:
        if "--parse-only" in sys.argv:
            continue
        cmd = ["python3", str(ROOT / "codex/ram_build.py"), "/usr/local/cuda-13.2/bin/nvcc", *flags,
               "-Xptxas=-v", "-c", str(ROOT / f"ggml/src/ggml-cuda/template-instances/{name}.cu"),
               "-o", str(ROOT / f"build-tmp/{name}.o")]
        with (ROOT / f"codex/logs/{name}-resources.log").open("w") as log:
            subprocess.run(cmd, cwd=BUILD, stdout=log, stderr=log, check=True)
    records = []
    for name in files:
        log = (ROOT / f"codex/logs/{name}-resources.log").read_text()
        for match in re.finditer(r"Compiling entry function '([^']+)'(.*?)(?=ptxas info\s*: Compiling entry function|\Z)", log, re.S):
            symbol, resource = match.groups()
            demangled = subprocess.check_output(["c++filt", symbol], text=True).strip()
            if "flash_attn_ext_vec<256, 1," in demangled or any(f"flash_attn_ext_f16<256, 256, {n}, 8," in demangled for n in (1, 4)):
                records.append(demangled + "\n" + resource.strip())
    assert records
    (ROOT / "codex/q8-cuda-resources.txt").write_text("M compile-only resources, sm_120a; runtime occupancy is unmeasured.\n" + "\n".join(records) + "\n")


if __name__ == "__main__":
    main()
