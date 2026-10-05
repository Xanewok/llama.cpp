#!/usr/bin/env python3
"""Compile and inspect the q8 instances offline; never load a CUDA backend."""
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    build = ROOT / "codex/build-fix2/ggml/src/ggml-cuda"
    make = (build / "CMakeFiles/ggml-cuda.dir/flags.make").read_text()
    parts = {line.split(" = ")[0]: line.split(" = ")[1] for line in make.splitlines() if " = " in line}
    flags = shlex.split(parts["CUDA_DEFINES"]) + ["--options-file", str(build / "CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp")] + shlex.split(parts["CUDA_FLAGS"])
    obj = ROOT / "build-tmp/q8-resources.o"
    source = ROOT / "ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-q8_0-ncols1_1-ncols2_8.cu"
    command = ["/usr/local/cuda-13.2/bin/nvcc", "-forward-unknown-to-host-compiler", *flags, "-Xptxas=-v", "-c", str(source), "-o", str(obj)]
    (ROOT / "codex/logs/fix2-resource-command.txt").write_text(shlex.join(command) + "\n")
    with (ROOT / "codex/logs/fix2-ptxas-original.log").open("w") as log:
        subprocess.run(command, cwd=build, stdout=log, stderr=log, check=True)
    dump = subprocess.check_output(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-sass", str(obj)], text=True)
    (ROOT / "codex/logs/fix2-sass-original.txt").write_text(dump)
    for d in (64, 96, 128, 256):
        symbol = f"_Z18flash_attn_ext_f16ILi{d}ELi{d}ELi1ELi8ELb0ELb0ELb0EL9ggml_type8"
        body = next(part for part in dump.split("Function : ") if symbol in part)
        counts = {op: len(re.findall(r"\b" + op + r"(?:\.|\s)", body)) for op in ("LDG", "LDGSTS", "LDS", "STS", "LDL", "STL")}
        print(f"M: D={d}, static SASS opcodes: {counts}")


if __name__ == "__main__":
    main()
