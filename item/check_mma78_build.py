#!/usr/bin/env python3
"""Reject missing MMA78 kernels, excessive registers, stack frames or local-memory traffic."""
from pathlib import Path
import re
import sys


def check(build, sass):
    entries = re.split(r"ptxas info\s+: Compiling entry function ", build)[1:]
    kernels = [entry for entry in entries if "mul_mat_vec_q_mma" in entry.splitlines()[0]]
    if len(kernels) != 2:
        raise ValueError(f"expected two MMA78 ptxas entries, found {len(kernels)}; use a fresh verbose build")
    for entry in kernels:
        regs = re.search(r"Used (\d+) registers", entry)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", entry)
        if not regs or int(regs[1]) > 128 or not frame or any(int(value) for value in frame.groups()):
            raise ValueError(f"MMA78 register/spill gate failed: {entry}; reduce live state and rebuild")
        print(entry.strip())
    functions = re.split(r"Function\s*:\s*", sass)[1:]
    kernels = [function for function in functions if "mul_mat_vec_q_mma" in function.splitlines()[0]]
    if len(kernels) != 2:
        raise ValueError(f"expected two MMA78 SASS functions, found {len(kernels)}; inspect the CUDA library")
    for function in kernels:
        if re.search(r"\b(?:LDL|STL)(?:\.|\s)", function):
            raise ValueError("MMA78 local-memory instruction found; reduce live state and rebuild")
        if not re.search(r"\bIMMA\.", function):
            raise ValueError("MMA78 int8 tensor-core instruction missing; inspect the CUDA architecture")


if __name__ == "__main__":
    try:
        check(Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text())
    except (OSError, ValueError) as error:
        print(f"mma78 build gate: {error}", file=sys.stderr)
        sys.exit(8)
