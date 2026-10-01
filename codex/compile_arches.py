#!/usr/bin/env python3
"""Compile CUDA sources without executing device code. All outputs stay in this tree."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(__file__).resolve().parent.parent
out = root / "build-arch-check"
out.mkdir(exist_ok=True)
entries = json.loads((root / "build-cuda/compile_commands.json").read_text())
selected = [e for e in entries if e["file"].endswith("/fattn.cu") or "/fattn-mma-f16-instance-q8_0-" in e["file"]]
arches = (75, 120) if "--probe" in sys.argv else (75, 80, 86, 89, 90, 120)
if "--probe" in sys.argv:
    selected = [e for e in selected if any(s in e["file"] for s in ("ncols1_8-ncols2_1", "ncols1_2-ncols2_4", "ncols1_2-ncols2_8", "/fattn.cu"))]

def compile_one(arch, entry):
    tag = f"sm{arch}"
    cc = "120a" if arch == 120 else str(arch)
    command = shlex.split(entry["command"])
    command = [f"--generate-code=arch=compute_{cc},code=sm_{cc}" if x.startswith("--generate-code=") else x for x in command]
    name = Path(entry["file"]).stem
    command[command.index("-o") + 1] = str(out / f"{name}-{tag}.o")
    command.extend(("--threads=1", "-Werror=all-warnings", "--ptxas-options=-v"))
    log = root / "codex" / f"compile-{name}-{tag}.log"
    start = time.monotonic()
    with log.open("w") as stream:
        proc = subprocess.run(command, cwd=entry["directory"], stdout=stream, stderr=subprocess.STDOUT)
    result = {"arch": arch, "source": name, "seconds_M": time.monotonic() - start, "rc": proc.returncode, "command": command, "cwd": entry["directory"]}
    data = log.read_text()
    result["kernels"] = []
    for block in data.split("ptxas info    : Compiling entry function ")[1:]:
        symbol = block.split(chr(39))[1]
        if "flash_attn_ext_f16I" not in symbol:
            continue
        spills = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", block)
        regs = re.search(r"Used (\d+) registers", block)
        result["kernels"].append({"symbol": symbol, "registers_M": int(regs[1]), "stack_M": int(spills[1]), "spill_stores_M": int(spills[2]), "spill_loads_M": int(spills[3])})
    print(f"M: {tag} {name}: rc={proc.returncode}, {result['seconds_M']:.2f} s", flush=True)
    return result

jobs = [(a, e) for a in arches for e in selected]
previous = []
if "--retry" in sys.argv:
    previous = json.loads((root / "codex/scope-arches.json").read_text())
    failed = {(r["arch"], r["source"]) for r in previous if r["rc"]}
    jobs = [(a, e) for a, e in jobs if (a, Path(e["file"]).stem) in failed]
    previous = [r for r in previous if not r["rc"]]

with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("ARCH_CHECK_JOBS", "2"))) as pool:
    results = list(pool.map(lambda job: compile_one(*job), jobs))
results = previous + results
name = "probe" if "--probe" in sys.argv else "arches"
(root / "codex" / f"scope-{name}.json").write_text(json.dumps(results, indent=2) + "\n")
assert all(r["rc"] == 0 for r in results), "See compile logs"
