#!/usr/bin/env python3
"""Compile paired head-256 f16 controls and compare their device instructions."""
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
entries = json.loads((root / "build-cuda/compile_commands.json").read_text())
entry = next(e for e in entries if "instance-q8_0-ncols1_4-ncols2_8" in e["file"])
pairs = ((8, 1), (4, 2), (8, 2), (2, 4), (4, 4), (8, 4), (2, 8), (4, 8), (8, 8))


def normalize(symbol):
    if symbol.startswith("_Z18flash_attn_ext_f16I") and "L9ggml_type1E" in symbol:
        return symbol.replace("L9ggml_type1E", "").replace("PKcS2_S2_S2_S2_", "PKcS1_S1_S1_S1_")
    return symbol


def parse_sass(path):
    kernels = {}
    for block in path.read_text().split("Function : ")[1:]:
        symbol = normalize(block.splitlines()[0].strip())
        kernels[symbol] = [re.sub(r"^\s*/\*[0-9a-f]+\*/", "", line).strip() for line in block.splitlines()
                           if re.search(r"/\*[0-9a-f]+\*/", line) or re.search(r"/\* 0x[0-9a-f]+ \*/", line)]
    return kernels


def compile_one(job):
    arch, n1, n2, arm = job
    cc = "120a" if arch == 120 else str(arch)
    name = f"f16-{arm}-ncols1_{n1}-ncols2_{n2}-sm{arch}"
    source = out / f"{name}.cu"
    header = root / ("build-master-src/ggml/src/ggml-cuda/fattn-mma-f16.cuh" if arm == "master" else "ggml/src/ggml-cuda/fattn-mma-f16.cuh")
    source.write_text(f'#include "{header}"\nDECL_FATTN_MMA_F16_CASE(256, 256, {n1}, {n2});\n')
    cmd = shlex.split(entry["command"])
    cmd = [f"--generate-code=arch=compute_{cc},code=sm_{cc}" if x.startswith("--generate-code=") else x for x in cmd]
    cmd[cmd.index(entry["file"])] = str(source)
    obj = out / f"{name}.o"
    cmd[cmd.index("-o") + 1] = str(obj)
    cmd += ["--threads=1", "-Werror=all-warnings", "--ptxas-options=-v"]
    start = time.monotonic()
    log = root / "codex" / f"compile-{name}.log"
    with log.open("w") as stream:
        subprocess.run(cmd, cwd=entry["directory"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = {"arch": arch, "ncols1": n1, "ncols2": n2, "arm": arm, "seconds_M": time.monotonic()-start,
              "command": cmd, "cwd": entry["directory"], "kernels": []}
    for block in log.read_text().split("ptxas info    : Compiling entry function ")[1:]:
        symbol = block.split(chr(39))[1]
        if "flash_attn_ext_f16I" not in symbol:
            continue
        spills = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", block)
        regs = re.search(r"Used (\d+) registers", block)
        result["kernels"].append({"symbol": symbol, "registers_M": int(regs[1]), "stack_M": int(spills[1]),
                                  "spill_stores_M": int(spills[2]), "spill_loads_M": int(spills[3])})
    sass = root / "codex" / f"sass-{name}.txt"
    with sass.open("w") as stream:
        subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-sass", str(obj)], stdout=stream, check=True)
    result["sass"] = parse_sass(sass)
    print("M:", name, round(result["seconds_M"], 2), "s", flush=True)
    return result


jobs = [(arch, n1, n2, arm) for arch in (75, 80, 86, 89, 90, 120) for n1, n2 in pairs for arm in ("master", "scope")]
if "--analyze" in sys.argv:
    results = json.loads((root / "codex/f16-controls.json").read_text())["results"]
    for result in results:
        name = f"f16-{result['arm']}-ncols1_{result['ncols1']}-ncols2_{result['ncols2']}-sm{result['arch']}"
        result["sass"] = parse_sass(root / "codex" / f"sass-{name}.txt")
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compile_one, jobs))
differences = []
for master, scope in zip(results[::2], results[1::2]):
    canonical = lambda kernels: sorted([{**k, "symbol": normalize(k["symbol"])} for k in kernels], key=lambda k: k["symbol"])
    if canonical(master["kernels"]) != canonical(scope["kernels"]) or master["sass"] != scope["sass"]:
        differences.append((master["arch"], master["ncols1"], master["ncols2"]))
for result in results:
    del result["sass"]
(root / "codex/f16-controls.json").write_text(json.dumps({"results": results, "differences": differences}, indent=2) + "\n")
assert differences == [(75, 8, 8)], differences
print("M: all paired f16 ptxas stats match; active SASS matches. Inactive sm75 (8,8) embeds changed NO_DEVICE_CODE line numbers.")
