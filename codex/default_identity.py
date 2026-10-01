#!/usr/bin/env python3
"""Read CUDA objects only: resource stats for all kernels, SASS for f16 FA."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(__file__).resolve().parent.parent
profile = sys.argv[1] if len(sys.argv) > 1 else "single"
tool = "/usr/local/cuda-13.2/bin/cuobjdump"


def normalize(symbol):
    symbol = re.sub(r"(_INTERNAL_)[0-9a-f]{8}", r"\1HASHHASH", symbol)
    symbol = re.sub(r"(_cu_)[0-9a-f]{8}", r"\1HASHHASH", symbol)
    # The added template argument adds one Itanium substitution-table entry.
    if symbol.startswith("_Z18flash_attn_ext_f16I") and "L9ggml_type1E" in symbol:
        return symbol.replace("L9ggml_type1E", "").replace("PKcS2_S2_S2_S2_", "PKcS1_S1_S1_S1_")
    return symbol


def resources(path):
    result, raw = {}, set()
    arch = symbol = None
    for line in path.read_text().splitlines():
        match = re.search(r"arch = (sm_\w+)", line)
        if match:
            arch = match[1]
        match = re.search(r"Function (\S+):", line)
        if match:
            symbol = match[1]
            if symbol.startswith("_Z18flash_attn_ext_f16I") and "L9ggml_type8E" not in symbol:
                raw.add(symbol)
        if symbol and "REG:" in line:
            result.setdefault((arch, normalize(symbol)), set()).add(line.strip())
    return result, raw


def sass(path):
    result = {}
    arch = symbol = None
    body = []

    def save():
        if symbol:
            digest = hashlib.sha256("\n".join(body).encode()).hexdigest()
            result.setdefault((arch, normalize(symbol)), set()).add(digest)

    with path.open() as stream:
        for line in stream:
            match = re.search(r"arch = (sm_\w+)", line)
            if match:
                save()
                symbol, body, arch = None, [], match[1]
            match = re.search(r"Function : (\S+)", line)
            if match:
                save()
                symbol, body = match[1], []
            elif symbol and (re.search(r"/\*[0-9a-f]+\*/", line) or re.search(r"/\* 0x[0-9a-f]+ \*/", line)):
                instruction = re.sub(r"^\s*/\*[0-9a-f]+\*/", "", line).strip()
                if instruction:
                    body.append(instruction)
    save()
    return result


summary = {"profile": profile}
stats, instructions = [], []
for arm in ("master", "scope"):
    library = root / f"build-cost-{profile}-{arm}/bin/libggml-cuda.so"
    log = root / "codex" / f"identity-{profile}-{arm}-resource-usage.txt"
    with log.open("w") as stream:
        subprocess.run([tool, "--dump-resource-usage", str(library)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    resource, names = resources(log)
    stats.append(resource)
    log = root / "codex" / f"identity-{profile}-{arm}-sass.txt"
    with log.open("w") as stream:
        subprocess.run([tool, "--dump-sass", "--function", ",".join(sorted(names)), str(library)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    instructions.append(sass(log))
common = set(stats[0]) & set(stats[1])
summary["resources"] = {"common_M": len(common), "changed_M": [str(k) for k in common if stats[0][k] != stats[1][k]],
                        "missing_M": [str(k) for k in set(stats[0]) - set(stats[1])]}
common = set(instructions[0]) & set(instructions[1])
assert common, "No f16 FA instructions selected"
inactive = {k for k in common if all("REG:24 STACK:40" in value for value in stats[0][k])}
summary["sass"] = {"active_common_M": len(common-inactive), "inactive_diagnostic_kernels_M": len(inactive),
                   "changed_M": [str(k) for k in common-inactive if instructions[0][k] != instructions[1][k]],
                   "missing_M": [str(k) for k in set(instructions[0]) - set(instructions[1])]}
(root / "codex" / f"default-identity-{profile}.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
assert not summary["resources"]["changed_M"] and not summary["resources"]["missing_M"], "Resource stats changed"
assert not summary["sass"]["changed_M"] and not summary["sass"]["missing_M"], "Active f16 SASS changed"
