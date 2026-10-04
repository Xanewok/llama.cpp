#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
os.nice(19)
root = Path(__file__).resolve().parent.parent
os.chdir(root)
os.environ["TMPDIR"] = str(root / "build-tmp")
Path("build-tmp").mkdir(exist_ok=True)


def timings(path):
    rows = {}
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "M":
            rows[(fields[1], int(fields[2]), int(fields[3]), fields[4])] = float(fields[5])
    return rows


def identical(a, b):
    size = 0
    digest = hashlib.sha256()
    with a.open("rb") as x, b.open("rb") as y:
        while True:
            u, v = x.read(1024*1024), y.read(1024*1024)
            if u != v:
                raise AssertionError(f"byte identity failed: {a}, {b}")
            if not u:
                break
            size += len(u); digest.update(u)
    if not size:
        raise AssertionError("empty identity artifact")
    return {"bytes_M": size, "sha256": digest.hexdigest()}


result = {"masks": identical(Path("build-ref/masks.bin"), Path("build-cpu/patch-masks.bin")),
          "logits": identical(Path("build-ref/logits.bin"), Path("build-cpu/patch-logits.bin"))}
logit_bytes = Path("build-cpu/patch-logits.bin").read_bytes()
values = struct.unpack("=" + "f"*(len(logit_bytes)//4), logit_bytes)
if min(values) == max(values) or not any(values):
    raise AssertionError("degenerate logit fixture")
result["logits"]["min_M"] = min(values)
result["logits"]["max_M"] = max(values)
for name in ("base", "patch"):
    if "PASS: negative controls, target and MTP logits finite at both depths" not in Path(f"codex/{name}-cpu.csv").read_text():
        raise AssertionError("incomplete fixture run")
with Path("build-cpu/stale.csv").open("w") as log:
    negative = subprocess.run(["build-cpu/host-depth", "build-cpu/stale-masks.bin", "build-cpu/stale-logits.bin"],
                              env=dict(os.environ, HOST_DEPTH_STALE_CONTROL="1"), stdout=log, stderr=subprocess.PIPE, text=True, timeout=60)
if negative.returncode != 1 or "negative control: sequence mutation retained stale mask" not in negative.stderr:
    raise AssertionError("stale mask escaped negative control")
result["negative_control"] = {"exit_M": negative.returncode, "error": negative.stderr.strip()}
broken_error = Path("build-cpu/broken-cpu.log").read_text().strip()
if broken_error != "negative control: sequence mutation retained stale mask":
    raise AssertionError("production guard mutation was not detected")
if "cells.used_max_p1() == n_used && positions.size() == n_used" not in Path("src/llama-kv-cache.cpp").read_text():
    raise AssertionError("production guard was not restored")
result["production_guard_negative_control"] = {"exit_M": 1, "error": broken_error,
    "mutation": "remove positions.size() == n_used from the dense-prefix guard"}
base, patch = timings(Path("codex/base-cpu.csv")), timings(Path("codex/patch-cpu.csv"))
result["rounds"] = []
for depth in (8192, 90112):
    for cache in ("target", "draft"):
        result["rounds"].append({"cache": cache, "depth": depth,
            "base_us_M": base[(cache, depth, 11, "mask_round_f16")],
            "patch_us_M": patch[(cache, depth, 11, "mask_round_f16")],
            "base_us_D": base[(cache, depth, 6, "mask_f16")] + 5*base[(cache, depth, 1, "mask_f16")],
            "patch_us_D": patch[(cache, depth, 6, "mask_f16")] + 5*patch[(cache, depth, 1, "mask_f16")]})
    before = base[("target", depth, 6, "mask_f16")] + 5*base[("draft", depth, 1, "mask_f16")]
    after = patch[("target", depth, 6, "mask_f16")] + 5*patch[("draft", depth, 1, "mask_f16")]
    result["rounds"].append({"cache": "target-verify-plus-five-drafts", "depth": depth,
        "base_us_D": before, "patch_us_D": after, "saving_ms_D": (before - after)/1000})
Path("codex/cpu-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
