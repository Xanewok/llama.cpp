#!/usr/bin/env python3
"""Measure touched CUDA units and relink each stage against identical common objects."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(__file__).resolve().parent.parent
env = dict(os.environ, GIT_DIR=str(root / "codex/git"), GIT_WORK_TREE=str(root), GIT_NO_LAZY_FETCH="1")
stages = (("initial", "6935009fc"), ("turing", "6a9cbe0d9"), ("rows", "554e0cbd4"), ("gqa", "b5a3a6a7a"))
out = root / "build-commit-costs"
out.mkdir(exist_ok=True)
results = []
for index, (name, revision) in enumerate(stages):
    src = out / f"src-{index:03d}"
    src.mkdir(exist_ok=True)
    archive = out / f"src-{index:03d}.tar"
    subprocess.run(["git", "-c", "protocol.allow=never", "archive", "-o", str(archive), revision, "ggml"], env=env, check=True)
    with tarfile.open(archive) as tf:
        tf.extractall(src, filter="data")

for profile in ("single", "default", "all-quants"):
    build = root / f"build-cost-{profile}-scope"
    # The full build and this one-worker audit together never exceed eight compiler workers.
    while not (build / "bin/libggml-cuda.so").exists():
        time.sleep(10)
    entries = json.loads((build / "compile_commands.json").read_text())
    cuda_dir = build / "ggml/src/ggml-cuda"
    objects = shlex.split((cuda_dir / "CMakeFiles/ggml-cuda.dir/objects1.rsp").read_text())
    link = shlex.split((cuda_dir / "CMakeFiles/ggml-cuda.dir/link.txt").read_text())
    for index, (name, revision) in enumerate(stages):
        src = out / f"src-{index:03d}"
        stage = out / f"{profile}-{index:03d}"
        stage.mkdir(exist_ok=True)
        compiled, units = {}, []
        for entry in entries:
            relative = Path(entry["file"]).relative_to(root)
            if relative.name != "fattn.cu" and "fattn-mma-f16-instance-q8_0-" not in relative.name:
                continue
            source = src / relative
            if not source.exists():
                continue
            cmd = shlex.split(entry["command"])
            cmd[cmd.index(entry["file"])] = str(source)
            obj = stage / (relative.name + ".o")
            cmd[cmd.index("-o") + 1] = str(obj)
            cmd += ["--ptxas-options=-v"]
            start = time.monotonic()
            log = root / "codex" / f"stage-{profile}-{name}-{relative.stem}.log"
            with log.open("w") as stream:
                subprocess.run(cmd, cwd=entry["directory"], stdout=stream, stderr=subprocess.STDOUT, check=True)
            units.append({"source": relative.name, "seconds_M": time.monotonic()-start, "object_bytes_M": obj.stat().st_size,
                          "command": cmd, "cwd": entry["directory"]})
            compiled[relative.name] = str(obj)
            print("M:", profile, name, relative.name, round(units[-1]["seconds_M"], 2), "s", flush=True)
        common = []
        for obj in objects:
            filename = Path(obj).name.removesuffix(".o")
            if filename == "fattn.cu" or "fattn-mma-f16-instance-q8_0-" in filename:
                if filename in compiled:
                    common.append(compiled[filename])
            else:
                common.append(str((cuda_dir / obj).resolve()))
        rsp = stage / "objects.rsp"
        rsp.write_text("\n".join(shlex.quote(p) for p in common) + "\n")
        cmd = link[:]
        cmd[cmd.index("-o")+1] = str(stage / "libggml-cuda.so")
        cmd = ["@" + str(rsp) if x.endswith("objects1.rsp") else x for x in cmd]
        start = time.monotonic()
        with (root / "codex" / f"stage-{profile}-{name}-link.log").open("w") as stream:
            subprocess.run(cmd, cwd=cuda_dir, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = {"profile": profile, "stage": name, "revision": revision, "units": units,
                  "units_seconds_sum_D": sum(u["seconds_M"] for u in units), "link_seconds_M": time.monotonic()-start,
                  "library_bytes_M": (stage / "libggml-cuda.so").stat().st_size, "link_command": cmd}
        results.append(result)
        (root / "codex/commit-costs.json").write_text(json.dumps(results, indent=2)+"\n")
        print("M:", profile, name, "library", result["library_bytes_M"], flush=True)
