#!/usr/bin/env python3
"""Verify CPU analysis and the parent benchmark's guard/summary contracts."""
import ast
import importlib.util
import json
import os
import pathlib
import subprocess
import tempfile
import time


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    os.chdir(root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES empty")
    checks = []
    for path in pathlib.Path("codex").glob("*.py"):
        ast.parse(path.read_text())
        checks.append(f"PASS AST {path}")
    subprocess.run(["bash", "-n", "codex/gpu_ab.sh"], check=True)
    checks.append("PASS bash -n codex/gpu_ab.sh")
    started = time.perf_counter()
    subprocess.run(["python3", "codex/analyze.py"], check=True, stdout=subprocess.DEVNULL)
    checks.append(f"M analyzer wall_ms={(time.perf_counter()-started)*1000:.3f}, CUDA_VISIBLE_DEVICES empty")
    subprocess.run(["bash", "codex/gpu_ab.sh"], check=True, stdout=subprocess.DEVNULL)
    checks.append("PASS CPU-only GPU A/B dry plan")
    guarded = subprocess.run(["python3", "codex/marlin_bench.py", "--execute"], capture_output=True, text=True)
    assert guarded.returncode == 2 and "requires PARENT_GPU" in guarded.stderr
    checks.append("PASS --execute rejects empty GPU authorization before torch import")
    spec = importlib.util.spec_from_file_location("summary", root / "codex/summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shapes = json.loads(pathlib.Path("codex/evidence/shapes.json").read_text())
    with tempfile.TemporaryDirectory(dir=root / "codex") as temporary:
        folder = pathlib.Path(temporary)
        for boot, arm in enumerate(["up", "patch", "patch", "up", "up", "patch"], 1):
            for device in [0, 1]:
                rows = []
                for shape in shapes:
                    for variant in ["w4g128", "w4g32", "w4g64", "w8channel"]:
                        for m in range(1, 7):
                            for cache in ["warm", "cold"]:
                                rows.append(dict(kind="latency", shape=shape["name"], variant=variant,
                                                 m=m, cache=cache, p50_us=10 if arm == "up" else 8,
                                                 output_sha256="same"))
                path = folder / f"{boot}-{arm}-d{device}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = module.summarize(folder)
        assert len(result) == 576 and all(row["delta_us_D"] == -2 for row in result)
        checks.append("PASS synthetic full-matrix A/B recovers -2 us_D")
        path = folder / "2-patch-d0.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[0])
        row["output_sha256"] = "corrupted"
        lines[0] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n")
        try:
            module.summarize(folder)
        except ValueError as error:
            assert "exactness gate failed" in str(error)
        else:
            raise AssertionError("negative control accepted corrupted output")
        checks.append("PASS negative control rejects mismatched up/patch output")
        path.unlink()
        try:
            module.summarize(folder)
        except ValueError as error:
            assert "six boots" in str(error)
        else:
            raise AssertionError("accepted missing boot")
        checks.append("PASS incomplete boot set rejected")
    pathlib.Path("codex/evidence/cpu-checks.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks))


if __name__ == "__main__":
    main()
