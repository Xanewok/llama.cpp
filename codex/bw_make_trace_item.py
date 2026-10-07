"""Render the standard depth item with a local wrapper for the writer-gap gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path("/home/xanewok/repos/qwen38-eval")
FROZEN = Path("/home/xanewok/repos/qwen38-eval-worktrees/frozen/trace-writer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("evidence", type=Path)
    p.add_argument("--prepare", action="store_true", help="Contract-only draft before the build; never submit it")
    a = p.parse_args()
    sys.path.insert(0, str(EVAL))
    from harness import lib, mkitem
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GIT_NO_LAZY_FETCH="1")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, env=env, text=True).strip()
    if not a.prepare:
        if FROZEN.joinpath("SOURCE").read_text().splitlines()[0] != sha:
            raise ValueError("freeze SOURCE differs from HEAD")
        ev = json.loads(a.evidence.read_text())
        if ev.get("commit") != sha:
            raise ValueError("CPU evidence commit differs from freeze")
        if ev.get("byte_identity") != "old == sync == async" or ev.get("async_over_sync_D", 1) >= 0.2:
            raise ValueError("CPU evidence did not pass")
    text = (ROOT / "codex/trace-writer-depth-j5q.toml.in").read_text()
    text = text.replace("@COMMIT@", sha).replace("@CPU_EVIDENCE@", str(a.evidence.resolve()))
    experiment = ROOT / "codex/items/trace-writer-depth-j5q.toml"
    experiment.write_text(text)
    x = lib.read_experiment(experiment)
    fixtures = EVAL / "orchestration/gpuq/fixtures"
    extra = sorted(p for p in fixtures.rglob("*") if p.is_file())
    extra += sorted(p for p in (ROOT / "codex").iterdir() if p.suffix in (".py", ".sh", ".in") and p.is_file())
    extra += [ROOT / "codex/test-round-trace.cpp", ROOT / "codex/trace-test/CMakeLists.txt",
              EVAL / "results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/round-trace.jsonl"]
    # The prepared item pins the future freeze anchors. After the window, mkitem also pins every actual frozen file.
    extra += [FROZEN / "SOURCE", FROZEN / "build/bin/SHA256SUMS", FROZEN / "build/bin/round-trace"]
    x["extra_pins"] = [str(p) for p in extra]
    rendered = mkitem.render(x)
    standard = json.dumps(["bash", str(EVAL / "harness/run.sh"), "depth"])[0:-1] + ","
    wrapper = json.dumps(["bash", str(ROOT / "codex/gpu_ab.sh")])[0:-1] + ","
    if rendered.count(standard) != 3:
        raise ValueError("mkitem stage layout changed")
    rendered = rendered.replace(standard, wrapper)
    if a.prepare:
        rendered = "# PREPARED ONLY: rebuild-window must regenerate this after CPU proof and freeze.\n" + rendered
    item = experiment.with_name("trace-writer-depth-j5q.gpuq.toml")
    item.write_text(rendered)
    tomllib.loads(rendered)
    print(item)


if __name__ == "__main__":
    main()
