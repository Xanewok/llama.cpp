#!/usr/bin/env python3
import hashlib
import os
import pathlib
import shutil
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
root = pathlib.Path(__file__).resolve().parent.parent
sp = pathlib.Path("/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad")
up = sp / "queue/build/stack-gpu/resync-fmq"
subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=up / "bin", check=True)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
for arm, source, description in (
    ("up", up / "bin", (up / "SOURCE").read_text()),
    ("patch", root / "build-cuda/bin", revision + " topic/backend-verify-sampling; CUDA 13.2, NCCL off, sm_120\n"),
):
    dest = root / "codex/frozen" / arm / "bin"
    if dest.exists():
        raise RuntimeError(f"refuse to replace frozen build: {dest}")
    dest.mkdir(parents=True)
    for p in source.iterdir():
        if p.name != "llama-server" and ".so" not in p.name:
            continue
        target = dest / ("verify-ab" if p.name == "llama-server" else p.name)
        if p.is_symlink():
            target.symlink_to(p.readlink())
        else:
            shutil.copy2(p, target)
            subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(target)], check=True)
    (dest.parent / "SOURCE").write_text(description)
    entries = []
    for p in sorted(dest.iterdir()):
        if p.is_file() and not p.is_symlink():
            entries.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    (dest / "SHA256SUMS").write_text("".join(entries))
    subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=dest, check=True)
    print(f"frozen {arm}: {dest}")
