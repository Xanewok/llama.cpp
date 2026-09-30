#!/usr/bin/env python3
"""Freeze compiled bytes; do not execute the server."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "0dba27b9002b8347c6f14a9f4a1651b3478065ef"


def run(*command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def main():
    source = ROOT / "llama.cpp-stack2"
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == COMMIT
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    frozen = ROOT / "frozen/stack2"
    if frozen.exists():
        raise SystemExit(f"refuse existing freeze: {frozen}")
    common = subprocess.check_output(["git", "-C", str(source), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True).strip()
    frozen.parent.mkdir(exist_ok=True)
    run("git", "clone", "--shared", "--no-checkout", common, str(frozen))
    run("git", "-C", str(frozen), "update-ref", "--no-deref", "HEAD", COMMIT)
    archive = ROOT / "build-tmp/gguf-py.tar"
    with archive.open("wb") as stream:
        run("git", "-C", str(source), "archive", COMMIT, "gguf-py", stdout=stream)
    with tarfile.open(archive) as stream:
        stream.extractall(frozen, filter="data")
    dest = frozen / "build/bin"
    dest.mkdir(parents=True)
    build = ROOT / "build-cuda/bin"
    for path in [build / "llama-server", *sorted(build.glob("*.so*"))]:
        shutil.copy2(path, dest / path.name, follow_symlinks=True)
        run("patchelf", "--set-rpath", "$ORIGIN", str(dest / path.name))
    # The campaign registry expects llama-server bytes, but exec uses this name.
    shutil.copy2(dest / "llama-server", dest / "weekend-srv")
    files = sorted(p for p in dest.iterdir() if p.is_file())
    (dest / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files))
    (frozen / "SOURCE").write_text(COMMIT + "\nstack/draft-vocab-host-vp; CUDA 13.2; NCCL OFF; arch 120 (CMake 120a); Release\n"
                                   "FA q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16; ccache OFF; nice 19; -j8\n")
    output = subprocess.check_output(["ldd", str(dest / "llama-server")], env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""}, text=True)
    (ROOT / "codex/frozen-ldd.log").write_text(output)
    relevant = [line for line in output.splitlines() if "libggml" in line or "libllama" in line]
    assert relevant and all(f"=> {dest}/" in line for line in relevant), output
    assert "not found" not in output, output
    print(f"frozen {COMMIT} in {frozen}; {len(files)} SHA256 pins; ldd resolves locally")


if __name__ == "__main__":
    main()
