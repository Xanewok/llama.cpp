"""Freeze the unchanged server and relink its only changed production object from 98c695111."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    changed = subprocess.check_output(["git", "diff", "--name-only", "98c695111", "HEAD", "--", "src", "ggml", "common", "tools"]).decode().splitlines()
    assert changed == ["src/llama-graph.cpp"], changed
    os.environ["TMPDIR"] = str(ROOT / "build-tmp")
    dest = ROOT / "build-cpu-up/bin"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "build-cpu/bin/llama-server", dest / "cpu-serving")
    for path in (ROOT / "build-cpu/bin").glob("*.so*"):
        shutil.copy2(path, dest / path.name, follow_symlinks=True)
    source = ROOT / "build-tmp/llama-graph-up.cpp"
    source.write_bytes(subprocess.check_output(["git", "show", "98c695111:src/llama-graph.cpp"]))
    entries = json.loads((ROOT / "build-cpu/compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/llama-graph.cpp"))
    command = shlex.split(entry["command"])
    command[command.index("-o") + 1] = str(ROOT / "build-tmp/llama-graph-up.cpp.o")
    command[-1] = str(source)
    subprocess.run(command, cwd=entry["directory"], check=True)
    command = shlex.split((ROOT / "build-cpu/src/CMakeFiles/llama.dir/link.txt").read_text())
    name = Path(command[command.index("-o") + 1]).name
    command[command.index("-o") + 1] = str(dest / name)
    command = [str(ROOT / "build-tmp/llama-graph-up.cpp.o") if arg == "CMakeFiles/llama.dir/llama-graph.cpp.o" else arg for arg in command]
    subprocess.run(command, cwd=ROOT / "build-cpu/src", check=True)
    for path in dest.glob("libllama.so*"):
        if path.name != name:
            shutil.copyfile(dest / name, path)
    for path in dest.iterdir():
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(path)], check=True)
    (dest.parent / "SOURCE").write_bytes(subprocess.check_output(["git", "rev-parse", "98c695111"]))
    with (dest.parent / "SHA256SUMS").open("w") as output:
        subprocess.run(["sha256sum", *[f"bin/{path.name}" for path in sorted(dest.iterdir())]], cwd=dest.parent, stdout=output, check=True)


if __name__ == "__main__":
    main()
