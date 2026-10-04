"""Build isolated CPU mutants; never replace a production source or build object."""
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    os.environ["TMPDIR"] = str(ROOT / "build-tmp")
    entries = json.loads((ROOT / "build-cpu/compile_commands.json").read_text())
    directory = ROOT / "build-tmp/np2-mutants/wrong-stream"
    directory.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "src/llama-kv-cache.cpp").read_text()
    needle = "const auto & cells = v_cells.at(seq_to_stream[seq_id]);"
    assert source.count(needle) == 2
    first, last = source.rsplit(needle, 1)
    mutated = directory / "llama-kv-cache.cpp"
    mutated.write_text(first + "const auto & cells = v_cells.at(0);" + last)
    entry = next(e for e in entries if e["file"].endswith("/llama-kv-cache.cpp"))
    command = shlex.split(entry["command"])
    command[command.index("-o") + 1] = str(directory / "llama-kv-cache.cpp.o")
    command[-1] = str(mutated)
    subprocess.run(command, cwd=entry["directory"], check=True)
    command = shlex.split((ROOT / "build-cpu/src/CMakeFiles/llama.dir/link.txt").read_text())
    command[command.index("-o") + 1] = str(directory / "libllama.so.0")
    command = [str(directory / "llama-kv-cache.cpp.o") if arg == "CMakeFiles/llama.dir/llama-kv-cache.cpp.o" else arg for arg in command]
    subprocess.run(command, cwd=ROOT / "build-cpu/src", check=True)
    results = []
    for name, binary, args, library, expected in (
        ("wrong-stream", "test-kv-mask", [], directory, "full mask differs from cell oracle"),
        ("old-gate", "mask-np2-graph", [str(ROOT / "build-tmp/np2-mutants/old-gate.bin"), "1"],
         ROOT / "build-cpu-up/bin", "production multistream compact gate failed"),
    ):
        env = dict(os.environ, LD_LIBRARY_PATH=str(library), OMP_NUM_THREADS="4")
        result = subprocess.run([str(ROOT / "build-cpu/bin" / binary), *args], env=env, capture_output=True, text=True)
        (ROOT / "build-tmp/np2-mutants" / f"{name}.log").write_text(result.stdout + result.stderr)
        assert result.returncode != 0 and expected in result.stderr, f"mutant survived or failed for the wrong reason: {name}"
        results.append({"mutant": name, "rc": result.returncode, "expected_failure": expected})
    (ROOT / "build-tmp/np2-mutants/results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
