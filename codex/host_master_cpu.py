"""Apply each product commit independently to master, build, and reject an incorrect mutation."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "build-master-src"
BUILD = ROOT / "build-master-cpu"
LOG = ROOT / "codex/logs"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert subprocess.check_output(["git", "-C", str(SRC), "branch", "--show-current"], text=True).strip() == "topic/host-bundle-master"
assert subprocess.check_output(["git", "-C", str(SRC), "rev-parse", "--short=9", "HEAD"], text=True).strip() == "050439614"
commits = subprocess.check_output(["git", "rev-list", "--reverse", "98c695111..HEAD", "--", "src", "common", "tests", "tools/server"], cwd=ROOT, text=True).split()
assert len(commits) == 3
results = []
jobs = int(os.environ.get("HOST_MASTER_JOBS", "8"))
assert 1 <= jobs <= 8

def run(argv, label, expected=0):
    with (LOG / (label + ".log")).open("w") as stream:
        proc = subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if expected == 0:
        assert proc.returncode == 0, (label, proc.returncode)
    else:
        assert proc.returncode != 0, (label, "negative control escaped")
    return proc.returncode

for sha, name, target, test, path, old, new in zip(
        commits, ("kv", "prompt", "host-buffers"),
        ("test-kv-cells", "test-server-tokens-view", "test-arg-parser"),
        (["--bench"], ["--bench"], ["--tensor-buffers-only"]),
        ("src/llama-kv-cells.h", "tools/server/server-common.cpp", "common/arg.cpp"),
        ("first = std::min(first, i);", "    if (!has_mtmd) {\n        return tokens;\n    }\n    storage = get_text_tokens();", "        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);"),
        ("first = i;", "    if (false) {\n        return tokens;\n    }\n    storage = get_text_tokens();", "        auto * host_buft = (ggml_backend_buffer_type_t) nullptr;")):
    run(["git", "-C", SRC, "restore", "--source=HEAD", "--staged", "--worktree", "."], "master-" + name + "-restore")
    run(["git", "-C", SRC, "cherry-pick", "--no-commit", sha], "master-" + name + "-cherry-pick")
    run(["cmake", "-S", SRC, "-B", BUILD, "-DGGML_CUDA=OFF", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF",
         "-DGGML_OPENMP_FETCH=OFF", "-DGGML_CCACHE=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DCMAKE_BUILD_TYPE=Release"], "master-" + name + "-configure")
    run(["cmake", "--build", BUILD, "--target", target, "test-alloc", f"-j{jobs}"], "master-" + name + "-build")
    run([BUILD / "bin" / target, *test], "master-" + name + "-test")
    run([BUILD / "bin/test-alloc"], "master-" + name + "-alloc")
    source = SRC / path
    original = source.read_text()
    assert old in original
    source.write_text(original.replace(old, new))
    try:
        run(["cmake", "--build", BUILD, "--target", target, f"-j{jobs}"], "master-" + name + "-mutant-build")
        rc = run([BUILD / "bin" / target, *test], "master-" + name + "-mutant", expected=1)
    finally:
        source.write_text(original)
    results.append({"commit": sha, "fix": name, "standalone_cherry_pick": "PASS", "test": target,
                    "positive_M": 0, "mutant_exit_M": rc, "mutation": [old, new]})
    (ROOT / "codex/host-results/master.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS M: {name}; standalone master cherry-pick, positive test, negative mutation", flush=True)
run(["git", "-C", SRC, "restore", "--source=HEAD", "--staged", "--worktree", "."], "master-final-restore")
